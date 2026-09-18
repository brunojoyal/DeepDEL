#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, contextvars, csv, itertools, logging, sys, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Sequence, Optional

import pandas as pd

from deepdelgfn.mols.dels import (
    TrimerBuilder,
    REACTION_MODE_AMIDE_SULFONAMIDE,
    REACTION_MODE_AMIDE_AMIDE,
    VALID_REACTION_MODES,
)
from deepdelgfn.mols.vina_scorer import (
    DockingScorer,
    Dock3Scorer,
    DEFAULT_4JNC_BOX,
    box_for_target,
)

# ----------------------------- generate-only (no scoring) -------------------

_RDKIT_BUILD_CONTEXT: contextvars.ContextVar[Optional[Dict[str, object]]] = contextvars.ContextVar(
    "deepdelgfn_rdkit_build_context",
    default=None,
)


class _RDKitBuildContextHandler(logging.Handler):
    """Emit RDKit warnings/errors with the active trimer-build context.

    RDKit warnings are otherwise written as bare lines such as
    ``Conflicting single bond directions...``.  During large random-library
    generation those lines are hard to connect back to the trimer being built.
    This handler adds the current BB triple to RDKit log records while keeping
    normal RDKit text intact.
    """

    _HANDLER_MARKER = "_deepdelgfn_rdkit_context_handler"

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        setattr(self, self._HANDLER_MARKER, True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            ctx = _RDKIT_BUILD_CONTEXT.get()
            if ctx:
                parts = [f"mode={ctx.get('mode')}", f"trimer=({ctx.get('bb1_id')},{ctx.get('bb2_id')},{ctx.get('bb3_id')})"]
                if ctx.get("sample_index") is not None:
                    parts.insert(1, f"sample={ctx.get('sample_index')}")
                if ctx.get("row_index") is not None:
                    parts.insert(1, f"row={ctx.get('row_index')}")
                msg = (
                    f"[rdkit-context {' '.join(parts)}]\n"
                    f"  BB1 SMILES: {ctx.get('bb1_smiles')}\n"
                    f"  BB2 SMILES: {ctx.get('bb2_smiles')}\n"
                    f"  BB3 SMILES: {ctx.get('bb3_smiles')}\n"
                    f"  RDKit: {msg}"
                )
            else:
                msg = f"[rdkit] {msg}"
            print(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def install_rdkit_context_logger() -> None:
    """Route RDKit logs through Python logging and add trimer context.

    This is best-effort: if this RDKit build lacks ``LogToPythonLogger`` we keep
    the original RDKit stderr behavior rather than failing the job.
    """
    try:
        from rdkit import rdBase

        rdBase.LogToPythonLogger()
    except Exception as e:
        print(
            f"[warn] Could not attach RDKit context logger; RDKit warnings will not include trimer context: {e}",
            file=sys.stderr,
        )
        return

    logger = logging.getLogger("rdkit")
    if not any(getattr(h, _RDKitBuildContextHandler._HANDLER_MARKER, False) for h in logger.handlers):
        logger.addHandler(_RDKitBuildContextHandler())
    logger.setLevel(logging.WARNING)
    # Avoid duplicate output via root handlers after LogToPythonLogger().
    logger.propagate = False


@contextlib.contextmanager
def trimer_build_context(
    *,
    mode: str,
    bb1_id: int,
    bb2_id: int,
    bb3_id: int,
    bb1_smiles: str,
    bb2_smiles: str,
    bb3_smiles: str,
    sample_index: Optional[int] = None,
    row_index: Optional[int] = None,
):
    token = _RDKIT_BUILD_CONTEXT.set(
        {
            "mode": mode,
            "bb1_id": bb1_id,
            "bb2_id": bb2_id,
            "bb3_id": bb3_id,
            "bb1_smiles": bb1_smiles,
            "bb2_smiles": bb2_smiles,
            "bb3_smiles": bb3_smiles,
            "sample_index": sample_index,
            "row_index": row_index,
        }
    )
    try:
        yield
    finally:
        _RDKIT_BUILD_CONTEXT.reset(token)

def generate_blockset_to_csv(
    B1: Sequence[int],
    B2: Sequence[int],
    B3: Sequence[int],
    *,
    id2smi1: Dict[int, str],
    id2smi2: Dict[int, str],
    id2smi3: Dict[int, str],
    builder: "TrimerBuilder",
    out_path: Path,
    verbose: bool,
) -> Path:
    """
    Enumerate B1 x B2 x B3, build each trimer SMILES, and write an **unscored**
    CSV (no docking).  The output has the same schema as score_blockset_to_csv
    but without a score column — ready to be fed to dock_library.py.
    """
    combos = list(itertools.product(B1, B2, B3))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, len(combos) // 10) if len(combos) >= 10 else 1

    with out_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow(["bb1_id", "bb2_id", "bb3_id",
                         "bb1_smiles", "bb2_smiles", "bb3_smiles",
                         "smiles"])
        for idx, (i, j, k) in enumerate(combos, 1):
            s1, s2, s3 = id2smi1[i], id2smi2[j], id2smi3[k]
            try:
                with trimer_build_context(
                    mode="generate-blockset",
                    bb1_id=i,
                    bb2_id=j,
                    bb3_id=k,
                    bb1_smiles=s1,
                    bb2_smiles=s2,
                    bb3_smiles=s3,
                    sample_index=idx,
                ):
                    rec = builder.build_trimer_from_smiles(s1, s2, s3,
                                                           reaction_mode=builder.reaction_mode)
                trimer_smi = rec.smi
            except Exception as e:
                print(
                    f"[warn] build failed ({i},{j},{k}): {e}\n"
                    f"  BB1 SMILES: {s1}\n"
                    f"  BB2 SMILES: {s2}\n"
                    f"  BB3 SMILES: {s3}",
                    file=sys.stderr,
                )
                trimer_smi = ""
            writer.writerow([i, j, k, s1, s2, s3, trimer_smi])
            if verbose and (idx % step == 0 or idx == len(combos)):
                print(f"[progress] {idx}/{len(combos)}")

    print(f"[done] wrote {len(combos)} rows to {out_path.resolve()}")
    return out_path


def generate_random_to_csv(
    triples: List[Tuple[int, int, int]],
    *,
    id2smi1: Dict[int, str],
    id2smi2: Dict[int, str],
    id2smi3: Dict[int, str],
    builder: "TrimerBuilder",
    out_path: Path,
    verbose: bool,
    reaction_mode: str,
) -> Path:
    """Like generate_blockset_to_csv but for a pre-sampled list of triples."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, len(triples) // 10) if len(triples) >= 10 else 1

    with out_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow(["bb1_id", "bb2_id", "bb3_id",
                         "bb1_smiles", "bb2_smiles", "bb3_smiles",
                         "smiles"])
        for idx, (i, j, k) in enumerate(triples, 1):
            s1, s2, s3 = id2smi1[i], id2smi2[j], id2smi3[k]
            try:
                with trimer_build_context(
                    mode="generate-random",
                    bb1_id=i,
                    bb2_id=j,
                    bb3_id=k,
                    bb1_smiles=s1,
                    bb2_smiles=s2,
                    bb3_smiles=s3,
                    sample_index=idx,
                ):
                    rec = builder.build_trimer_from_smiles(s1, s2, s3,
                                                           reaction_mode=reaction_mode)
                trimer_smi = rec.smi
            except Exception as e:
                print(
                    f"[warn] build failed ({i},{j},{k}): {e}\n"
                    f"  BB1 SMILES: {s1}\n"
                    f"  BB2 SMILES: {s2}\n"
                    f"  BB3 SMILES: {s3}",
                    file=sys.stderr,
                )
                trimer_smi = ""
            writer.writerow([i, j, k, s1, s2, s3, trimer_smi])
            if verbose and (idx % step == 0 or idx == len(triples)):
                print(f"[progress] {idx}/{len(triples)}")

    print(f"[done] wrote {len(triples)} rows to {out_path.resolve()}")
    return out_path

# ----------------------------- helpers -----------------------------

def parse_block_spec(spec: str) -> Tuple[List[int], List[int], List[int]]:
    blocks = [blk.strip() for blk in spec.split(",")]
    if len(blocks) != 3:
        raise ValueError(f"--blocks must have exactly 3 comma-separated blocks; got {len(blocks)}")
    def to_ints(blk: str) -> List[int]:
        toks = [t.strip() for t in blk.split("|") if t.strip()]
        return [int(t) for t in toks]
    B1, B2, B3 = (to_ints(b) for b in blocks)
    if not B1 or not B2 or not B3:
        raise ValueError("Each block must list at least one ID (e.g., '1|2,3|4,5|6').")
    return B1, B2, B3

def parse_pipe_ids(pipe_str: str, label: str) -> List[int]:
    if pd.isna(pipe_str):
        raise ValueError(f"{label}: value is missing/NaN")
    toks = [t.strip() for t in str(pipe_str).split("|") if t.strip()]
    if not toks:
        raise ValueError(f"{label}: must contain at least one integer ID")
    try:
        return [int(t) for t in toks]
    except Exception:
        raise ValueError(f"{label}: all tokens must be integers; got {pipe_str!r}")

def parse_random_block_sizes(spec: str) -> Tuple[int, int, int]:
    toks = [t.strip() for t in spec.split("|") if t.strip()]
    if len(toks) != 3:
        raise ValueError(f"--random-block must be exactly 'n1|n2|n3' (e.g., '6|6|6'); got {spec!r}")
    try:
        n1, n2, n3 = (int(t) for t in toks)
    except Exception:
        raise ValueError(f"--random-block sizes must be integers; got {spec!r}")
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        raise ValueError("--random-block sizes must be positive integers.")
    return n1, n2, n3

def _resolve_smiles_col(df: pd.DataFrame) -> str:
    """Return the name of the SMILES column (handles 'SMILES', 'Smiles', 'smiles')."""
    for col in ("SMILES", "Smiles", "smiles"):
        if col in df.columns:
            return col
    raise ValueError("Pool CSV must contain a SMILES-like column ('SMILES', 'Smiles', or 'smiles').")

def make_id_to_smiles(df: pd.DataFrame, id_col: str) -> Dict[int, str]:
    smiles_col = _resolve_smiles_col(df)
    if id_col in df.columns:
        if df[id_col].duplicated().any():
            dups = df.loc[df[id_col].duplicated(), id_col].tolist()
            raise ValueError(f"Duplicate IDs in column '{id_col}': {dups[:10]} ...")
        return {int(r[id_col]): str(r[smiles_col]) for _, r in df[[id_col, smiles_col]].iterrows()}
    # Fallback: use dataframe index as ID
    return {int(i): str(smi) for i, smi in zip(df.index, df[smiles_col])}

def preflight_ids_present(all_ids: Iterable[int], id2smi: Dict[int, str], label: str) -> None:
    missing = [x for x in all_ids if x not in id2smi]
    if missing:
        raise SystemExit(
            f"[preflight] {label}: {len(missing)} IDs not found in pool: {missing[:20]}{' ...' if len(missing)>20 else ''}\n"
            "Hint: set --id-col to the correct column, or ensure your CSV has an 'ID' column or uses integer index."
        )

def resolve_engine(args) -> str:
    engine_cli = getattr(args, "engine", None) or "vina"
    engine_path_cli = getattr(args, "engine_path", None) or "vina"
    if engine_cli != engine_path_cli and engine_cli != "vina" and engine_path_cli != "vina":
        print(f"[warn] --engine ('{engine_cli}') != --engine-path ('{engine_path_cli}'); using --engine-path.",
              file=sys.stderr)
    return engine_path_cli if engine_path_cli != "vina" else engine_cli

# map linear index -> (i,j,k) without materializing all triples
def index_to_triple(idx: int, B1: Sequence[int], B2: Sequence[int], B3: Sequence[int]) -> Tuple[int,int,int]:
    n1, n2, n3 = len(B1), len(B2), len(B3)
    ij_size = n2 * n3
    i = idx // ij_size
    rem = idx % ij_size
    j = rem // n3
    k = rem % n3
    return B1[i], B2[j], B3[k]

# find the next N.csv in a directory (1.csv, 2.csv, ...)
_DIGIT_RE = re.compile(r"^(\d+)\.csv$")

def next_csv_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in out_dir.glob("*.csv"):
        m = _DIGIT_RE.match(p.name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass
    return out_dir / f"{max_n + 1}.csv"

# -------------------- pool loading helpers --------------------

def load_pool_df(path: str) -> pd.DataFrame:
    """Load a pool CSV and normalise the SMILES column to 'SMILES'."""
    df = pd.read_csv(path)
    smiles_col = _resolve_smiles_col(df)
    if smiles_col != "SMILES":
        df = df.rename(columns={smiles_col: "SMILES"})
    return df

def load_combined_pool_df(
    path: str,
    *,
    pool_col: str = "pool",
    id_col: str = "ID",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a combined BBs CSV (with a `pool` column tagging origin 1/2/3) and split it.

    Returns ``(combined, df1, df2, df3)`` where:
      - ``combined`` is the full frame with a normalised ``SMILES`` column,
      - ``df1/df2/df3`` are the per-pool slices (rows where ``pool == 1/2/3``).

    The combined CSV is the artifact produced by
    ``scripts/active_learning_submit._build_combined_bbs_csv`` and uses globally
    unique IDs across all three pools. Splitting by the ``pool`` column lets
    ``TrimerBuilder`` see slot-specific chemistry (amines vs. sulfonyl chlorides)
    while a single global id→SMILES map is used for all three slots.
    """
    combined = load_pool_df(path)
    if pool_col not in combined.columns:
        raise SystemExit(
            f"[error] --bbs-combined '{path}' is missing required column "
            f"'{pool_col}'. Expected a combined BBs CSV produced by "
            f"scripts/active_learning_submit.py with a `pool` column tagging "
            f"origin 1/2/3."
        )
    if id_col not in combined.columns:
        raise SystemExit(
            f"[error] --bbs-combined '{path}' is missing required ID column "
            f"'{id_col}'. Use --id-col to override."
        )
    # Normalise pool column to int.
    try:
        combined[pool_col] = pd.to_numeric(combined[pool_col], errors="raise").astype(int)
    except Exception as e:
        raise SystemExit(f"[error] --bbs-combined '{path}' has non-integer values in '{pool_col}': {e}")

    pool_values = set(combined[pool_col].unique().tolist())
    missing = {1, 2, 3} - pool_values
    if missing:
        raise SystemExit(
            f"[error] --bbs-combined '{path}' is missing pool tag(s) {sorted(missing)} in '{pool_col}'. "
            f"Found values: {sorted(pool_values)}."
        )

    df1 = combined[combined[pool_col] == 1].reset_index(drop=True)
    df2 = combined[combined[pool_col] == 2].reset_index(drop=True)
    df3 = combined[combined[pool_col] == 3].reset_index(drop=True)
    return combined, df1, df2, df3


def build_pools(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Optional[pd.DataFrame]]:
    """
    Return (df1, df2, df3, mode, combined_df) based on CLI args.

    Two modes are supported:

    * ``"per_pool"`` (default, legacy): each slot is loaded from its own CSV via
      ``--pool1/--pool2/--pool3`` (or the ``--pool`` shorthand) and IDs are
      interpreted in each pool's local space.
    * ``"combined"``: a single combined BBs CSV (``--bbs-combined``) with a
      ``pool`` column tagging origin (1/2/3) and globally unique IDs is loaded
      and split into per-pool dataframes. ``combined_df`` is returned so the
      caller can build a single global id→SMILES map shared by all three slots.

    The two modes are mutually exclusive.
    """
    bbs_combined = getattr(args, "bbs_combined", None)
    pool_fallback = getattr(args, "pool", None)
    p1 = getattr(args, "pool1", None)
    p2 = getattr(args, "pool2", None)
    p3 = getattr(args, "pool3", None)

    reaction_mode = getattr(args, "reaction_mode", REACTION_MODE_AMIDE_SULFONAMIDE)

    if bbs_combined is not None:
        if any(x is not None for x in (pool_fallback, p1, p2, p3)):
            raise SystemExit(
                "[error] --bbs-combined is mutually exclusive with --pool / --pool1 / --pool2 / --pool3."
            )
        pool_col = getattr(args, "pool_col", None) or "pool"
        id_col = getattr(args, "id_col", None) or "ID"
        try:
            combined, df1, df2, df3 = load_combined_pool_df(
                bbs_combined, pool_col=pool_col, id_col=id_col
            )
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"[error] reading --bbs-combined '{bbs_combined}': {e}")
        return df1, df2, df3, "combined", combined

    # ---- Legacy per-pool mode ----
    p1 = p1 or pool_fallback
    p2 = p2 or pool_fallback
    p3 = p3 or pool_fallback

    if p1 is None:
        raise SystemExit("[error] --pool1 (or --pool, or --bbs-combined) is required.")
    if p2 is None:
        raise SystemExit("[error] --pool2 (or --pool, or --bbs-combined) is required.")
    if p3 is None:
        if reaction_mode == REACTION_MODE_AMIDE_SULFONAMIDE:
            raise SystemExit(
                "[error] --pool3 is required for --reaction-mode amide_sulfonamide "
                "(pool3 should be the sulfonyl chloride CSV)."
            )
        p3 = p1  # amide_amide: default to pool1 if not specified

    try:
        df1 = load_pool_df(p1)
    except Exception as e:
        raise SystemExit(f"[error] reading --pool1 '{p1}': {e}")
    try:
        df2 = load_pool_df(p2)
    except Exception as e:
        raise SystemExit(f"[error] reading --pool2 '{p2}': {e}")
    try:
        df3 = load_pool_df(p3)
    except Exception as e:
        raise SystemExit(f"[error] reading --pool3 '{p3}': {e}")

    return df1, df2, df3, "per_pool", None


# ------------------------- core scoring logic -------------------------

def score_blockset_to_csv(
    B1: Sequence[int],
    B2: Sequence[int],
    B3: Sequence[int],
    *,
    id2smi1: Dict[int, str],
    id2smi2: Dict[int, str],
    id2smi3: Dict[int, str],
    builder: TrimerBuilder,
    scorer: DockingScorer,
    out_dir: Path,
    out_csv: Optional[Path] = None,
    jobs: int,
    flush_every: int,
    fsync_every: int,
    verbose: bool,
    extra_prefix_cols: Optional[Tuple[str, ...]] = None,
) -> Path:
    """
    Enumerate B1 x B2 x B3, score, and write a CSV to the next ordinal filename in out_dir.
    Optionally prepend extra metadata columns (values supplied via extra_prefix_cols tuple).
    """
    combos = list(itertools.product(B1, B2, B3))
    out_path = out_csv if out_csv is not None else next_csv_path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def work(triple: Tuple[int,int,int]) -> Tuple[int,int,int,str, str, str, str, float]:
        i, j, k = triple
        s1, s2, s3 = id2smi1[i], id2smi2[j], id2smi3[k]
        with trimer_build_context(
            mode="score-blockset",
            bb1_id=i,
            bb2_id=j,
            bb3_id=k,
            bb1_smiles=s1,
            bb2_smiles=s2,
            bb3_smiles=s3,
        ):
            rec = builder.build_trimer_from_smiles(s1, s2, s3, reaction_mode=builder.reaction_mode)
        trimer_smi = rec.smi
        score = scorer.score_smiles(trimer_smi)
        return i, j, k, s1, s2, s3, trimer_smi, score

    file_exists = out_path.exists()  # should be False
    mode = "a" if file_exists else "w"
    written = 0

    with out_path.open(mode, newline="", encoding="utf-8", buffering=1) as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        if not file_exists:
            writer.writerow(["bb1_id", "bb2_id", "bb3_id",
                             "bb1_smiles", "bb2_smiles", "bb3_smiles",
                             "smiles", "docking_score"])
            fcsv.flush()
            if fsync_every == 1 and hasattr(os, "fsync"):
                os.fsync(fcsv.fileno())

        if jobs > 1:
            chunk = max(1, (len(combos) + 9) // 10)
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futs = {ex.submit(work, t): t for t in combos}
                for n, fut in enumerate(as_completed(futs), 1):
                    try:
                        i,j,k,s1,s2,s3,trimer,score = fut.result()
                        writer.writerow([i,j,k,s1,s2,s3,trimer,score])
                        written += 1
                    except Exception as e:
                        i,j,k = futs[fut]
                        print(f"[warn] ({i},{j},{k}) failed: {e}", file=sys.stderr)
                        writer.writerow([i,j,k,"","","","", "nan"])
                        written += 1
                    if written % max(1, flush_every) == 0:
                        fcsv.flush()
                    if fsync_every > 0 and hasattr(os, "fsync") and written % fsync_every == 0:
                        os.fsync(fcsv.fileno())
                    if verbose and (n % chunk == 0 or n == len(combos)):
                        print(f"[progress] {n}/{len(combos)}")
        else:
            step = max(1, len(combos) // 10) if len(combos) >= 10 else 1
            for idx, t in enumerate(combos, 1):
                i,j,k = t
                try:
                    i,j,k,s1,s2,s3,trimer,score = work(t)
                    writer.writerow([i,j,k,s1,s2,s3,trimer,score])
                    written += 1
                except Exception as e:
                    print(f"[warn] ({i},{j},{k}) failed: {e}", file=sys.stderr)
                    writer.writerow([i,j,k,"","","","", "nan"])
                    written += 1
                if written % max(1, flush_every) == 0:
                    fcsv.flush()
                if fsync_every > 0 and hasattr(os, "fsync") and written % fsync_every == 0:
                    os.fsync(fcsv.fileno())
                if verbose and (idx % step == 0 or idx == len(combos)):
                    print(f"[progress] {idx}/{len(combos)}")

    print(f"[done] wrote {written} rows to {out_path.resolve()}")
    return out_path

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Score a trimer library; output CSV with (ids, bb1,bb2,bb3,trimer,score)."
    )

    # ---- Pool arguments ----
    ap.add_argument("--pool", default=None,
                    help="CSV with SMILES and optional ID column. Shorthand that sets all three pools "
                         "when --pool1/2/3 are not individually specified. "
                         "For AmpC (amide_sulfonamide) you must use --pool3 for the sulfonyl chlorides.")
    ap.add_argument("--pool1", default=None,
                    help="Pool-1 (BB1, amino acid) CSV with SMILES and optional ID column.")
    ap.add_argument("--pool2", default=None,
                    help="Pool-2 (BB2, amino acid) CSV with SMILES and optional ID column.")
    ap.add_argument("--pool3", default=None,
                    help="Pool-3 CSV. For amide_sulfonamide (AmpC): sulfonyl chloride CSV. "
                         "For amide_amide (sEH): amino acid CSV.")
    ap.add_argument("--bbs-combined", default=None,
                    help="Combined BBs CSV (with a 'pool' column tagging origin 1/2/3 and "
                         "globally unique IDs across all pools), as produced by "
                         "scripts/active_learning_submit.py. When provided, the script splits "
                         "this CSV by --pool-col into the three slot-specific dataframes and "
                         "uses a single global id->SMILES map shared by all three slots. "
                         "Mutually exclusive with --pool / --pool1 / --pool2 / --pool3.")
    ap.add_argument("--pool-col", default="pool",
                    help="Name of the pool-tag column in the combined BBs CSV (default: 'pool').")
    ap.add_argument("--id-col", default="ID",
                    help="ID column name shared across all pools (default: ID).")
    ap.add_argument("--id-col1", default=None)
    ap.add_argument("--id-col2", default=None)
    ap.add_argument("--id-col3", default=None)


    # ---- Reaction mode ----
    ap.add_argument(
        "--reaction-mode",
        default=REACTION_MODE_AMIDE_SULFONAMIDE,
        choices=list(VALID_REACTION_MODES),
    )

    # ---- Generate-only flag ----
    ap.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Generate trimer SMILES WITHOUT docking and write an unscored CSV "
            "(columns: bb1_id, bb2_id, bb3_id, bb1_smiles, bb2_smiles, bb3_smiles, smiles). "
            "No docking backend is required.  Pass the output to dock_library.py to score it."
        ),
    )

    # ---- Batch / block mode ----
    ap.add_argument("--batch-csv", default=None)
    ap.add_argument("--batch-size", default=1, type=int)
    ap.add_argument("--blocks", default=None)
    ap.add_argument("--random-block", default=None)

    # ---- Docking backend ----
    ap.add_argument("--docking-backend", default="vina", choices=["vina", "dock3"])
    ap.add_argument("--engine", default="vina")
    ap.add_argument("--receptor", default="data/seh/4jnc/4jnc.nohet.aligned.pdbqt")
    ap.add_argument("--engine-path", default="vina")
    ap.add_argument(
        "--target",
        default=None,
        choices=["Mpro", "TBLR1", "ClpP", "sEH"],
        help="Vina target; selects data/targets/<target>.pdbqt and its configured box.",
    )
    ap.add_argument("--box", default=None)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--indock-template", default=None)
    ap.add_argument("--dockfiles", default=None)
    ap.add_argument(
        "--dockenv-sh",
        default="/project/rrg-mailhoto/share/dockingpackages/dockenv.sh",
    )
    ap.add_argument("--dock64", default="/project/rrg-mailhoto/share/dock64")
    ap.add_argument("--strain-weight", type=float, default=0.0)
    ap.add_argument(
        "--dock3-timeout",
        type=int,
        default=600,
        help="Outer subprocess timeout in seconds for DOCK3 ligbuild/dock64 calls (default: %(default)s).",
    )
    ap.add_argument(
        "--ligbuild-timeout",
        type=int,
        default=300,
        help="Inner ligbuild DB2/protomer timeout in seconds written to custom_parms.json (default: %(default)s).",
    )

    # ---- Output ----
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--flush-every", type=int, default=1)
    ap.add_argument("--fsync-every", type=int, default=0)

    # ---- Random sampling ----
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--random-out")
    ap.add_argument("--random-only", action="store_true")
    ap.add_argument("--sample-with-replacement", action="store_true")
    ap.add_argument("--rand-seed", type=int, default=None)
    ap.add_argument(
        "--random-shard-index",
        type=int,
        default=None,
        help=(
            "Zero-based shard index for deterministic sharding of --random samples. "
            "Use with --random-num-shards, e.g. SLURM_ARRAY_TASK_ID for an array 0-(N-1)."
        ),
    )
    ap.add_argument(
        "--random-num-shards",
        type=int,
        default=None,
        help="Total number of shards for deterministic sharding of --random samples.",
    )
    ap.add_argument("--cpu", type=int, default=None)

    args = ap.parse_args()
    reaction_mode: str = args.reaction_mode

    install_rdkit_context_logger()

    # ---- Load pools ----
    df1, df2, df3, pool_mode, combined_df = build_pools(args)

    id_col1 = args.id_col1 or args.id_col
    id_col2 = args.id_col2 or args.id_col
    id_col3 = args.id_col3 or args.id_col

    if pool_mode == "combined":
        # Combined mode: any block ID may live in any pool slot (the combined CSV
        # uses globally unique IDs across all three pools), so we share a single
        # id->SMILES map across all three slots. The per-pool dataframes (df1/df2/df3)
        # are still used by TrimerBuilder for slot-specific chemistry.
        if args.id_col1 or args.id_col2 or args.id_col3:
            print(
                "[warn] --id-col1/2/3 are ignored in combined mode (--bbs-combined); "
                "using --id-col for the global ID column.",
                file=sys.stderr,
            )
        id_col_combined = args.id_col or "ID"
        id2smi_combined = make_id_to_smiles(combined_df, id_col_combined)
        id2smi1 = id2smi2 = id2smi3 = id2smi_combined
    else:
        id2smi1 = make_id_to_smiles(df1, id_col1)
        id2smi2 = make_id_to_smiles(df2, id_col2)
        id2smi3 = make_id_to_smiles(df3, id_col3)


    pool1_ids: List[int] = list(id2smi1.keys())
    pool2_ids: List[int] = list(id2smi2.keys())
    pool3_ids: List[int] = list(id2smi3.keys())

    if args.batch_csv:
        if args.blocks or args.random_block:
            print("[error] --batch-csv is mutually exclusive with --blocks/--random-block.", file=sys.stderr)
            sys.exit(2)
    elif args.blocks and args.random_block:
        print("[error] --blocks and --random-block are mutually exclusive.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build scorer (skipped when --generate-only) ----
    docking_backend: str = args.docking_backend
    scorer = None
    engine_bin = "(generate-only, no scorer)"
    box = None

    if not args.generate_only:
        if docking_backend == "dock3":
            if not args.indock_template:
                print("[error] --indock-template is required when --docking-backend dock3", file=sys.stderr)
                sys.exit(2)
            if not args.dockfiles:
                print("[error] --dockfiles is required when --docking-backend dock3", file=sys.stderr)
                sys.exit(2)
            if args.receptor != "data/seh/4jnc/4jnc.nohet.aligned.pdbqt" or args.box is not None:
                print("[warn] --receptor / --box are ignored when --docking-backend dock3", file=sys.stderr)
            try:
                scorer = Dock3Scorer(
                    indock_template=args.indock_template,
                    dockfiles_dir=args.dockfiles,
                    dockenv_sh=args.dockenv_sh,
                    dock64_exe=args.dock64,
                    strain_weight=args.strain_weight,
                    timeout=args.dock3_timeout,
                    ligbuild_timeout=args.ligbuild_timeout,
                )
            except Exception as e:
                print(f"[error] initializing Dock3Scorer: {e}", file=sys.stderr)
                sys.exit(2)
            engine_bin = f"dock64@{args.dock64}"
            box = None
        else:
            if args.target and args.box:
                print("[error] --target and --box are mutually exclusive", file=sys.stderr)
                sys.exit(2)
            if args.target:
                box = box_for_target(args.target)
                receptor = args.receptor
                if receptor == "data/seh/4jnc/4jnc.nohet.aligned.pdbqt":
                    receptor = f"data/targets/{args.target}.pdbqt"
            elif args.box:
                try:
                    toks = [float(x) for x in args.box.split(",")]
                    assert len(toks) == 6
                    box = tuple(toks)  # type: ignore
                except Exception:
                    print("[error] --box must be 6 comma-separated numbers", file=sys.stderr)
                    sys.exit(2)
            else:
                box = DEFAULT_4JNC_BOX
                receptor = args.receptor
            engine_bin = resolve_engine(args)
            try:
                scorer = DockingScorer(
                    receptor_pdbqt=receptor,
                    engine_path=engine_bin,
                    center_size=box,
                    exhaustiveness=args.exhaustiveness,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"[error] initializing DockingScorer: {e}", file=sys.stderr)
                sys.exit(2)

    # Chemistry builder
    builder = TrimerBuilder(df1, df2, df3, reaction_mode=reaction_mode)

    print(f"[config] reaction_mode='{reaction_mode}'  docking_backend='{docking_backend}'  target={args.target!r}  generate_only={args.generate_only}")
    if pool_mode == "combined":
        print(
            f"[config] pool_mode=combined  combined_BBs={len(id2smi1)} (shared global ID space)  "
            f"slot1={len(df1)} slot2={len(df2)} slot3={len(df3)}"
        )
    else:
        print(
            f"[config] pool_mode=per_pool  pool1 ({len(id2smi1)} BBs)  pool2 ({len(id2smi2)} BBs)  pool3 ({len(id2smi3)} BBs)"
        )


    # ---- BATCH MODE (not supported for --generate-only) ----
    if args.batch_csv:
        if args.generate_only:
            print("[error] --batch-csv is not supported with --generate-only.", file=sys.stderr)
            sys.exit(2)
        if args.out_csv:
            print("[error] --out-csv cannot be used with --batch-csv.", file=sys.stderr)
            sys.exit(2)
        try:
            bdf = pd.read_csv(args.batch_csv)[0:args.batch_size]
        except Exception as e:
            print(f"[error] reading --batch-csv: {e}", file=sys.stderr)
            sys.exit(2)
        for c in ["B1_id", "B2_id", "B3_id"]:
            if c not in bdf.columns:
                print(f"[error] --batch-csv missing column '{c}'", file=sys.stderr)
                sys.exit(2)
        for rix, row in bdf.iterrows():
            try:
                B1 = parse_pipe_ids(row["B1_id"], "B1_id")
                B2 = parse_pipe_ids(row["B2_id"], "B2_id")
                B3 = parse_pipe_ids(row["B3_id"], "B3_id")
            except Exception as e:
                print(f"[error] parsing row {rix}: {e}", file=sys.stderr)
                sys.exit(2)
            preflight_ids_present(B1, id2smi1, f"Row {rix} Block 1")
            preflight_ids_present(B2, id2smi2, f"Row {rix} Block 2")
            preflight_ids_present(B3, id2smi3, f"Row {rix} Block 3")
            n1, n2, n3 = len(B1), len(B2), len(B3)
            print(f"[batch] row {rix+1}/{len(bdf)} |B1|={n1} |B2|={n2} |B3|={n3} → total={n1*n2*n3}")
            score_blockset_to_csv(
                B1, B2, B3,
                id2smi1=id2smi1, id2smi2=id2smi2, id2smi3=id2smi3,
                builder=builder, scorer=scorer,
                out_dir=out_dir, out_csv=None,
                jobs=args.jobs, flush_every=args.flush_every,
                fsync_every=args.fsync_every, verbose=args.verbose,
            )
        print("[done batch] processed all rows.")
        return

    # ---- LEGACY SINGLE-RUN MODE: resolve B1,B2,B3 ----
    if args.blocks:
        try:
            B1, B2, B3 = parse_block_spec(args.blocks)
        except Exception as e:
            print(f"[error] parsing --blocks: {e}", file=sys.stderr)
            sys.exit(2)
        preflight_ids_present(B1, id2smi1, "Block 1")
        preflight_ids_present(B2, id2smi2, "Block 2")
        preflight_ids_present(B3, id2smi3, "Block 3")
        blocks_source = "from --blocks"
    elif args.random_block:
        try:
            n1, n2, n3 = parse_random_block_sizes(args.random_block)
        except Exception as e:
            print(f"[error] parsing --random-block: {e}", file=sys.stderr)
            sys.exit(2)
        if n1 > len(pool1_ids) or n2 > len(pool2_ids) or n3 > len(pool3_ids):
            print("[error] --random-block sizes exceed pool sizes.", file=sys.stderr)
            sys.exit(2)
        rng_state = None
        if args.rand_seed is not None:
            rng_state = random.getstate()
            random.seed(args.rand_seed)
        try:
            B1 = random.sample(pool1_ids, n1)
            B2 = random.sample(pool2_ids, n2)
            B3 = random.sample(pool3_ids, n3)
        finally:
            if rng_state is not None:
                random.setstate(rng_state)
        blocks_source = f"random blocks ({n1}|{n2}|{n3}) from pools"
    else:
        if args.random > 0:
            B1 = pool1_ids
            B2 = pool2_ids
            B3 = pool3_ids
            blocks_source = "entire pools (random sampling)"
        else:
            print("[error] --blocks required for full enumeration (or use --random N).", file=sys.stderr)
            sys.exit(2)

    print(f"[config] blocks={blocks_source}")

    # ---- Full enumeration (--blocks / --random-block) ----
    if (args.blocks or args.random_block) and not args.random_only:
        out_path = Path(args.out_csv) if args.out_csv else next_csv_path(out_dir)
        if args.generate_only:
            # Phase 1: write unscored CSV, no docking
            generate_blockset_to_csv(
                B1, B2, B3,
                id2smi1=id2smi1, id2smi2=id2smi2, id2smi3=id2smi3,
                builder=builder, out_path=out_path, verbose=args.verbose,
            )
            return
        score_blockset_to_csv(
            B1, B2, B3,
            id2smi1=id2smi1, id2smi2=id2smi2, id2smi3=id2smi3,
            builder=builder, scorer=scorer,
            out_dir=out_dir, out_csv=out_path,
            jobs=args.jobs, flush_every=args.flush_every,
            fsync_every=args.fsync_every, verbose=args.verbose,
        )

    # ---- Random sampling ----
    if args.random > 0:
        if args.rand_seed is not None:
            random.seed(args.rand_seed)

        n1, n2, n3 = len(B1), len(B2), len(B3)
        total = n1 * n2 * n3
        N = args.random
        if not args.sample_with_replacement:
            if N > total:
                print(f"[warn] --random {N} > total {total}; sampling {total}.", file=sys.stderr)
                N = total
            indices = random.sample(range(total), N)
        else:
            indices = [random.randrange(total) for _ in range(N)]

        shard_index = args.random_shard_index
        num_shards = args.random_num_shards
        if (shard_index is None) != (num_shards is None):
            print(
                "[error] --random-shard-index and --random-num-shards must be provided together.",
                file=sys.stderr,
            )
            sys.exit(2)
        if shard_index is not None and num_shards is not None:
            if num_shards <= 0:
                print("[error] --random-num-shards must be a positive integer.", file=sys.stderr)
                sys.exit(2)
            if shard_index < 0 or shard_index >= num_shards:
                print(
                    f"[error] --random-shard-index must satisfy 0 <= index < num_shards; "
                    f"got index={shard_index}, num_shards={num_shards}.",
                    file=sys.stderr,
                )
                sys.exit(2)

            global_n = len(indices)
            indices = indices[shard_index::num_shards]
            print(
                f"[config random shard] shard={shard_index}/{num_shards} "
                f"global_random_N={global_n} shard_N={len(indices)}"
            )

        triples = [index_to_triple(idx, B1, B2, B3) for idx in indices]

        rand_out_path = Path(args.random_out)

        if args.generate_only:
            # Phase 1: write unscored CSV for random sample
            generate_random_to_csv(
                triples,
                id2smi1=id2smi1, id2smi2=id2smi2, id2smi3=id2smi3,
                builder=builder, out_path=rand_out_path,
                verbose=args.verbose, reaction_mode=reaction_mode,
            )
            return

        # Scored random sampling
        file_exists = rand_out_path.exists()
        mode = "a" if file_exists else "w"
        written_rand = 0

        def work_rand(triple: Tuple[int,int,int]) -> Tuple[int,int,int,str, str, str, str, float]:
            i, j, k = triple
            s1, s2, s3 = id2smi1[i], id2smi2[j], id2smi3[k]
            with trimer_build_context(
                mode="score-random",
                bb1_id=i,
                bb2_id=j,
                bb3_id=k,
                bb1_smiles=s1,
                bb2_smiles=s2,
                bb3_smiles=s3,
            ):
                rec = builder.build_trimer_from_smiles(s1, s2, s3, reaction_mode=reaction_mode)
            trimer_smi = rec.smi
            score = scorer.score_smiles(trimer_smi)
            return i, j, k, s1, s2, s3, trimer_smi, score

        with rand_out_path.open(mode, newline="", encoding="utf-8", buffering=1) as frand:
            writer = csv.writer(frand, lineterminator="\n")
            if not file_exists:
                writer.writerow(["bb1_id", "bb2_id", "bb3_id",
                                 "bb1_smiles", "bb2_smiles", "bb3_smiles",
                                 "smiles", "trimer_score"])
                frand.flush()

            if args.jobs > 1:
                chunk = max(1, (len(triples) + 9) // 10)
                with ThreadPoolExecutor(max_workers=args.jobs) as ex:
                    futs = {ex.submit(work_rand, t): t for t in triples}
                    for n, fut in enumerate(as_completed(futs), 1):
                        try:
                            i,j,k,s1,s2,s3,trimer,score = fut.result()
                            writer.writerow([i,j,k,s1,s2,s3,trimer,score])
                            written_rand += 1
                        except Exception as e:
                            i,j,k = futs[fut]
                            print(f"[warn] (rand {i},{j},{k}) failed: {e}", file=sys.stderr)
                            writer.writerow([i,j,k,"","","","", "nan"])
                            written_rand += 1
                        if written_rand % max(1, args.flush_every) == 0:
                            frand.flush()
                        if args.fsync_every > 0 and hasattr(os, "fsync") and written_rand % args.fsync_every == 0:
                            os.fsync(frand.fileno())
                        if args.verbose and (n % chunk == 0 or n == len(triples)):
                            print(f"[progress random] {n}/{len(triples)}")
            else:
                step = max(1, len(triples) // 10) if len(triples) >= 10 else 1
                for idx, t in enumerate(triples, 1):
                    i,j,k = t
                    try:
                        i,j,k,s1,s2,s3,trimer,score = work_rand(t)
                        writer.writerow([i,j,k,s1,s2,s3,trimer,score])
                        written_rand += 1
                    except Exception as e:
                        print(f"[warn] (rand {i},{j},{k}) failed: {e}", file=sys.stderr)
                        writer.writerow([i,j,k,"","","","", "nan"])
                        written_rand += 1
                    if written_rand % max(1, args.flush_every) == 0:
                        frand.flush()
                    if args.fsync_every > 0 and hasattr(os, "fsync") and written_rand % args.fsync_every == 0:
                        os.fsync(frand.fileno())
                    if args.verbose and (idx % step == 0 or idx == len(triples)):
                        print(f"[progress random] {idx}/{len(triples)}")

        print(f"[done random] wrote {written_rand} rows to {rand_out_path.resolve()}")

if __name__ == "__main__":
    main()
