#!/usr/bin/env python3
"""Audit DEL trimer build failures with building-block provenance.

This answers: which BB triples trigger RDKit sanitize/kekulization failures
after coupling?  Individual BBs can sanitize fine and fail only in a coupled
product, so the primary audit unit is a BB triple; the script also aggregates
failures by BB ID to identify recurring offenders.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd

from deepdelgfn.mols import dels as tri_mod


def load_bbs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    smiles_col = tri_mod.PoolIO.resolve_smiles_col(df)
    if smiles_col != "SMILES":
        df = df.rename(columns={smiles_col: "SMILES"})
    if "ID" not in df.columns:
        df["ID"] = np.arange(len(df), dtype=int)
    if "Name" not in df.columns:
        df["Name"] = df["Catalog_ID"].astype(str) if "Catalog_ID" in df.columns else [f"BB_{i}" for i in range(len(df))]
    return df


def split_bbs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "pool" in df.columns:
        pools = pd.to_numeric(df["pool"], errors="coerce").fillna(0).astype(int)
        if np.any(pools.to_numpy() != 0):
            return tri_mod.PoolIO.split_by_pool(df.assign(pool=pools))
    d = df.reset_index(drop=True)
    return d, d.copy(), d.copy()


def parse_id_list(value: str) -> List[int]:
    return [int(tok.strip()) for tok in str(value).split("|") if tok.strip()]


def id_to_local_index(df: pd.DataFrame) -> Dict[int, int]:
    return {int(bb_id): int(local_idx) for local_idx, bb_id in enumerate(df["ID"].tolist())}


def sample_deepdel_triplet_sets(
    sizes: Tuple[int, int, int],
    min_sizes: Tuple[int, int, int],
    max_sizes: Tuple[int, int, int],
    *,
    n_triplet_sets: int,
    seed: int | None,
) -> Iterator[Tuple[int, Sequence[int], Sequence[int], Sequence[int]]]:
    rng = np.random.default_rng(seed)
    idx = [np.arange(n, dtype=int) for n in sizes]
    for triplet_idx in range(int(n_triplet_sets)):
        picked = []
        for pool_idx, smin, smax in zip(idx, min_sizes, max_sizes):
            k = int(rng.integers(int(smin), int(smax) + 1))
            picked.append(rng.choice(pool_idx, size=k, replace=False).astype(int).tolist())
        yield triplet_idx, picked[0], picked[1], picked[2]


def iter_random_triples(sizes: Tuple[int, int, int], *, n_triples: int, seed: int | None) -> Iterator[Tuple[int, int, int, int]]:
    rng = np.random.default_rng(seed)
    n1, n2, n3 = sizes
    for idx in range(int(n_triples)):
        yield idx, int(rng.integers(n1)), int(rng.integers(n2)), int(rng.integers(n3))


def iter_triples_csv(
    path: str,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    *,
    max_products_per_row: int,
) -> Iterator[Tuple[int, int, int, int]]:
    frame = pd.read_csv(path)
    id_maps = (id_to_local_index(df1), id_to_local_index(df2), id_to_local_index(df3))
    col_options = [("bb1_id", "B1_id"), ("bb2_id", "B2_id"), ("bb3_id", "B3_id")]
    cols = []
    for opts in col_options:
        col = next((c for c in opts if c in frame.columns), None)
        if col is None:
            raise SystemExit(f"{path} must contain one of {opts!r}")
        cols.append(col)

    emitted = 0
    for row_idx, row in frame.iterrows():
        id_lists = [parse_id_list(row[col]) for col in cols]
        local_lists = []
        for slot, (ids, mapping) in enumerate(zip(id_lists, id_maps), start=1):
            missing = [x for x in ids if x not in mapping]
            if missing:
                raise SystemExit(f"row {row_idx}: slot {slot} IDs missing from pool: {missing[:10]}")
            local_lists.append([mapping[x] for x in ids])
        for n, (i, j, k) in enumerate(itertools.product(local_lists[0], local_lists[1], local_lists[2])):
            if max_products_per_row > 0 and n >= max_products_per_row:
                break
            yield emitted, int(i), int(j), int(k)
            emitted += 1


def failure_record(
    *,
    sample_idx: int,
    triplet_set_idx: int | None,
    i: int,
    j: int,
    k: int,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    exc: Exception,
) -> Dict[str, object]:
    r1, r2, r3 = df1.loc[int(i)], df2.loc[int(j)], df3.loc[int(k)]
    return {
        "sample_idx": sample_idx,
        "triplet_set_idx": "" if triplet_set_idx is None else triplet_set_idx,
        "bb1_local_idx": i,
        "bb2_local_idx": j,
        "bb3_local_idx": k,
        "bb1_id": int(r1["ID"]),
        "bb2_id": int(r2["ID"]),
        "bb3_id": int(r3["ID"]),
        "bb1_name": r1.get("Name", ""),
        "bb2_name": r2.get("Name", ""),
        "bb3_name": r3.get("Name", ""),
        "bb1_smiles": r1.get("SMILES", ""),
        "bb2_smiles": r2.get("SMILES", ""),
        "bb3_smiles": r3.get("SMILES", ""),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def write_summary(failures_csv: Path, summary_csv: Path) -> None:
    failures = pd.read_csv(failures_csv)
    rows = []
    for slot in (1, 2, 3):
        id_col = f"bb{slot}_id"
        name_col = f"bb{slot}_name"
        smi_col = f"bb{slot}_smiles"
        grouped = failures.groupby([id_col, name_col, smi_col], dropna=False).size().reset_index(name="n_failures")
        for row in grouped.to_dict("records"):
            rows.append(
                {
                    "slot": slot,
                    "bb_id": row[id_col],
                    "bb_name": row[name_col],
                    "bb_smiles": row[smi_col],
                    "n_failures": int(row["n_failures"]),
                }
            )
    pd.DataFrame(rows).sort_values(["n_failures", "slot", "bb_id"], ascending=[False, True, True]).to_csv(summary_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Find DEL trimer build failures and aggregate problematic BBs.")
    ap.add_argument("--bbs", default="data/bbs_JP.csv", help="BB CSV; may be a combined CSV with pool tags.")
    ap.add_argument(
        "--reaction-mode",
        default=tri_mod.REACTION_MODE_AMIDE_AMIDE,
        choices=list(tri_mod.VALID_REACTION_MODES),
    )
    ap.add_argument("--out", required=True, help="Failure-detail CSV to write.")
    ap.add_argument("--summary-out", default=None, help="Per-BB failure-count CSV. Default: <out>.summary.csv")
    ap.add_argument("--seed", type=int, default=None)

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--random-triples", type=int, help="Audit N uniformly random individual triples.")
    mode.add_argument("--deepdel-triplet-sets", type=int, help="Sample N DeepDEL triplet sets and audit their Cartesian products.")
    mode.add_argument("--triples-csv", default=None, help="CSV with bb1_id/bb2_id/bb3_id or B1_id/B2_id/B3_id columns.")

    ap.add_argument("--min-size1", type=int, default=3)
    ap.add_argument("--min-size2", type=int, default=3)
    ap.add_argument("--min-size3", type=int, default=3)
    ap.add_argument("--max-size1", type=int, default=3)
    ap.add_argument("--max-size2", type=int, default=3)
    ap.add_argument("--max-size3", type=int, default=3)
    ap.add_argument("--max-products-per-row", type=int, default=0, help="For --triples-csv pipe-list rows, 0 means no cap.")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--stop-after-failures", type=int, default=0, help="Stop after this many failures (0=disabled).")
    args = ap.parse_args()

    df = load_bbs(args.bbs)
    df1, df2, df3 = split_bbs(df)
    builder = tri_mod.TrimerBuilder(df1, df2, df3, reaction_mode=args.reaction_mode)
    print(f"[audit] pool sizes: |B1|={len(df1)}, |B2|={len(df2)}, |B3|={len(df3)}", file=sys.stderr)

    if args.random_triples is not None:
        triples: Iterable[Tuple[int, int | None, int, int, int]] = (
            (sample_idx, None, i, j, k)
            for sample_idx, i, j, k in iter_random_triples((len(df1), len(df2), len(df3)), n_triples=args.random_triples, seed=args.seed)
        )
    elif args.deepdel_triplet_sets is not None:
        def _deepdel_iter() -> Iterator[Tuple[int, int, int, int, int]]:
            sample_idx = 0
            for set_idx, I, J, K in sample_deepdel_triplet_sets(
                (len(df1), len(df2), len(df3)),
                (args.min_size1, args.min_size2, args.min_size3),
                (args.max_size1, args.max_size2, args.max_size3),
                n_triplet_sets=args.deepdel_triplet_sets,
                seed=args.seed,
            ):
                for i, j, k in itertools.product(I, J, K):
                    yield sample_idx, set_idx, int(i), int(j), int(k)
                    sample_idx += 1
        triples = _deepdel_iter()
    else:
        triples = (
            (sample_idx, None, i, j, k)
            for sample_idx, i, j, k in iter_triples_csv(args.triples_csv, df1, df2, df3, max_products_per_row=args.max_products_per_row)
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(out_path.suffix + ".summary.csv")
    fieldnames = [
        "sample_idx", "triplet_set_idx",
        "bb1_local_idx", "bb2_local_idx", "bb3_local_idx",
        "bb1_id", "bb2_id", "bb3_id",
        "bb1_name", "bb2_name", "bb3_name",
        "bb1_smiles", "bb2_smiles", "bb3_smiles",
        "error_type", "error_message",
    ]

    n_tested = 0
    n_failed = 0
    error_counts: Counter[str] = Counter()
    with out_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for sample_idx, set_idx, i, j, k in triples:
            n_tested += 1
            try:
                builder.build_trimer(int(i), int(j), int(k))
            except Exception as exc:
                n_failed += 1
                error_counts[type(exc).__name__] += 1
                wr.writerow(
                    failure_record(
                        sample_idx=int(sample_idx),
                        triplet_set_idx=None if set_idx is None else int(set_idx),
                        i=int(i),
                        j=int(j),
                        k=int(k),
                        df1=df1,
                        df2=df2,
                        df3=df3,
                        exc=exc,
                    )
                )
                if args.stop_after_failures and n_failed >= int(args.stop_after_failures):
                    break
            if args.progress_every > 0 and n_tested % int(args.progress_every) == 0:
                print(f"[audit] tested={n_tested} failures={n_failed}", file=sys.stderr)

    print(f"[audit] done: tested={n_tested} failures={n_failed} out={out_path}", file=sys.stderr)
    if error_counts:
        print(f"[audit] error counts: {dict(error_counts)}", file=sys.stderr)
        write_summary(out_path, summary_path)
        print(f"[audit] BB failure summary: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()