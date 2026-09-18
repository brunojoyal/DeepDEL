#!/usr/bin/env python3
"""Parallel molecule-level docking runner for pre-generated library CSVs.

This module is the package-local version used by the active-learning pipeline.
It scores one pooled unscored CSV while keeping a thread pool fed at the
molecule level, then writes the input rows back with an added ``docking_score``
column.  It intentionally avoids importing from ``scripts/`` and does not depend
on ``omltk``.

The original implementation was DOCK3-only.  The CLI now also supports
``--backend vina`` so pre-generated DEL CSVs can be docked against sEH with
AutoDock Vina while preserving the same generate-then-dock workflow.

Design notes
------------
* A single ``Dock3Scorer`` is constructed once on the main thread.  Its
  ``__init__`` runs ``_warmup_dockenv()`` which serially provisions the
  ``build_3d_dock_py`` venv on a fresh compute node.  If we instead built a
  ``Dock3Scorer`` inside every worker process, those warmups would race the
  ``pip install`` of OpenEye / build_3d_dock_py and cause sporadic
  ``No such file or directory: .../bin/extract_ligand_oedu.py`` errors.
* Each ``score_smiles`` call uses ``Dock3Scorer``'s built-in short workdir at
  ``/tmp/d.<JOB_TAG>/dXXXXXX`` (~25 chars).  AMSOL silently corrupts builds
  when total paths exceed ~80 chars, so we deliberately do NOT pass a long
  ``tmp_dir`` under ``$SCRATCH``.
* ``Dock3Scorer.score_smiles`` is thread-safe (it makes a unique workdir per
  call and uses no shared mutable state besides the warmup-once dockenv), so a
  ``ThreadPoolExecutor`` is sufficient; ligbuild and dock64 are subprocesses
  that release the GIL.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def extract_best_score_from_dock_out(dock_out_dir: Path) -> float:
    """Read DOCKDF.df under ``dock_out_dir`` and return the best score."""
    dockdf_files = list(dock_out_dir.rglob("DOCKDF.df"))
    if not dockdf_files:
        return 0.0

    frames = []
    for df_path in dockdf_files:
        try:
            frames.append(pd.read_csv(df_path, sep=r"\s+", comment="#"))
        except Exception as e:
            print(f"[warn] could not read {df_path}: {e}", file=sys.stderr)

    if not frames:
        return 0.0
    dockdf = pd.concat(frames, ignore_index=True)

    score_col = None
    for candidate in ("total", "Total", "score", "Score"):
        if candidate in dockdf.columns:
            score_col = candidate
            break
    if score_col is None:
        print(f"[warn] Could not find score column in DOCKDF.df. Columns: {list(dockdf.columns)}", file=sys.stderr)
        return 0.0

    try:
        return float(dockdf[score_col].min())
    except Exception:
        return 0.0


def debug_dump_dock_out(dock_out_dir: Path, mol_name: str) -> None:
    """Print concise diagnostics for a per-molecule DOCK3 output directory."""
    print(f"[dock3-debug] Molecule: {mol_name}", file=sys.stderr)
    print(f"[dock3-debug] Output dir: {dock_out_dir}", file=sys.stderr)
    entries = []
    try:
        for p in sorted(dock_out_dir.rglob("*")):
            rel = p.relative_to(dock_out_dir)
            if p.is_dir():
                entries.append(f"    {rel}/")
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = -1
                entries.append(f"    {rel} ({size} bytes)")
            if len(entries) >= 60:
                entries.append("    ...")
                break
    except Exception as e:
        entries.append(f"    [could not list output tree: {e}]")
    print("[dock3-debug] Output tree:", file=sys.stderr)
    print("\n".join(entries) if entries else "    [empty]", file=sys.stderr)

    for pattern in ("DOCKDF.df", "OUTDOCK", "*.err", "*.out"):
        for p in sorted(dock_out_dir.rglob(pattern)):
            try:
                text = p.read_text(errors="replace")
            except Exception as e:
                print(f"[dock3-debug] Could not read {p}: {e}", file=sys.stderr)
                continue
            print(f"[dock3-debug] Begin {p}", file=sys.stderr)
            print(text[:8000] if text else "[empty]", file=sys.stderr)
            if len(text) > 8000:
                print("[dock3-debug] ... truncated ...", file=sys.stderr)
            print(f"[dock3-debug] End {p}", file=sys.stderr)


def run_ligbuild_for_smi(
    smi_file: Path,
    out_dir: Path,
    *,
    smiles: str | None = None,
    ligbuild_exe: str = "ligbuild",
    timeout: int = 300,
) -> Optional[Path]:
    """Run ligbuild on ``smi_file`` and return the first produced ``.tgz``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parms_path = out_dir / "custom_parms.json"
    parms_path.write_text(json.dumps({"verbose": 1, "timeout": int(timeout)}))

    cmd = [ligbuild_exe, str(smi_file), str(out_dir), str(parms_path)]
    print(f"[ligbuild] CMD: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, text=True, timeout=timeout, cwd=str(out_dir))

    tgz_files = list(out_dir.glob("*.tgz")) or list((out_dir / "db2_archives").glob("*.tgz")) or list(out_dir.rglob("*.tgz"))
    if not tgz_files:
        smiles_msg = f"\n  SMILES: {smiles}" if smiles is not None else ""
        print(
            f"[warn] ligbuild produced no .tgz for {smi_file.name} (rc={result.returncode})."
            f"{smiles_msg}\n  Output dir: {out_dir}",
            file=sys.stderr,
        )
        return None
    return tgz_files[0]


def cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[warn] cleanup failed for {path}: {e}", file=sys.stderr)


# Shared, warmed-up scorer for the thread pool.  Constructed once in main()
# before workers are launched so that ``_warmup_dockenv`` (which provisions the
# build_3d_dock_py venv) runs exactly one time, serially, on a fresh node.
_SHARED_SCORER = None


def parse_box(box: str | None):
    """Parse Vina box as ``cx,cy,cz,sx,sy,sz`` or return the 4JNC default."""
    from deepdelgfn.mols.vina_scorer import DEFAULT_4JNC_BOX

    if not box:
        return DEFAULT_4JNC_BOX
    try:
        toks = tuple(float(x) for x in box.split(","))
    except Exception as e:
        raise ValueError("--box must be 6 comma-separated numbers: cx,cy,cz,sx,sy,sz") from e
    if len(toks) != 6:
        raise ValueError("--box must be 6 comma-separated numbers: cx,cy,cz,sx,sy,sz")
    return toks


def dock_one_molecule(task: dict) -> dict:
    smiles = task["smiles"]
    name = task["name"]
    index = int(task["index"])
    backend = task.get("backend", "dock3")

    result = {"index": index, "score": 0.0, "built": False, "scored": False}

    scorer = _SHARED_SCORER
    if scorer is None:
        print(f"[error] dock_one_molecule called without a shared scorer (mol={name})", file=sys.stderr)
        return result

    try:
        if backend == "vina":
            score = scorer.score_smiles(smiles)
        else:
            score = scorer.score_smiles(smiles, name=name)
        result["score"] = float(score) if not pd.isna(score) else 0.0
        result["scored"] = True
        if backend == "vina":
            result["built"] = True
            return result
        # The scorer reports last_failure_reason via instance state; reading it
        # here from a thread is fine because each score_smiles call also returns
        # serially before the next ligand starts on the same thread.
        last_reason = getattr(scorer, "last_failure_reason", None)
        ligbuild_failures = {
            "ligbuild_failed",
            "ligbuild_no_tgz",
            "ligbuild_db2_timeout",
            "ligbuild_subprocess_timeout",
        }
        result["built"] = result["score"] != 0.0 or last_reason not in ligbuild_failures
    except Exception as e:
        print(f"[warn] docking failed for {name}: {e}", file=sys.stderr)

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Score an unscored library CSV using DOCK3 or Vina in parallel.")
    ap.add_argument("--unscored-csv", required=True, help="Input CSV containing at least a smiles column.")
    ap.add_argument("--backend", choices=["dock3", "vina"], default="dock3", help="Docking backend (default: dock3).")
    ap.add_argument("--indock", default=None, help="Path to INDOCK template (required for --backend dock3).")
    ap.add_argument("--dockfiles", default=None, help="Path to DOCK3 dockfiles directory (required for --backend dock3).")
    ap.add_argument("--dock64", default="/project/rrg-mailhoto/share/dock64", help="Path to dock64 binary.")
    ap.add_argument("--n-proc", type=int, default=1, help="Number of molecules to process in parallel.")
    ap.add_argument("--out-csv", required=True, help="Output scored CSV path.")
    ap.add_argument("--ligbuild", default="ligbuild", help="ligbuild executable name or path.")
    ap.add_argument("--ligbuild-timeout", type=int, default=720, help="Timeout per ligbuild call in seconds.")
    ap.add_argument("--dock3-timeout", type=int, default=900, help="Timeout per dock64 call in seconds.")
    ap.add_argument("--receptor", default=None, help="Receptor PDBQT for --backend vina. Defaults to data/targets/<target>.pdbqt when --target is set, else data/seh/4jnc/4jnc.nohet.aligned.pdbqt.")
    ap.add_argument("--engine", default="vina", help="Vina/QuickVina executable for --backend vina.")
    ap.add_argument("--engine-path", default=None, help="Alias for --engine; if set, takes precedence.")
    ap.add_argument("--target", default=None, help="Target name (Mpro, TBLR1, ClpP, sEH). Selects the box from TARGET_BOXES and defaults --receptor to data/targets/<target>.pdbqt. Mutually exclusive with --box.")
    ap.add_argument("--box", default=None, help="Vina box as cx,cy,cz,sx,sy,sz. Defaults to the sEH/4JNC box.")
    ap.add_argument("--exhaustiveness", type=int, default=8, help="Vina exhaustiveness.")
    ap.add_argument("--seed", type=int, default=42, help="Vina random seed.")
    ap.add_argument("--cpu", type=int, default=1, help="CPU threads per Vina ligand. Usually keep this at 1 when --n-proc > 1.")
    ap.add_argument("--work-dir", default=None, help="Scratch directory for ligbuild/dock3.")
    ap.add_argument("--keep-zero-score-artifacts", action="store_true")
    args = ap.parse_args()

    # ``Dock3Scorer`` uses a built-in short workdir under /tmp/d.<JOB_TAG>/dXXXXXX
    # to keep AMSOL/ligbuild paths under the ~80-char ceiling. For Vina,
    # ``--work-dir`` is used as the parent directory for per-ligand temporary
    # ligand/log files.
    if args.work_dir and args.backend == "dock3":
        print(
            f"[info] --work-dir={args.work_dir} is accepted for backward compatibility "
            "but ignored; Dock3Scorer manages its own short /tmp workdir to avoid "
            "AMSOL path-length corruption.",
            file=sys.stderr,
        )

    out_csv = Path(args.out_csv).resolve()
    indock = dockfiles = dock64 = None
    if args.backend == "dock3":
        if not args.indock or not args.dockfiles:
            print("[error] --indock and --dockfiles are required when --backend dock3", file=sys.stderr)
            sys.exit(1)
        indock = Path(args.indock).resolve()
        dockfiles = Path(args.dockfiles).resolve()
        dock64 = Path(args.dock64).resolve()
        for p, name in [(indock, "--indock"), (dockfiles, "--dockfiles"), (dock64, "--dock64")]:
            if not p.exists():
                print(f"[error] {name} path does not exist: {p}", file=sys.stderr)
                sys.exit(1)
    else:
        if args.target and args.box:
            print("[error] --target and --box are mutually exclusive", file=sys.stderr)
            sys.exit(1)
        if args.target:
            from deepdelgfn.mols.vina_scorer import box_for_target

            try:
                box = box_for_target(args.target)
            except ValueError as e:
                print(f"[error] {e}", file=sys.stderr)
                sys.exit(1)
            receptor_arg = args.receptor or f"data/targets/{args.target}.pdbqt"
        else:
            box = parse_box(args.box)
            receptor_arg = args.receptor or "data/seh/4jnc/4jnc.nohet.aligned.pdbqt"
        receptor = Path(receptor_arg).resolve()
        if not receptor.exists():
            print(f"[error] --receptor path does not exist: {receptor}", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(args.unscored_csv)
    if "smiles" not in df.columns:
        print("[error] --unscored-csv is missing required column 'smiles'", file=sys.stderr)
        sys.exit(1)
    failed_mask = df["smiles"].isna() | ~df["smiles"].astype(str).str.strip().astype(bool)
    df = df.loc[~failed_mask].copy().reset_index(drop=True)
    df["smiles"] = df["smiles"].astype(str)
    print(f"[info] {len(df)} molecules to dock (after dropping empty SMILES).")

    rows_by_index = {i: list(row) for i, row in enumerate(df.itertuples(index=False, name=None))}
    smiles_col_index = df.columns.get_loc("smiles")
    total_tasks = len(rows_by_index)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Build one shared scorer in the main thread. For DOCK3, its ``__init__``
    # runs ``_warmup_dockenv`` serially to avoid worker races. For Vina, the
    # shared scorer is lightweight; each score call creates a unique workdir.
    global _SHARED_SCORER
    if args.backend == "dock3":
        from deepdelgfn.mols.vina_scorer import Dock3Scorer

        print(f"[info] Initializing shared Dock3Scorer (warming up dockenv)…", file=sys.stderr)
        _SHARED_SCORER = Dock3Scorer(
            indock_template=str(indock),
            dockfiles_dir=str(dockfiles),
            dock64_exe=str(dock64),
            ligbuild_exe=args.ligbuild,
            # tmp_dir=None -> Dock3Scorer creates a short /tmp/d.<JOB_TAG>/dXXXXXX
            # workdir per call, well under the AMSOL path-length ceiling.
            tmp_dir=None,
            timeout=int(args.dock3_timeout),
            ligbuild_timeout=int(args.ligbuild_timeout),
        )
    else:
        from deepdelgfn.mols.vina_scorer import DockingScorer

        engine_bin = args.engine_path or args.engine
        print(
            f"[info] Initializing shared DockingScorer "
            f"(engine={engine_bin}, receptor={receptor}, box={box}, cpu={args.cpu})…",
            file=sys.stderr,
        )
        _SHARED_SCORER = DockingScorer(
            receptor_pdbqt=str(receptor),
            engine_path=engine_bin,
            center_size=box,
            exhaustiveness=int(args.exhaustiveness),
            seed=int(args.seed),
            cpu=int(args.cpu),
            tmp_dir=args.work_dir,
        )

    results_by_index: dict[int, dict] = {}
    n_built = n_scored = completed = written = 0
    next_index_to_write = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow(list(df.columns) + ["docking_score"])
        if total_tasks == 0:
            print(f"[done] 0/0 molecules scored. Output: {out_csv}")
            return

        with ThreadPoolExecutor(max_workers=max(1, int(args.n_proc))) as executor:
            future_to_index = {}
            next_index_to_submit = 0

            def submit_one(index: int) -> None:
                row_values = rows_by_index[index]
                task = {
                    "index": index,
                    "name": f"m{index}",
                    "smiles": str(row_values[smiles_col_index]),
                    "backend": args.backend,
                }
                future_to_index[executor.submit(dock_one_molecule, task)] = index

            for _ in range(min(max(1, int(args.n_proc)), total_tasks)):
                submit_one(next_index_to_submit)
                next_index_to_submit += 1

            while future_to_index:
                future = next(as_completed(future_to_index))
                index = future_to_index.pop(future)
                try:
                    res = future.result()
                except Exception as e:
                    print(f"[warn] worker failed for molecule index {index}: {e}", file=sys.stderr)
                    res = {"index": index, "score": 0.0, "built": False, "scored": False}

                if next_index_to_submit < total_tasks:
                    submit_one(next_index_to_submit)
                    next_index_to_submit += 1

                results_by_index[res["index"]] = res
                completed += 1
                n_built += int(bool(res["built"]))
                n_scored += int(bool(res["scored"]))

                while next_index_to_write in results_by_index:
                    ready = results_by_index.pop(next_index_to_write)
                    writer.writerow(rows_by_index.pop(next_index_to_write) + [ready["score"]])
                    written += 1
                    next_index_to_write += 1
                fcsv.flush()
                if completed % max(1, total_tasks // 10) == 0 or completed == total_tasks:
                    print(f"  [dock_library_parallel] {completed}/{total_tasks} done, {n_built} bundles built, {n_scored} molecules scored so far, {written} rows written.")

    print(f"[done] {n_scored}/{written} molecules scored. Output: {out_csv}")


if __name__ == "__main__":
    main()