#!/usr/bin/env python3
"""
Parallel Phase 2: score an unscored library CSV using DOCK3 natively via omltk.

This variant preserves the existing dock_library.py workflow, but runs multiple
molecules concurrently using a process pool. Each worker handles one molecule at
a time and invokes run_docking_from_tgz(..., n_proc=1) to avoid nested CPU
oversubscription.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def extract_best_score_from_dock_out(dock_out_dir: Path) -> float:
    """Read DOCKDF.df under dock_out_dir and return the best score, or 0.0 if absent."""
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
        print(
            f"[warn] Could not find score column in DOCKDF.df. Columns: {list(dockdf.columns)}",
            file=sys.stderr,
        )
        return 0.0

    try:
        return float(dockdf[score_col].min())
    except Exception:
        return 0.0


def debug_dump_dock_out(dock_out_dir: Path, mol_name: str) -> None:
    """Print verbose diagnostics for a per-molecule DOCK3 output directory."""
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
    smiles: str | None = None,
    ligbuild_exe: str = "ligbuild",
    timeout: int = 300,
) -> Optional[Path]:
    """
    Run ligbuild on *smi_file* and return the first .tgz found, or None.
    ligbuild must already be on PATH (sourced via dockenv.sh in the job script).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parms_path = out_dir / "custom_parms.json"
    parms_path.write_text(json.dumps({"verbose": 1, "timeout": 720}))

    cmd = [ligbuild_exe, str(smi_file), str(out_dir), str(parms_path)]
    print(f"[ligbuild] CMD: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd,
        text=True,
        timeout=timeout,
        cwd=str(out_dir),
    )

    tgz_files = (
        list(out_dir.glob("*.tgz"))
        or list((out_dir / "db2_archives").glob("*.tgz"))
        or list(out_dir.rglob("*.tgz"))
    )
    if not tgz_files:
        smiles_msg = f"\n  SMILES: {smiles}" if smiles is not None else ""

        tree_entries = []
        try:
            for p in sorted(out_dir.rglob("*")):
                rel = p.relative_to(out_dir)
                if p.is_dir():
                    tree_entries.append(f"    {rel}/")
                else:
                    try:
                        size = p.stat().st_size
                    except OSError:
                        size = -1
                    tree_entries.append(f"    {rel} ({size} bytes)")
                if len(tree_entries) >= 40:
                    tree_entries.append("    ...")
                    break
        except Exception as e:
            tree_entries.append(f"    [could not list output tree: {e}]")

        tree_msg = "\n".join(tree_entries) if tree_entries else "    [no files created under output dir]"
        print(
            f"[warn] ligbuild produced no .tgz for {smi_file.name} "
            f"(rc={result.returncode})."
            f"{smiles_msg}\n"
            f"  Output dir: {out_dir}\n"
            f"  Note: ligbuild stdout/stderr were streamed directly to the job log above.\n"
            f"  Output tree:\n{tree_msg}",
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


def maybe_cleanup_molecule_artifacts(
    *,
    smi_file: Path,
    mol_lb_dir: Path,
    mol_run_root: Path,
    mol_dock_out: Path,
    keep_artifacts: bool,
) -> None:
    if keep_artifacts:
        return
    cleanup_paths([smi_file, mol_lb_dir, mol_run_root, mol_dock_out])


def dock_one_molecule(task: dict) -> dict:
    indock = Path(task["indock"])
    dockfiles = Path(task["dockfiles"])
    dock64 = Path(task["dock64"])
    work_root = Path(task["work_root"])
    ligbuild_exe = task["ligbuild"]
    ligbuild_timeout = int(task["ligbuild_timeout"])
    smiles = task["smiles"]
    name = task["name"]
    index = int(task["index"])
    keep_zero_score_artifacts = bool(task.get("keep_zero_score_artifacts", False))

    smi_dir = work_root / "smi_files"
    lb_dir = work_root / "ligbuild_out"
    dock_root = work_root / "dock_out"
    smi_dir.mkdir(parents=True, exist_ok=True)
    lb_dir.mkdir(parents=True, exist_ok=True)
    dock_root.mkdir(parents=True, exist_ok=True)

    smi_file = smi_dir / f"{name}.smi"
    smi_file.write_text(f"{smiles} {name}\n")

    result = {
        "index": index,
        "score": 0.0,
        "built": False,
        "scored": False,
    }

    mol_lb_dir = lb_dir / name
    tgz = run_ligbuild_for_smi(
        smi_file,
        mol_lb_dir,
        smiles=smiles,
        ligbuild_exe=ligbuild_exe,
        timeout=ligbuild_timeout,
    )

    if tgz is None:
        return result

    result["built"] = True
    mol_dock_out = dock_root / name
    mol_dock_out.mkdir(parents=True, exist_ok=True)
    mol_run_root = work_root / f"dock_run_{name}"
    if mol_run_root.exists():
        shutil.rmtree(mol_run_root, ignore_errors=True)
    mol_run_root.mkdir(parents=True, exist_ok=True)

    old_slurm_tmp = os.environ.get("SLURM_TMPDIR")
    try:
        from omltk.docking import run_docking_from_tgz

        os.environ["SLURM_TMPDIR"] = str(mol_run_root)
        with pushd(mol_run_root):
            run_docking_from_tgz(
                indock_template=str(indock),
                dockfiles=str(dockfiles),
                tgz_files_list=[str(tgz)],
                output_folder=str(mol_dock_out),
                n_proc=1,
                dock_exec_path=str(dock64),
            )
        score = extract_best_score_from_dock_out(mol_dock_out)
        if pd.isna(score) or score == 0.0:
            debug_dump_dock_out(mol_dock_out, name)
        result["score"] = float(score) if not pd.isna(score) else 0.0
        result["scored"] = result["score"] != 0.0
        maybe_cleanup_molecule_artifacts(
            smi_file=smi_file,
            mol_lb_dir=mol_lb_dir,
            mol_run_root=mol_run_root,
            mol_dock_out=mol_dock_out,
            keep_artifacts=(result["score"] == 0.0 and keep_zero_score_artifacts),
        )
    except Exception as e:
        print(f"[warn] docking failed for {name}: {e}", file=sys.stderr)
        result["score"] = 0.0
        result["scored"] = False
    finally:
        if old_slurm_tmp is None:
            os.environ.pop("SLURM_TMPDIR", None)
        else:
            os.environ["SLURM_TMPDIR"] = old_slurm_tmp

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Score an unscored trimer library CSV using DOCK3 via omltk in parallel. "
            "Must run inside a Slurm job after `source dockenv.sh`."
        )
    )
    ap.add_argument("--unscored-csv", required=True,
                    help="Unscored library CSV (from score_library.py --generate-only). "
                         "Required columns: bb1_id, bb2_id, bb3_id, smiles.")
    ap.add_argument("--indock", required=True,
                    help="Path to the (pre-patched) INDOCK file. "
                         "Tip: use `sed '1s/DOCK 3.7/DOCK 3.8/' ampc_dockfiles/INDOCK > $SLURM_TMPDIR/INDOCK`.")
    ap.add_argument("--dockfiles", required=True,
                    help="Path to the dockfiles directory (ampc_dockfiles/).")
    ap.add_argument("--dock64",
                    default="/project/rrg-mailhoto/share/dock64",
                    help="Path to dock64 binary (default: %(default)s).")
    ap.add_argument("--n-proc", type=int, default=1,
                    help="Number of molecules to process in parallel (default: 1). "
                         "Set to $SLURM_CPUS_PER_TASK in the job script.")
    ap.add_argument("--out-csv", required=True,
                    help="Output path for the scored CSV.")
    ap.add_argument("--ligbuild", default="ligbuild",
                    help="ligbuild executable name or path (default: 'ligbuild', must be on PATH).")
    ap.add_argument("--ligbuild-timeout", type=int, default=720,
                    help="Timeout per ligbuild call in seconds (default: 300).")
    ap.add_argument("--work-dir", default=None,
                    help="Working directory for ligbuild/dock3 scratch (default: $SLURM_TMPDIR or /tmp).")
    ap.add_argument("--append", action="store_true",
                    help="Append to --out-csv instead of overwriting it. If the file already exists and is non-empty, the header is not rewritten.")
    ap.add_argument("--keep-zero-score-artifacts", action="store_true",
                    help="Preserve scratch/output files for molecules that complete docking but receive a 0.0 score.")
    ap.add_argument("--failed-triples-csv", default=None,
                    help="Where to write failed building-block triples for rows with empty SMILES. "
                         "Default: failed_building_block_triples.csv at the project root.")
    args = ap.parse_args()

    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if args.work_dir:
        work_root = Path(args.work_dir)
    elif slurm_tmp:
        work_root = Path(slurm_tmp)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="dock_library_parallel_"))
        print(f"[warn] SLURM_TMPDIR not set; using {work_root}", file=sys.stderr)
    work_root.mkdir(parents=True, exist_ok=True)

    indock = Path(args.indock).resolve()
    dockfiles = Path(args.dockfiles).resolve()
    dock64 = Path(args.dock64).resolve()
    out_csv = Path(args.out_csv).resolve()

    for p, name in [(indock, "--indock"), (dockfiles, "--dockfiles"), (dock64, "--dock64")]:
        if not p.exists():
            print(f"[error] {name} path does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    try:
        import omltk.docking  # noqa: F401
    except ImportError as e:
        print(
            f"[error] cannot import omltk.docking: {e}\n"
            "  Make sure local_omltk is on PYTHONPATH (set in the Slurm job script).",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(args.unscored_csv)
    required = {"smiles"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[error] --unscored-csv is missing columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    failed_triples_csv = (
        Path(args.failed_triples_csv).resolve()
        if args.failed_triples_csv
        else PROJECT_ROOT / "failed_building_block_triples.csv"
    )
    failed_triples_csv.parent.mkdir(parents=True, exist_ok=True)

    failed_build_mask = df["smiles"].isna()
    non_null_smiles = df["smiles"].notna()
    failed_build_mask.loc[non_null_smiles] = (
        df.loc[non_null_smiles, "smiles"].astype(str).str.strip() == ""
    )

    failed_triples_cols = [c for c in ["bb1_id", "bb2_id", "bb3_id"] if c in df.columns]
    if not failed_triples_cols:
        print(
            "[warn] No bb1_id/bb2_id/bb3_id columns found; writing all failed rows instead.",
            file=sys.stderr,
        )
        failed_triples_df = df.loc[failed_build_mask].copy()
    else:
        failed_triples_df = df.loc[failed_build_mask, failed_triples_cols].copy()

    failed_triples_df.to_csv(failed_triples_csv, index=False)
    print(
        f"[info] Wrote {len(failed_triples_df)} failed building block triples to {failed_triples_csv}",
        file=sys.stderr,
    )

    df = df[~failed_build_mask].copy()
    df["smiles"] = df["smiles"].astype(str)
    df = df[df["smiles"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"[info] {len(df)} molecules to dock (after dropping empty SMILES).")

    rows_by_index = {i: list(row) for i, row in enumerate(df.itertuples(index=False, name=None))}
    smiles_col_index = df.columns.get_loc("smiles")
    total_tasks = len(rows_by_index)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_mode = "a" if out_csv.exists() else "w"
    write_header = not (out_csv.exists() and out_csv.stat().st_size > 0)

    print(f"[step 1+2] Running ligbuild and docking with up to {args.n_proc} molecules in parallel …")

    results_by_index = {}
    n_built = 0
    n_scored = 0
    completed = 0
    next_index_to_write = 0

    written = 0
    with out_csv.open(csv_mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        if write_header:
            writer.writerow(list(df.columns) + ["docking_score"])

        with ProcessPoolExecutor(max_workers=max(1, args.n_proc)) as executor:
            future_to_index = {}
            next_index_to_submit = 0

            def submit_one(index: int) -> None:
                row_values = rows_by_index[index]
                task = {
                    "index": index,
                    "name": f"m{index}",
                    "smiles": str(row_values[smiles_col_index]),
                    "indock": str(indock),
                    "dockfiles": str(dockfiles),
                    "dock64": str(dock64),
                    "work_root": str(work_root),
                    "ligbuild": args.ligbuild,
                    "ligbuild_timeout": args.ligbuild_timeout,
                    "keep_zero_score_artifacts": args.keep_zero_score_artifacts,
                }
                future_to_index[executor.submit(dock_one_molecule, task)] = index

            initial_submissions = min(max(1, args.n_proc), total_tasks)
            for _ in range(initial_submissions):
                submit_one(next_index_to_submit)
                next_index_to_submit += 1

            while future_to_index:
                future = next(as_completed(future_to_index))
                try:
                    res = future.result()
                except Exception as e:
                    index = future_to_index[future]
                    print(f"[warn] worker failed for molecule index {index}: {e}", file=sys.stderr)
                    res = {
                        "index": index,
                        "score": 0.0,
                        "built": False,
                        "scored": False,
                    }
                finally:
                    future_to_index.pop(future, None)

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
                    print(
                        f"  [dock_library_parallel] {completed}/{total_tasks} done, "
                        f"{n_built} bundles built, {n_scored} molecules scored so far, "
                        f"{written} rows written."
                    )

        while next_index_to_write < total_tasks:
            if next_index_to_write not in results_by_index:
                results_by_index[next_index_to_write] = {
                    "index": next_index_to_write,
                    "score": 0.0,
                    "built": False,
                    "scored": False,
                }
            ready = results_by_index.pop(next_index_to_write)
            writer.writerow(rows_by_index.pop(next_index_to_write) + [ready["score"]])
            written += 1
            next_index_to_write += 1

        fcsv.flush()

    print(f"[done] {n_scored}/{written} molecules scored. Output: {out_csv}")


if __name__ == "__main__":
    main()