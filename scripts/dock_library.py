#!/usr/bin/env python3
"""
Phase 2: score an unscored library CSV using DOCK3 natively via omltk.

Workflow (must run inside a Slurm job after `source dockenv.sh`):
  1. Read the unscored CSV produced by `score_library.py --generate-only`.
  2. Write one .smi file per molecule in $SLURM_TMPDIR/smi_files/.
  3. Run ligbuild (on PATH via dockenv.sh) for each molecule → .tgz bundles.
  4. Call omltk.docking.run_docking_from_tgz() directly (no subprocess for dock64).
  5. Read the DOCKDF.df output, join best total scores back to the CSV.
  6. Write the scored CSV.

Usage (example Slurm job wrapper: jobs/job_dock_library.sh):
    source /project/rrg-mailhoto/share/dockingpackages/dockenv.sh
    export PYTHONPATH=/path/to/DEL-GFN-2/src:/path/to/local_omltk:$PYTHONPATH
    sed '1s/DOCK 3\\.7 parameter/DOCK 3.8 parameter/' \\
        /path/to/ampc_dockfiles/INDOCK > $SLURM_TMPDIR/INDOCK

    python3 scripts/dock_library.py \\
        --unscored-csv /path/to/unscored_library.csv \\
        --indock      $SLURM_TMPDIR/INDOCK \\
        --dockfiles   /path/to/ampc_dockfiles \\
        --dock64      /project/rrg-mailhoto/share/dock64 \\
        --n-proc      $SLURM_CPUS_PER_TASK \\
        --out-csv     /path/to/scored_library.csv
"""
from __future__ import annotations
import argparse, os, sys, subprocess, shutil, json, tempfile, csv, contextlib
from pathlib import Path
from typing import Optional

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
        print(f"[warn] Could not find score column in DOCKDF.df. Columns: {list(dockdf.columns)}",
              file=sys.stderr)
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

# ---------------------------------------------------------------------------
# ligbuild helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Score an unscored trimer library CSV using DOCK3 via omltk. "
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
                    help="Number of parallel dock64 jobs (default: 1). "
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
    args = ap.parse_args()

    # ── Resolve working directory ──────────────────────────────────────────────
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if args.work_dir:
        work_root = Path(args.work_dir)
    elif slurm_tmp:
        work_root = Path(slurm_tmp)
    else:
        work_root = Path(tempfile.mkdtemp(prefix="dock_library_"))
        print(f"[warn] SLURM_TMPDIR not set; using {work_root}", file=sys.stderr)
    work_root.mkdir(parents=True, exist_ok=True)

    indock   = Path(args.indock).resolve()
    dockfiles = Path(args.dockfiles).resolve()
    dock64   = Path(args.dock64).resolve()
    out_csv  = Path(args.out_csv).resolve()

    for p, name in [(indock, "--indock"), (dockfiles, "--dockfiles"), (dock64, "--dock64")]:
        if not p.exists():
            print(f"[error] {name} path does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    # ── Load omltk ────────────────────────────────────────────────────────────
    try:
        from omltk.docking import run_docking_from_tgz
    except ImportError as e:
        print(
            f"[error] cannot import omltk.docking: {e}\n"
            "  Make sure local_omltk is on PYTHONPATH (set in the Slurm job script).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Read unscored library ─────────────────────────────────────────────────
    df = pd.read_csv(args.unscored_csv)
    required = {"smiles"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[error] --unscored-csv is missing columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    failed_triples_csv = PROJECT_ROOT / "failed_building_block_triples.csv"

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

    # Drop rows where the trimer build failed (null/empty SMILES)
    df = df[~failed_build_mask].copy()
    df["smiles"] = df["smiles"].astype(str)
    df = df[df["smiles"].str.strip().astype(bool)].reset_index(drop=True)
    print(f"[info] {len(df)} molecules to dock (after dropping empty SMILES).")

    # ── Step 1: Write .smi files and run ligbuild ─────────────────────────────
    smi_dir = work_root / "smi_files"
    lb_dir  = work_root / "ligbuild_out"
    smi_dir.mkdir(exist_ok=True)
    lb_dir.mkdir(exist_ok=True)

    # Assign a short molecule name: m0, m1, m2, ...
    mol_names = [f"m{i}" for i in range(len(df))]
    name_to_row: dict[str, int] = {name: i for i, name in enumerate(mol_names)}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    dock_root = work_root / "dock_out"
    dock_root.mkdir(exist_ok=True)

    write_header = True
    csv_mode = "w"
    if args.append:
        csv_mode = "a"
        write_header = not (out_csv.exists() and out_csv.stat().st_size > 0)

    written = 0
    n_scored = 0
    n_built = 0

    with out_csv.open(csv_mode, newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        if write_header:
            writer.writerow(list(df.columns) + ["docking_score"])
            fcsv.flush()

        print("[step 1+2] Running ligbuild and docking molecules one by one …")
        for i, (name, row) in enumerate(zip(mol_names, df.itertuples(index=False))):
            smiles = row.smiles
            smi_file = smi_dir / f"{name}.smi"
            smi_file.write_text(f"{smiles} {name}\n")

            mol_lb_dir = lb_dir / name
            tgz = run_ligbuild_for_smi(
                smi_file, mol_lb_dir,
                smiles=smiles,
                ligbuild_exe=args.ligbuild,
                timeout=args.ligbuild_timeout,
            )

            score = 0.0
            if tgz is not None:
                n_built += 1
                mol_dock_out = dock_root / name
                mol_dock_out.mkdir(parents=True, exist_ok=True)
                mol_run_root = work_root / f"dock_run_{name}"
                if mol_run_root.exists():
                    shutil.rmtree(mol_run_root, ignore_errors=True)
                mol_run_root.mkdir(parents=True, exist_ok=True)
                try:
                    old_slurm_tmp = os.environ.get("SLURM_TMPDIR")
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
                    if old_slurm_tmp is None:
                        os.environ.pop("SLURM_TMPDIR", None)
                    else:
                        os.environ["SLURM_TMPDIR"] = old_slurm_tmp
                    score = extract_best_score_from_dock_out(mol_dock_out)
                    if pd.isna(score) or score == 0.0:
                        debug_dump_dock_out(mol_dock_out, name)
                except Exception as e:
                    if old_slurm_tmp is None:
                        os.environ.pop("SLURM_TMPDIR", None)
                    else:
                        os.environ["SLURM_TMPDIR"] = old_slurm_tmp
                    print(f"[warn] docking failed for {name}: {e}", file=sys.stderr)
                    score = 0.0

            if score != 0.0:
                n_scored += 1

            writer.writerow(list(row) + [score])
            fcsv.flush()
            written += 1

            if (i + 1) % max(1, len(df) // 10) == 0 or (i + 1) == len(df):
                print(
                    f"  [dock_library] {i+1}/{len(df)} done, "
                    f"{n_built} bundles built, {n_scored} molecules scored so far."
                )

    print(f"[done] {n_scored}/{written} molecules scored. Output: {out_csv}")


if __name__ == "__main__":
    main()
