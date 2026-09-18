#!/usr/bin/env python3
"""Retry DOCK3 for rows with zero docking scores and repair CSVs in place.

This utility scans scored-library CSV files, finds rows whose score column is
exactly ``0.0``, re-docks those SMILES in parallel with a more generous
``ligbuild`` timeout, and replaces only successfully recovered scores.

Workers only perform docking and return scores.

By default the parent process is the only writer and updates each CSV atomically
as retries complete.

For Slurm array sharding, use --num-shards/--shard-id together with --no-apply.
In that mode, the script only produces the per-shard summary CSV (which can be
used as a patch file) and does not modify any input libraries. A separate merge
step must apply the recovered scores.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdelgfn.mols.vina_scorer import Dock3Scorer  # noqa: E402


SCORE_COL_CANDIDATES = ("docking_score", "trimer_score", "score")


@dataclass(frozen=True)
class RetryTask:
    task_id: int
    csv_path: str
    row_index: int
    smiles: str
    name: str
    score_col: str


_SCORER: Optional[Dock3Scorer] = None


def _init_worker(
    indock: str,
    dockfiles: str,
    dockenv_sh: str,
    dock64: str,
    ligbuild: str,
    subprocess_timeout: int,
    ligbuild_timeout: int,
    keep_artifacts: bool,
    work_dir: Optional[str],
) -> None:
    """Create one Dock3Scorer per worker process."""
    global _SCORER
    _SCORER = Dock3Scorer(
        indock_template=indock,
        dockfiles_dir=dockfiles,
        dockenv_sh=dockenv_sh,
        dock64_exe=dock64,
        ligbuild_exe=ligbuild,
        tmp_dir=work_dir if keep_artifacts else None,
        timeout=subprocess_timeout,
        ligbuild_timeout=ligbuild_timeout,
    )


def _score_task(task: RetryTask) -> dict:
    if _SCORER is None:
        raise RuntimeError("Dock3Scorer worker was not initialized")
    score = _SCORER.score_smiles(task.smiles, name=task.name)
    recovered = math.isfinite(float(score)) and float(score) != 0.0
    failure_reason = "" if recovered else (_SCORER.last_failure_reason or "unrecovered_zero_score")
    return {
        "task_id": task.task_id,
        "csv_path": task.csv_path,
        "row_index": task.row_index,
        "name": task.name,
        "smiles": task.smiles,
        "score_col": task.score_col,
        "new_score": float(score),
        "recovered": bool(recovered),
        "error": failure_reason,
    }


def parse_float(value: object) -> Optional[float]:
    try:
        x = float(str(value).strip())
    except Exception:
        return None
    return x if math.isfinite(x) else None


def resolve_score_col(fieldnames: list[str], requested: str) -> Optional[str]:
    if requested != "auto":
        return requested if requested in fieldnames else None
    for col in SCORE_COL_CANDIDATES:
        if col in fieldnames:
            return col
    return None


def csv_paths(csv_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(
        p
        for p in csv_dir.glob(pattern)
        if p.is_file() and not p.name.startswith("retry_zero_dock_scores_summary")
    )


def make_ligand_name(csv_path: Path, row_index: int, row: dict[str, str]) -> str:
    ids = [str(row.get(c, "")).strip() for c in ("bb1_id", "bb2_id", "bb3_id")]
    if all(ids):
        suffix = "_".join(ids)
    else:
        suffix = f"row{row_index}"
    safe_stem = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in csv_path.stem)
    safe_suffix = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in suffix)
    return f"retry_{safe_stem}_{safe_suffix}"[:180]


def collect_tasks(
    *,
    csv_dir: Path,
    recursive: bool,
    score_col_arg: str,
    smiles_col: str,
    max_molecules: int,
    num_shards: int,
    shard_id: int,
) -> tuple[list[RetryTask], list[dict]]:
    tasks: list[RetryTask] = []
    scan_rows: list[dict] = []
    # Cap applies to the tasks selected for *this shard*.
    capped = max_molecules > 0
    global_task_id = 0
    for path in csv_paths(csv_dir, recursive):
        try:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                score_col = resolve_score_col(fieldnames, score_col_arg)
                if score_col is None:
                    scan_rows.append({"csv_path": str(path), "status": "skipped_no_score_col", "zero_rows": 0})
                    continue
                if smiles_col not in fieldnames:
                    scan_rows.append({"csv_path": str(path), "status": "skipped_no_smiles_col", "zero_rows": 0})
                    continue
                zero_count = 0
                cap_reached = False
                for row_index, row in enumerate(reader):
                    score = parse_float(row.get(score_col, ""))
                    smiles = str(row.get(smiles_col, "")).strip()
                    if score == 0.0 and smiles:
                        zero_count += 1
                        task_id = global_task_id
                        global_task_id += 1
                        # Deterministic task-level sharding by global scan order.
                        if (task_id % max(1, int(num_shards))) == int(shard_id):
                            tasks.append(
                                RetryTask(
                                    task_id=task_id,
                                    csv_path=str(path),
                                    row_index=row_index,
                                    smiles=smiles,
                                    # DOCK3/DB2/OUTDOCK parsing is brittle with long
                                    # molecule names (fixed-width legacy fields).
                                    # Each score runs in an isolated workdir, so the
                                    # known-good short name is safest; source row
                                    # identity is preserved in the summary fields.
                                    name="lig",
                                    score_col=score_col,
                                )
                            )
                            if capped and len(tasks) >= max_molecules:
                                cap_reached = True
                                # Note: we break out early; subsequent CSVs are not scanned.
                                break
                scan_rows.append(
                    {
                        "csv_path": str(path),
                        "status": "ok_cap_reached" if cap_reached else "ok",
                        "zero_rows": zero_count,
                    }
                )
                if cap_reached:
                    break
        except Exception as e:
            scan_rows.append({"csv_path": str(path), "status": f"scan_error: {e}", "zero_rows": 0})
    return tasks, scan_rows


SUMMARY_FIELDNAMES = [
    "timestamp",
    "task_id",
    "csv_path",
    "row_index",
    "name",
    "score_col",
    "old_score",
    "new_score",
    "recovered",
    "error",
    "smiles",
]


def append_summary(summary_csv: Path, rows: list[dict]) -> None:
    """Append rows to summary_csv, writing the header if the file is new/empty."""
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists() or summary_csv.stat().st_size == 0
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_FIELDNAMES,
            lineterminator="\n",
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def update_csvs(
    *,
    results: list[dict],
    backup: bool,
    dry_run: bool,
    backed_up_paths: Optional[set[str]] = None,
) -> tuple[int, int]:
    recovered = [r for r in results if r.get("recovered")]
    by_csv: dict[str, list[dict]] = {}
    for res in recovered:
        by_csv.setdefault(str(res["csv_path"]), []).append(res)

    files_updated = 0
    cells_updated = 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for csv_path_str, csv_results in sorted(by_csv.items()):
        path = Path(csv_path_str)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        changed = 0
        for res in csv_results:
            idx = int(res["row_index"])
            score_col = str(res["score_col"])
            if idx >= len(rows) or score_col not in fieldnames:
                res["error"] = "row_or_score_col_missing_during_update"
                continue
            current = parse_float(rows[idx].get(score_col, ""))
            if current != 0.0:
                res["error"] = f"skipped_current_score_is_{current}"
                continue
            rows[idx][score_col] = f'{float(res["new_score"]):.6g}'
            res["old_score"] = "0.0"
            changed += 1

        if changed == 0:
            continue
        if dry_run:
            files_updated += 1
            cells_updated += changed
            continue

        if backup:
            already_backed_up = backed_up_paths is not None and str(path) in backed_up_paths
            if not already_backed_up:
                backup_path = path.with_name(path.name + f".bak.{stamp}")
                shutil.copy2(path, backup_path)
                if backed_up_paths is not None:
                    backed_up_paths.add(str(path))

        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

        files_updated += 1
        cells_updated += changed

    return files_updated, cells_updated


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Retry DOCK3 for all 0.0-scored SMILES in CSV files and repair successful rows in place."
    )
    ap.add_argument("--csv-dir", required=True, help="Directory containing scored CSV files.")
    ap.add_argument("--recursive", action="store_true", help="Scan CSV files recursively under --csv-dir.")
    ap.add_argument("--score-col", default="auto", help="Score column, or 'auto' for docking_score/trimer_score/score.")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES column name (default: %(default)s).")
    ap.add_argument("--indock", default=str(PROJECT_ROOT / "ampc_dockfiles" / "INDOCK"))
    ap.add_argument("--dockfiles", default=str(PROJECT_ROOT / "ampc_dockfiles"))
    ap.add_argument("--dockenv-sh", default="/project/rrg-mailhoto/share/dockingpackages/dockenv.sh")
    ap.add_argument("--dock64", default="/project/rrg-mailhoto/share/dock64")
    ap.add_argument("--ligbuild", default="ligbuild")
    ap.add_argument("--n-proc", type=int, default=1, help="Number of molecules to retry in parallel.")
    ap.add_argument("--ligbuild-timeout", type=int, default=600, help="Inner ligbuild DB2 timeout in seconds.")
    ap.add_argument("--subprocess-timeout", type=int, default=900, help="Outer subprocess timeout in seconds.")
    ap.add_argument("--work-dir", default=None, help="Persistent work root used only with --keep-artifacts.")
    ap.add_argument("--keep-artifacts", action="store_true", help="Keep per-molecule Dock3Scorer scratch directories.")
    ap.add_argument("--summary-csv", default=None, help="Path for retry summary CSV.")
    ap.add_argument("--no-backup", action="store_true", help="Do not create timestamped .bak files before rewriting CSVs.")
    ap.add_argument("--dry-run", action="store_true", help="Only report rows that would be retried; do not dock or edit files.")
    ap.add_argument(
        "--no-apply",
        action="store_true",
        help=(
            "Do not update source CSVs in place; only write the summary CSV. "
            "Recommended when using --num-shards/--shard-id across multiple Slurm jobs."
        ),
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Total number of shards for task-level partitioning across zero-score rows "
            "(default: %(default)s)."
        ),
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help=(
            "This shard id in [0, num_shards). Only tasks where task_id %% num_shards == shard_id are processed "
            "(default: %(default)s)."
        ),
    )
    ap.add_argument(
        "--update-every",
        type=int,
        default=50,
        help=(
            "Checkpoint completed retry results every N finished molecules by rewriting "
            "recovered scores into source CSVs and refreshing the summary CSV. "
            "Use 0 to update only once at the end (default: %(default)s)."
        ),
    )
    ap.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Maximum number of zero-score molecules to retry; 0 means no cap (default: %(default)s).",
    )
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    if not csv_dir.is_dir():
        print(f"[error] --csv-dir is not a directory: {csv_dir}", file=sys.stderr)
        return 2
    if args.max_molecules < 0:
        print(f"[error] --max-molecules must be >= 0, got {args.max_molecules}", file=sys.stderr)
        return 2
    if args.update_every < 0:
        print(f"[error] --update-every must be >= 0, got {args.update_every}", file=sys.stderr)
        return 2
    if args.num_shards <= 0:
        print(f"[error] --num-shards must be >= 1, got {args.num_shards}", file=sys.stderr)
        return 2
    if not (0 <= args.shard_id < args.num_shards):
        print(f"[error] --shard-id must be in [0, {args.num_shards}), got {args.shard_id}", file=sys.stderr)
        return 2

    tasks, scan_rows = collect_tasks(
        csv_dir=csv_dir,
        recursive=args.recursive,
        score_col_arg=args.score_col,
        smiles_col=args.smiles_col,
        max_molecules=int(args.max_molecules),
        num_shards=int(args.num_shards),
        shard_id=int(args.shard_id),
    )
    n_files = len(scan_rows)
    n_zero = len(tasks)
    cap_label = "unlimited" if args.max_molecules == 0 else str(args.max_molecules)
    shard_label = f" shard={args.shard_id}/{args.num_shards}" if args.num_shards > 1 else ""
    print(
        f"[scan] files={n_files} zero-score rows with SMILES selected={n_zero} "
        f"max_molecules={cap_label}{shard_label}"
    )
    for row in scan_rows:
        if row["status"] != "ok" or int(row["zero_rows"]) > 0:
            print(f"  [scan] {row['csv_path']} status={row['status']} zero_rows={row['zero_rows']}")

    if args.summary_csv:
        summary_csv = Path(args.summary_csv).resolve()
    else:
        if args.num_shards > 1:
            summary_csv = csv_dir / f"retry_zero_dock_scores_summary_shard_{args.shard_id:03d}_of_{args.num_shards:03d}.csv"
        else:
            summary_csv = csv_dir / "retry_zero_dock_scores_summary.csv"
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")

    if args.dry_run or not tasks:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=SUMMARY_FIELDNAMES,
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            for t in tasks:
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "task_id": t.task_id,
                        "csv_path": t.csv_path,
                        "row_index": t.row_index,
                        "name": t.name,
                        "score_col": t.score_col,
                        "old_score": "0.0",
                        "new_score": "",
                        "recovered": False,
                        "error": "dry_run" if args.dry_run else "",
                        "smiles": t.smiles,
                    }
                )
        print(f"[done] dry_run={args.dry_run}; summary={summary_csv}")
        return 0

    indock = str(Path(args.indock).resolve())
    dockfiles = str(Path(args.dockfiles).resolve())
    dock64 = str(Path(args.dock64).resolve())
    for p, label in [(indock, "--indock"), (dockfiles, "--dockfiles"), (dock64, "--dock64")]:
        if not Path(p).exists():
            print(f"[error] {label} path does not exist: {p}", file=sys.stderr)
            return 2

    work_dir = None
    if args.keep_artifacts:
        work_dir = str(Path(args.work_dir or os.environ.get("SLURM_TMPDIR") or tempfile.mkdtemp(prefix="retry_zero_dock_")).resolve())
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        print(f"[info] preserving Dock3Scorer artifacts under {work_dir}")

    apply_label = "no_apply" if args.no_apply else "apply_in_place"
    shard_label = f" shard={args.shard_id}/{args.num_shards}" if args.num_shards > 1 else ""
    print(
        f"[retry] tasks={len(tasks)} n_proc={args.n_proc} "
        f"max_molecules={cap_label}{shard_label} "
        f"update_every={args.update_every} mode={apply_label} "
        f"ligbuild_timeout={args.ligbuild_timeout}s subprocess_timeout={args.subprocess_timeout}s"
    )
    if args.no_apply and args.num_shards > 1:
        print(
            "[info] running in sharded no-apply mode; summary CSV acts as a patch file. "
            "Run a merge step to apply recovered scores to the original libraries."
        )

    pending_flush: list[dict] = []
    backed_up_paths: set[str] = set()
    total_files_updated = 0
    total_cells_updated = 0
    completed = recovered = 0
    # Ensure we start from a clean summary file for this run.
    summary_csv.unlink(missing_ok=True)

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.n_proc)),
        initializer=_init_worker,
        initargs=(
            indock,
            dockfiles,
            args.dockenv_sh,
            dock64,
            args.ligbuild,
            int(args.subprocess_timeout),
            int(args.ligbuild_timeout),
            bool(args.keep_artifacts),
            work_dir,
        ),
    ) as executor:
        max_in_flight = max(1, int(args.n_proc)) * 2
        task_iter = iter(tasks)
        future_to_task: dict = {}

        def submit_one() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            future_to_task[executor.submit(_score_task, task)] = task
            return True

        for _ in range(min(max_in_flight, len(tasks))):
            if not submit_one():
                break

        while future_to_task:
            fut = next(as_completed(future_to_task))
            task = future_to_task.pop(fut)
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "task_id": task.task_id,
                    "csv_path": task.csv_path,
                    "row_index": task.row_index,
                    "name": task.name,
                    "smiles": task.smiles,
                    "score_col": task.score_col,
                    "new_score": 0.0,
                    "recovered": False,
                    "error": repr(e),
                }
            completed += 1
            recovered += int(bool(res.get("recovered")))
            res.setdefault("old_score", "0.0")
            res["timestamp"] = timestamp
            pending_flush.append(res)
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                print(f"  [retry] {completed}/{len(tasks)} done; recovered={recovered}")
            if args.update_every > 0 and len(pending_flush) >= args.update_every:
                if args.no_apply:
                    append_summary(summary_csv, pending_flush)
                    print(
                        f"  [checkpoint] completed={completed}/{len(tasks)}; "
                        f"flushed_results={len(pending_flush)}; "
                        f"summary={summary_csv}"
                    )
                else:
                    files_updated, cells_updated = update_csvs(
                        results=pending_flush,
                        backup=not args.no_backup,
                        dry_run=False,
                        backed_up_paths=backed_up_paths,
                    )
                    total_files_updated += files_updated
                    total_cells_updated += cells_updated
                    append_summary(summary_csv, pending_flush)
                    print(
                        f"  [checkpoint] completed={completed}/{len(tasks)}; "
                        f"flushed_results={len(pending_flush)}; "
                        f"files_updated={files_updated}; cells_updated={cells_updated}; "
                        f"summary={summary_csv}"
                    )
                pending_flush.clear()

            while len(future_to_task) < max_in_flight and submit_one():
                pass

    if args.no_apply:
        files_updated, cells_updated = 0, 0
    else:
        files_updated, cells_updated = update_csvs(
            results=pending_flush,
            backup=not args.no_backup,
            dry_run=False,
            backed_up_paths=backed_up_paths,
        )
        total_files_updated += files_updated
        total_cells_updated += cells_updated
    append_summary(summary_csv, pending_flush)
    if args.no_apply:
        print(f"[done] recovered={recovered}/{len(tasks)}; mode=no_apply; summary={summary_csv}")
    else:
        print(
            f"[done] recovered={recovered}/{len(tasks)}; "
            f"files_updated={total_files_updated}; cells_updated={total_cells_updated}; summary={summary_csv}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())