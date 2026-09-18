#!/usr/bin/env python3
"""Apply one or more retry_zero_dock_scores summary CSVs as patches.

This is intended to complement `scripts/retry_zero_dock_scores.py --no-apply`
when sharding retries across multiple Slurm jobs.

Each shard/job produces a summary CSV with columns including:

- csv_path
- row_index
- score_col
- new_score
- recovered

This merge script reads one or more such summary files, filters to recovered
rows, and updates the original source CSVs **atomically** (tempfile + replace).

Safety checks:
- fails if multiple patch rows target the same (csv_path, row_index, score_col)
  with different new_score values.
- optionally requires that the current score in the source CSV is still 0.0.

Usage example:

  python3 scripts/apply_retry_zero_patches.py \
    --patch-glob 'retry_results/run_123/*.csv' \
    --require-current-zero

"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def parse_float(value: object) -> float | None:
    try:
        x = float(str(value).strip())
    except Exception:
        return None
    return x if math.isfinite(x) else None


@dataclass(frozen=True)
class PatchRow:
    csv_path: str
    row_index: int
    score_col: str
    new_score: float
    patch_source: str


def iter_patch_files(patch_files: list[str], patch_glob: str | None) -> list[Path]:
    files: list[Path] = []
    for p in patch_files:
        files.append(Path(p))
    if patch_glob:
        # pathlib.Path.glob() does not reliably handle absolute patterns.
        files.extend(Path(p) for p in glob.glob(patch_glob))
    uniq = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def read_patches(files: list[Path]) -> list[PatchRow]:
    rows: list[PatchRow] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(str(path))
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"csv_path", "row_index", "score_col", "new_score", "recovered"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Patch file {path} missing columns: {sorted(missing)}")
            for r in reader:
                recovered = str(r.get("recovered", "")).strip().lower() in {"true", "1", "yes"}
                if not recovered:
                    continue
                new_score = parse_float(r.get("new_score", ""))
                if new_score is None or new_score == 0.0:
                    continue
                rows.append(
                    PatchRow(
                        csv_path=str(r["csv_path"]),
                        row_index=int(str(r["row_index"])),
                        score_col=str(r["score_col"]),
                        new_score=float(new_score),
                        patch_source=str(path),
                    )
                )
    return rows


def validate_no_conflicts(patches: list[PatchRow]) -> None:
    by_key: dict[tuple[str, int, str], set[float]] = defaultdict(set)
    srcs: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for p in patches:
        key = (p.csv_path, p.row_index, p.score_col)
        by_key[key].add(p.new_score)
        srcs[key].add(p.patch_source)
    conflicts = [(k, scores, srcs[k]) for k, scores in by_key.items() if len(scores) > 1]
    if conflicts:
        msg_lines = ["Conflicting patch values detected:"]
        for (csv_path, row_index, score_col), scores, sources in conflicts[:20]:
            msg_lines.append(
                f"  {csv_path} row={row_index} col={score_col} scores={sorted(scores)} sources={sorted(sources)}"
            )
        if len(conflicts) > 20:
            msg_lines.append(f"  ... ({len(conflicts) - 20} more)")
        raise ValueError("\n".join(msg_lines))


def apply_patches(
    *,
    patches: list[PatchRow],
    backup: bool,
    require_current_zero: bool,
    dry_run: bool,
) -> tuple[int, int]:
    by_csv: dict[str, list[PatchRow]] = defaultdict(list)
    for p in patches:
        by_csv[p.csv_path].append(p)

    files_updated = 0
    cells_updated = 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for csv_path_str, csv_patches in sorted(by_csv.items()):
        path = Path(csv_path_str)
        if not path.exists():
            print(f"[warn] source CSV missing, skipping: {path}", file=sys.stderr)
            continue

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        changed = 0
        for p in csv_patches:
            if p.score_col not in fieldnames:
                print(f"[warn] score_col missing in {path}: {p.score_col}", file=sys.stderr)
                continue
            if p.row_index < 0 or p.row_index >= len(rows):
                print(f"[warn] row_index out of range in {path}: {p.row_index}", file=sys.stderr)
                continue
            current = parse_float(rows[p.row_index].get(p.score_col, ""))
            if require_current_zero and current != 0.0:
                # Probably already fixed by another run; don't overwrite.
                continue
            rows[p.row_index][p.score_col] = f"{p.new_score:.6g}"
            changed += 1

        if changed == 0:
            continue

        files_updated += 1
        cells_updated += changed
        if dry_run:
            continue

        if backup:
            backup_path = path.with_name(path.name + f".bak.{stamp}")
            shutil.copy2(path, backup_path)

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

    return files_updated, cells_updated


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply retry_zero_dock_scores summary CSVs as patches to library CSVs.")
    ap.add_argument("--patch", action="append", default=[], help="Path to a patch/summary CSV (can repeat).")
    ap.add_argument("--patch-glob", default=None, help="Glob for patch/summary CSVs, e.g. 'retry_results/run_123/*.csv'.")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change but do not rewrite any CSV.")
    ap.add_argument("--backup", action="store_true", help="Create timestamped .bak files before rewriting source CSVs.")
    ap.add_argument(
        "--require-current-zero",
        action="store_true",
        help="Only apply a patch if the current score cell is still 0.0 in the source CSV.",
    )
    args = ap.parse_args()

    patch_files = iter_patch_files(args.patch, args.patch_glob)
    if not patch_files:
        print("[error] no patches provided; use --patch and/or --patch-glob", file=sys.stderr)
        return 2

    patches = read_patches(patch_files)
    print(f"[load] patch_files={len(patch_files)} recovered_rows={len(patches)}")

    validate_no_conflicts(patches)
    files_updated, cells_updated = apply_patches(
        patches=patches,
        backup=bool(args.backup),
        require_current_zero=bool(args.require_current_zero),
        dry_run=bool(args.dry_run),
    )
    print(f"[done] dry_run={args.dry_run} files_updated={files_updated} cells_updated={cells_updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
