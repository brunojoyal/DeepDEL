#!/usr/bin/env python3
"""Split scored libraries by shape and completeness.

This script scans a directory of scored library CSVs (each CSV = one library),
infers the library shape as:

    |unique bb1_id| x |unique bb2_id| x |unique bb3_id|

A library is considered *complete* if it contains *all* combinations of its
unique IDs (i.e., number of unique (bb1_id,bb2_id,bb3_id) triples equals
|B1|*|B2|*|B3|). Complete libraries are moved into a shape-named directory
under the input directory, e.g.:

    data/scored_libraries/not_random/6.6.6/<library>.csv

Incomplete libraries are moved into:

    data/scored_libraries/not_random/incomplete/<library>.csv

For each shape directory, the script also writes a ranked list of libraries
by reward(mode='threshold') computed from the 'docking_score' column.

The ranking is written to:

    <shape_dir>/ranked_list.csv

By default this script performs real moves. Use --dry-run to preview.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from deepdelgfn.rewards import reward


TRIPLE = Tuple[str, str, str]


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


@dataclass(frozen=True)
class LibraryInfo:
    path: Path
    file: str
    n1: int
    n2: int
    n3: int
    bb1_pool: str
    bb2_pool: str
    bb3_pool: str
    n_rows: int
    n_unique_triples: int
    complete: bool
    shape_str: str
    reward: Optional[float]
    reason_incomplete: Optional[str] = None


def iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("missing header")
        required = {"bb1_id", "bb2_id", "bb3_id", "docking_score"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        for row in reader:
            yield row


def analyze_library(path: Path, threshold: float) -> LibraryInfo:
    b1: Set[str] = set()
    b2: Set[str] = set()
    b3: Set[str] = set()
    triples: Set[TRIPLE] = set()
    vals: List[float] = []
    n_rows = 0

    for row in iter_csv_rows(path):
        n_rows += 1
        i = str(row.get("bb1_id", ""))
        j = str(row.get("bb2_id", ""))
        k = str(row.get("bb3_id", ""))
        b1.add(i)
        b2.add(j)
        b3.add(k)
        triples.add((i, j, k))
        v = _safe_float(row.get("docking_score"))
        if v is not None:
            vals.append(v)

    n1, n2, n3 = len(b1), len(b2), len(b3)
    bb1_pool = "|".join(sorted(b1, key=lambda x: int(x) if x.isdigit() else x))
    bb2_pool = "|".join(sorted(b2, key=lambda x: int(x) if x.isdigit() else x))
    bb3_pool = "|".join(sorted(b3, key=lambda x: int(x) if x.isdigit() else x))
    expected = n1 * n2 * n3
    n_unique_triples = len(triples)
    complete = (expected > 0) and (n_unique_triples == expected)
    shape_str = f"{n1}.{n2}.{n3}"

    r: Optional[float] = None
    if vals:
        arr = np.asarray(vals, dtype=float)
        r = float(reward(arr, mode="threshold", threshold=float(threshold)))

    reason: Optional[str] = None
    if not complete:
        if expected == 0:
            reason = "empty (no IDs found)"
        elif n_unique_triples < expected:
            reason = f"missing combos: unique_triples={n_unique_triples} expected={expected}"
        else:
            # This case can happen if file has extra triples beyond the Cartesian product
            # induced by its unique ID sets (e.g., corrupt IDs).
            reason = f"extra combos: unique_triples={n_unique_triples} expected={expected}"

    return LibraryInfo(
        path=path,
        file=path.name,
        n1=n1,
        n2=n2,
        n3=n3,
        bb1_pool=bb1_pool,
        bb2_pool=bb2_pool,
        bb3_pool=bb3_pool,
        n_rows=n_rows,
        n_unique_triples=n_unique_triples,
        complete=complete,
        shape_str=shape_str,
        reward=r,
        reason_incomplete=reason,
    )


def safe_move(src: Path, dst: Path, *, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] move {src} -> {dst}")
        return
    # shutil.move handles cross-filesystem moves.
    shutil.move(str(src), str(dst))


def write_ranked_list(shape_dir: Path, infos: Sequence[LibraryInfo], *, dry_run: bool) -> None:
    # Sort by decreasing reward (best first). None goes last.
    def sort_key(x: LibraryInfo):
        if x.reward is None or (isinstance(x.reward, float) and math.isnan(x.reward)):
            return (1, 0.0)
        return (0, float(x.reward))

    infos_sorted = sorted(infos, key=sort_key, reverse=True)
    out_path = shape_dir / "ranked_list.csv"
    if dry_run:
        print(f"[dry-run] write ranked list: {out_path} ({len(infos_sorted)} rows)")
        return

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "file",
                "reward",
                "shape",
                "bb1_pool",
                "bb2_pool",
                "bb3_pool",
                "n_rows",
                "n_unique_triples",
                "complete",
            ],
        )
        w.writeheader()
        for rank, info in enumerate(infos_sorted, 1):
            w.writerow(
                {
                    "rank": rank,
                    "file": info.file,
                    "reward": "" if info.reward is None else info.reward,
                    "shape": info.shape_str,
                    "bb1_pool": info.bb1_pool,
                    "bb2_pool": info.bb2_pool,
                    "bb3_pool": info.bb3_pool,
                    "n_rows": info.n_rows,
                    "n_unique_triples": info.n_unique_triples,
                    "complete": int(info.complete),
                }
            )


def regenerate_ranked_lists_only(
    *,
    in_dir: Path,
    threshold: float,
    dry_run: bool,
) -> int:
    """Scan existing shape subdirectories and (re)write ranked_list.csv files.

    This is useful after libraries have already been moved into shape folders.
    """
    # A shape directory is named like "6.6.6" (digits separated by dots).
    shape_dirs = [p for p in sorted(in_dir.iterdir()) if p.is_dir() and p.name.count(".") == 2]
    if not shape_dirs:
        print(f"[warn] no shape directories found under {in_dir}")
        return 0

    n_shapes = 0
    for sd in shape_dirs:
        csvs = [p for p in sorted(sd.glob("*.csv")) if p.is_file() and p.name != "ranked_list.csv"]
        if not csvs:
            continue
        infos: List[LibraryInfo] = []
        for p in csvs:
            try:
                infos.append(analyze_library(p, threshold=float(threshold)))
            except Exception as e:
                print(f"[warn] skipping {p}: {e}")
        if infos:
            write_ranked_list(sd, infos, dry_run=dry_run)
            n_shapes += 1

    print(f"[done] regenerated ranked lists for {n_shapes} shape directories (dry_run={bool(dry_run)})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Split scored libraries by shape, move complete ones, and rank them.")
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("data/scored_libraries/not_random"),
        help="Directory containing library CSVs.",
    )
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern for input libraries (default: *.csv)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=-9.0,
        help="Threshold used for reward(mode='threshold') (default: -9.0)",
    )
    ap.add_argument(
        "--incomplete-dirname",
        default="incomplete",
        help="Subdirectory name to move incomplete libraries into (default: incomplete)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions but don't move files or write ranked lists.",
    )
    ap.add_argument(
        "--rank-only",
        action="store_true",
        help="Do not move any files; just (re)generate ranked_list.csv inside existing shape subdirectories.",
    )

    args = ap.parse_args(argv)

    in_dir: Path = args.dir
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"[error] --dir is not a directory: {in_dir}")

    if args.rank_only:
        return regenerate_ranked_lists_only(
            in_dir=in_dir,
            threshold=float(args.threshold),
            dry_run=bool(args.dry_run),
        )

    # Only consider CSVs at top-level of in_dir (ignore shape subfolders).
    files = [p for p in sorted(in_dir.glob(args.pattern)) if p.is_file()]
    if not files:
        print(f"[warn] no files found in {in_dir} matching {args.pattern!r}")
        return 0

    incomplete_dir = in_dir / args.incomplete_dirname
    grouped_complete: Dict[str, List[LibraryInfo]] = defaultdict(list)
    n_ok = 0
    n_bad = 0
    n_err = 0

    for p in files:
        # Skip our outputs if someone reruns with pattern matching them.
        if p.parent != in_dir:
            continue
        if p.name == "ranked_list.csv":
            continue
        try:
            info = analyze_library(p, threshold=float(args.threshold))
        except Exception as e:
            n_err += 1
            dst = incomplete_dir / p.name
            print(f"[warn] treating as incomplete (parse error): {p.name}: {e}")
            safe_move(p, dst, dry_run=args.dry_run)
            continue

        if info.complete:
            n_ok += 1
            shape_dir = in_dir / info.shape_str
            dst = shape_dir / p.name
            grouped_complete[info.shape_str].append(info)
            safe_move(p, dst, dry_run=args.dry_run)
        else:
            n_bad += 1
            dst = incomplete_dir / p.name
            print(f"[info] incomplete {p.name} shape={info.shape_str} ({info.reason_incomplete})")
            safe_move(p, dst, dry_run=args.dry_run)

    # Write ranked lists for each shape dir.
    for shape_str, infos in grouped_complete.items():
        shape_dir = in_dir / shape_str
        write_ranked_list(shape_dir, infos, dry_run=args.dry_run)

    print(
        f"[done] complete={n_ok} incomplete={n_bad} errors={n_err} shapes={len(grouped_complete)} "
        f"(dry_run={bool(args.dry_run)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
