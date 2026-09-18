#!/usr/bin/env python3
"""Evaluate docking rewards for all libraries (CSVs) in a folder.

Each *library* is assumed to be a single docking CSV produced by `score_library.py`
containing a `docking_score` column.

Example:

  python -u eval_library_rewards.py \
    --dir data/scored_libraries/not_random \
    --threshold -9.0 \
    --threshold-alpha 1.0 \
    --out outputs/library_rewards_sorted.csv

By default, results are sorted by decreasing reward (highest/best first).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import heapq
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from deepdelgfn.rewards import REWARD_MODES, reward


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _row_molecular_weight(row: dict[str, Any], *, weight_source: str) -> float:
    """Return the molecular weight used for max-weight threshold filtering.

    ``smiles`` uses the exact molecular weight of the final product SMILES.
    ``bb_sum`` uses the additive exact weights of bb1/bb2/bb3 SMILES columns.
    """
    # Import lazily so the script's existing no-cutoff path does not require
    # RDKit unless molecular-weight filtering is requested.
    from deepdelgfn.utils.weights import smiles_exact_molwt

    if weight_source == "smiles":
        smi = row.get("smiles") or row.get("SMILES") or row.get("Smiles")
        if smi in (None, ""):
            raise ValueError("--max-weight with --weight-source smiles requires a 'smiles' column")
        return smiles_exact_molwt(str(smi))

    if weight_source == "bb_sum":
        total = 0.0
        for col in ("bb1_smiles", "bb2_smiles", "bb3_smiles"):
            smi = row.get(col)
            if smi in (None, ""):
                raise ValueError(
                    "--max-weight with --weight-source bb_sum requires columns "
                    "'bb1_smiles', 'bb2_smiles', and 'bb3_smiles'"
                )
            total += smiles_exact_molwt(str(smi))
        return float(total)

    raise ValueError("weight_source must be 'smiles' or 'bb_sum'")


def compute_threshold_reward_from_docking_csv(
    path: Path,
    threshold: float,
    alpha: float = 1.0,
    *,
    max_weight: float | None = None,
    weight_source: str = "smiles",
) -> float:
    """Compute threshold reward over all per-trimer docking scores in a docking CSV."""
    # Avoid pandas dependency: parse using csv.DictReader.
    vals: list[float] = []
    weights: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "docking_score" not in reader.fieldnames:
            raise ValueError(f"Expected column 'docking_score' in {path}")
        for row in reader:
            raw = row.get("docking_score")
            if raw is None:
                continue
            try:
                x = float(raw)
            except Exception:
                continue
            if math.isfinite(x):
                vals.append(x)
                if max_weight is not None:
                    weights.append(_row_molecular_weight(row, weight_source=weight_source))
    arr = np.asarray(vals, dtype=float)
    weights_arr = np.asarray(weights, dtype=float) if max_weight is not None else None
    return float(
        reward(
            arr,
            mode="threshold",
            threshold=threshold,
            alpha=alpha,
            weights=weights_arr,
            max_weight=max_weight,
        )
    )


def update_topk_smallest_rows(
    *,
    heap: list[tuple[float, int, dict[str, Any]]],
    row: dict[str, Any],
    score: float,
    k: int,
    counter: int,
) -> None:
    """Maintain a fixed-size heap of the k smallest scores.

    We keep a max-heap using (-score) so the *largest* among the current k
    smallest is at the top and can be evicted cheaply.
    """
    item = (-score, counter, row)
    if len(heap) < k:
        heapq.heappush(heap, item)
        return
    # heap[0] has the most negative key => largest score among kept rows.
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def iter_docking_csv_rows(path: Path) -> tuple[list[float], list[dict[str, Any]]]:
    """Parse a docking CSV and return (vina_scores, rows).

    Rows are returned as dictionaries (all columns preserved).
    """
    vals: list[float] = []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "docking_score" not in reader.fieldnames:
            raise ValueError(f"Expected column 'docking_score' in {path}")
        for row in reader:
            raw = row.get("docking_score")
            v = _safe_float(raw)
            if v is None:
                continue
            vals.append(v)
            rows.append(row)
    return vals, rows


def evaluate_library_file(
    task: tuple[Path, str, float, float, float | None, str, int, list[float]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one docking CSV and return its summary row plus top-score candidates.

    This function is intentionally top-level so it can be used by
    ``ProcessPoolExecutor`` workers.  The final global sorting/deduplication is
    still done in the parent process to preserve the script's output format.
    """
    p, reward_mode, threshold, threshold_alpha, max_weight, weight_source, topk_candidate_k, pki_list = task

    vals, mol_rows = iter_docking_csv_rows(p)
    arr = np.asarray(vals, dtype=float)
    weights = None
    if max_weight is not None:
        weights = np.asarray(
            [_row_molecular_weight(mr, weight_source=str(weight_source)) for mr in mol_rows],
            dtype=float,
        )

    # Compute a reward for each requested pKi threshold.
    reward_cols: dict[str, float] = {}
    for pki in pki_list:
        r = float(
            reward(
                arr,
                mode=str(reward_mode),
                threshold=float(threshold),
                alpha=float(threshold_alpha),
                weights=weights,
                max_weight=max_weight,
                pki=float(pki),
            )
        )
        # Column name: replace '.' with '_' so "6.0" → "reward_pki6_0"
        key = f"reward_pki{str(pki).replace('.', '_')}"
        reward_cols[key] = r

    # Gather unique building block IDs present in this library CSV.
    bb1_ids: set[str] = set()
    bb2_ids: set[str] = set()
    bb3_ids: set[str] = set()

    def _accumulate_bb_ids(row: dict[str, Any]) -> None:
        v1 = row.get("bb1_id")
        v2 = row.get("bb2_id")
        v3 = row.get("bb3_id")
        if v1 not in (None, ""):
            bb1_ids.add(str(v1))
        if v2 not in (None, ""):
            bb2_ids.add(str(v2))
        if v3 not in (None, ""):
            bb3_ids.add(str(v3))

    topk_heap: list[tuple[float, int, dict[str, Any]]] = []
    local_counter = 0
    for mr in mol_rows:
        _accumulate_bb_ids(mr)
        score = _safe_float(mr.get("docking_score"))
        if score is None:
            continue
        out_row: dict[str, Any] = {"source_path": str(p), "source_file": p.name}
        out_row.update(mr)
        update_topk_smallest_rows(
            heap=topk_heap,
            row=out_row,
            score=score,
            k=topk_candidate_k,
            counter=local_counter,
        )
        local_counter += 1

    topk_rows = [item[2] for item in sorted(topk_heap, key=lambda t: (-t[0], t[1]))]

    summary_row: dict[str, Any] = {
        "path": str(p),
        "file": p.name,
        "bb1_ids": "|".join(
            sorted(bb1_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        ),
        "bb2_ids": "|".join(
            sorted(bb2_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        ),
        "bb3_ids": "|".join(
            sorted(bb3_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        ),
    }
    # Merge per-pKi reward columns into the summary row.  When only one pKi is
    # requested the legacy "reward" key is included for backwards compatibility.
    summary_row.update(reward_cols)
    if len(pki_list) == 1:
        summary_row["reward"] = reward_cols[
            f"reward_pki{str(pki_list[0]).replace('.', '_')}"
        ]
    return summary_row, topk_rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute a library reward for each docking CSV in a directory and sort them."
    )
    ap.add_argument(
        "--dir",
        type=Path,
        default="data/scored_libraries/not_random",
        help="Directory containing docking CSVs (each file = one library).",
    )
    ap.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for selecting docking CSV files (default: *.csv).",
    )
    ap.add_argument(
        "--reward",
        choices=REWARD_MODES,
        default="threshold",
        help=(
            "Library reward mode. Use 'ampc_pki6_hits' for expected number of "
            "AmpC hits with pKi>=6 among molecules passing any --max-weight cutoff. "
            "Default: threshold."
        ),
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=-9.0,
        help="Threshold used for reward(mode='threshold') (default: -9.0).",
    )
    ap.add_argument(
        "--threshold-alpha",
        type=float,
        default=1.0,
        help=(
            "Smoothing parameter alpha for reward(mode='threshold'): "
            "sum(1 / (1 + exp((value - threshold) / alpha))). "
            "Default: 1.0."
        ),
    )
    ap.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help=(
            "Maximum molecular weight (Daltons) allowed to contribute to "
            "threshold reward. By default, no molecular-weight cutoff is applied."
        ),
    )
    ap.add_argument(
        "--weight-source",
        choices=["smiles", "bb_sum"],
        default="smiles",
        help=(
            "How to compute molecular weights when applying --max-weight: "
            "'smiles' uses the final product smiles column; 'bb_sum' sums "
            "bb1_smiles, bb2_smiles, and bb3_smiles. Default: smiles."
        ),
    )
    ap.add_argument(
        "--k",
        type=int,
        default=20,
        help="How many top molecules to list",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/library_rewards_sorted.csv"),
        help="Path to write output CSV (summary + top-k molecules).",
    )
    ap.add_argument(
        "--ascending",
        action="store_true",
        help="Sort rewards increasing instead of decreasing.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast if a file is missing required columns / can't be read.",
    )
    ap.add_argument(
        "--n-proc",
        type=int,
        default=1,
        help=(
            "Number of worker processes to use across library CSV files. "
            "Use ${SLURM_CPUS_PER_TASK} in Slurm jobs. Default: 1."
        ),
    )
    ap.add_argument(
        "--pki",
        type=str,
        default="6.0",
        help=(
            "Comma-separated list of pKi thresholds for ampc_pki6_hits / "
            "ampc_pki6_proportion reward modes. Each value produces a column "
            "'reward_pki{val}' (e.g. '--pki 6.0,4.5'). Default: 6.0."
        ),
    )

    args = ap.parse_args(argv)

    # Parse comma-separated pKi list.
    pki_str = str(getattr(args, "pki", "6.0") or "6.0")
    pki_list: list[float] = [float(x.strip()) for x in pki_str.split(",") if x.strip()]
    if not pki_list:
        raise ValueError("--pki must contain at least one valid float value")

    in_dir: Path = args.dir
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"[error] --dir is not a directory: {in_dir}", file=sys.stderr)
        return 2

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"[warn] no files found in {in_dir} matching pattern {args.pattern!r}")
        return 0

    rows: list[dict[str, Any]] = []
    # We will deduplicate the final "top molecules" list by (bb1_id, bb2_id, bb3_id).
    # To still reliably output topk_k unique molecules, we keep a larger candidate pool
    # (topk_candidate_k) and dedupe down to topk_k.
    topk_heap: list[tuple[float, int, dict[str, Any]]] = []
    topk_k = args.k
    topk_candidate_k = max(topk_k * 10, topk_k)
    counter = 0

    worker_tasks = [
        (
            p,
            str(args.reward),
            float(args.threshold),
            float(args.threshold_alpha),
            args.max_weight,
            str(args.weight_source),
            topk_candidate_k,
            pki_list,
        )
        for p in files
    ]

    n_proc = max(1, int(args.n_proc))

    def _consume_result(summary_row: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> None:
        nonlocal counter
        rows.append(summary_row)
        for candidate in candidate_rows:
            score = _safe_float(candidate.get("docking_score"))
            if score is None:
                continue
            update_topk_smallest_rows(
                heap=topk_heap,
                row=candidate,
                score=score,
                k=topk_candidate_k,
                counter=counter,
            )
            counter += 1

    if n_proc == 1:
        iterator = zip(files, worker_tasks)
        for p, task in iterator:
            try:
                summary_row, candidate_rows = evaluate_library_file(task)
                _consume_result(summary_row, candidate_rows)
            except Exception as e:
                if args.strict:
                    raise
                print(f"[warn] skipping {p}: {e}", file=sys.stderr)
    else:
        print(f"[info] evaluating {len(files)} libraries with {n_proc} worker processes")
        try:
            with ProcessPoolExecutor(max_workers=n_proc) as ex:
                future_to_path = {ex.submit(evaluate_library_file, task): task[0] for task in worker_tasks}
                for fut in as_completed(future_to_path):
                    p = future_to_path[fut]
                    try:
                        summary_row, candidate_rows = fut.result()
                        _consume_result(summary_row, candidate_rows)
                    except Exception as e:
                        if args.strict:
                            raise
                        print(f"[warn] skipping {p}: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            raise

    if not rows:
        print("[warn] no valid docking CSVs were evaluated.")
        return 0

    # Determine the reward columns that actually exist in the summary rows.
    reward_fieldnames = [f"reward_pki{str(p).replace('.', '_')}" for p in pki_list]
    if len(pki_list) == 1:
        reward_fieldnames.insert(0, "reward")

    sort_col = "reward" if len(pki_list) == 1 else reward_fieldnames[1]

    # Print summary statistics for each reward column.
    print("\n[reward statistics]")
    print(f"{'column':<20} {'n':>6} {'mean':>14} {'std':>14}")
    print("-" * 58)
    for col in reward_fieldnames:
        vals_for_col = np.asarray(
            [d.get(col) for d in rows if d.get(col) is not None],
            dtype=float,
        )
        n_valid = vals_for_col.size
        mean_val = float(np.mean(vals_for_col)) if n_valid else float("nan")
        std_val = float(np.std(vals_for_col, ddof=0)) if n_valid else float("nan")
        print(f"{col:<20} {n_valid:>6} {mean_val:>14.6f} {std_val:>14.6f}")
    print()

    # Sort: NaNs always last.
    def sort_key(d: dict):
        r = d.get(sort_col)
        if r is None or (isinstance(r, float) and math.isnan(r)):
            return (1, 0.0)
        return (0, float(r))

    rows_sorted = sorted(rows, key=sort_key, reverse=not bool(args.ascending))

    # Materialize candidate rows sorted by increasing (smallest) docking_score.
    # Heap stores (-score, counter, row)
    topk_rows = [item[2] for item in sorted(topk_heap, key=lambda t: (-t[0], t[1]))]
    topk_rows = sorted(
        topk_rows,
        key=lambda d: (_safe_float(d.get("docking_score")) is None, _safe_float(d.get("docking_score")) or 0.0),
    )

    # Deduplicate by (bb1_id, bb2_id, bb3_id). Keep any row (first encountered).
    def _mol_key(d: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(d.get("bb1_id", "")),
            str(d.get("bb2_id", "")),
            str(d.get("bb3_id", "")),
        )

    seen: set[tuple[str, str, str]] = set()
    topk_unique_rows: list[dict[str, Any]] = []
    for r in topk_rows:
        k = _mol_key(r)
        if k in seen:
            continue
        seen.add(k)
        topk_unique_rows.append(r)
        if len(topk_unique_rows) >= topk_k:
            break

    # Write output file (single CSV with 2 sections).
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        # Section 1: per-library summary
        summary_fieldnames = ["path", "file"]
        if len(pki_list) == 1:
            summary_fieldnames.append("reward")
        summary_fieldnames.extend(reward_fieldnames[1:] if len(pki_list) == 1 else reward_fieldnames)
        summary_fieldnames.extend(["bb1_ids", "bb2_ids", "bb3_ids"])

        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        for d in rows_sorted:
            w.writerow(d)

        # Blank line + marker
        f.write("\n")
        f.write(f"# top_{topk_k}_molecules_by_docking_score (smallest = best)\n")

        if topk_unique_rows:
            # Union of keys across topk rows
            key_set: set[str] = set()
            for r in topk_unique_rows:
                key_set.update(r.keys())

            # Column order: ids together, then smiles, then vina.
            preferred = [
                "source_path",
                "source_file",
                "bb1_id",
                "bb2_id",
                "bb3_id",
                "bb1_smiles",
                "bb2_smiles",
                "bb3_smiles",
                "smiles",
                "docking_score",
            ]

            ordered: list[str] = [k for k in preferred if k in key_set]
            for k in sorted(key_set):
                if k not in ordered:
                    ordered.append(k)

            w2 = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            w2.writeheader()
            for r in topk_unique_rows:
                w2.writerow(r)

    print(
        f"[done] wrote summary ({len(rows_sorted)} libraries) + top-{len(topk_unique_rows)} unique molecules to {out_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
