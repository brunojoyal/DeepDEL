#!/usr/bin/env python3
"""Aggregate Experiment 1 two-fidelity pipeline CSVs.

Fidelity hierarchy:
  F2 (DeepDEL)     — GFN terminal reward; ``topm_rewards.csv`` column ``reward``.
  F1 (docking proxy) — library-level reward from frozen autodock proxy model;
                        ``docking_proxy_topm.csv`` column ``autodock_proxy_value``.

Primary metrics span both fidelities: DeepDEL (F2) top-1 / top-100 mean,
docking proxy (F1) top-1 / top-100 mean on the DeepDEL shortlist.

Manifest column ``docking_csv`` and per-run file ``docking_proxy_topm.csv`` keep
legacy names for artifact compatibility; both hold F1 proxy-eval output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


KEY = ["B1_id", "B2_id", "B3_id"]
def finite(a):
    return pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_bb_fingerprints(path, radius=2, n_bits=2048):
    bbs = pd.read_csv(path)
    smiles_col = "SMILES" if "SMILES" in bbs.columns else "smiles"
    if "ID" not in bbs.columns or smiles_col not in bbs.columns:
        raise ValueError(f"BBS must contain ID and SMILES columns: {path}")
    fps = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(n_bits)
    )
    for bb_id, smiles in bbs[["ID", smiles_col]].itertuples(index=False):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            fps[str(int(bb_id))] = generator.GetFingerprint(mol)
    return fps


def top_k_diversity(topm, bb_fps):
    reps = []
    for rec in topm.to_dict("records"):
        cycles = []
        for col in KEY:
            fps = [bb_fps[x] for x in str(rec[col]).split("|") if x in bb_fps]
            if not fps:
                raise ValueError(f"no valid fingerprints for {col}={rec[col]}")
            cycles.append(np.asarray([list(fp) for fp in fps], dtype=float).mean(axis=0))
        reps.append(np.concatenate(cycles))
    if len(reps) < 2:
        return 0.0
    distances = []
    for i in range(len(reps)):
        for j in range(i):
            a, b = reps[i], reps[j]
            ab = float(np.dot(a, b))
            denom = float(np.dot(a, a) + np.dot(b, b) - ab)
            distances.append(0.0 if denom <= 0 else 1.0 - max(0.0, min(1.0, ab / denom)))
    return float(np.mean(distances))


def metric_names(top_k):
    """Return ordered metric names — spans F2, F1, and diversity."""
    suffix = f"top{top_k}"
    return [
        "deepdel_top1", f"deepdel_{suffix}_mean",                                      # F2 (DeepDEL)
        f"docking_proxy_top1_on_deepdel_{suffix}",                                     # F1 (proxy)
        f"docking_proxy_{suffix}_mean_on_deepdel_{suffix}",                            # F1 (proxy)
        f"{suffix}_diversity",
    ]


def experiment1_model_dimensions(config=None):
    """Return the Experiment 1 dimensions and parameter counts.

    When ``config`` (loaded from ``experiment_config.json``) supplies the
    resolved values persisted by the launcher, they are returned directly so
    the summary reflects the actual run.  For legacy runs that never persisted
    a config, fall back to the historical ``BB_POOL_SIZE=3000`` defaults below.
    """
    config = config or {}
    resolved = all(
        key in config
        for key in (
            "gfn_dim", "mh_hidden_dim", "mh_n_layers",
            "hier_hidden_dim", "hier_n_layers", "hier_max_clusters",
            "deepsets_params", "multihot_params", "hierarchical_params",
        )
    )
    if resolved:
        return [
            ("deepsets", "JOINT_DIM=STATE_DIM=ACTION_DIM",
             int(config["gfn_dim"]), int(config["deepsets_params"])),
            ("hierarchical",
             f"hidden_dim={config['hier_hidden_dim']}, n_layers={config['hier_n_layers']}, max_clusters={config['hier_max_clusters']}",
             int(config["hier_hidden_dim"]), int(config["hierarchical_params"])),
            ("multihot",
             f"hidden_dim={config['mh_hidden_dim']}, n_layers={config['mh_n_layers']}",
             int(config["mh_hidden_dim"]), int(config["multihot_params"])),
        ]
    return [
        ("deepsets", "JOINT_DIM=STATE_DIM=ACTION_DIM", 2182, 37143819),
        ("hierarchical", "hidden_dim=1536, n_layers=5, max_clusters=20", 1536, 37175615),
        ("multihot", "hidden_dim=1536, n_layers=5", 1536, 37101864),
    ]


def read_wall_clock_seconds(topm_path):
    path = Path(topm_path).parent / "wall_clock_seconds.txt"
    try:
        return float(path.read_text().strip())
    except (OSError, ValueError):
        return np.nan


def _markdown_value(value):
    """Format a resolved configuration value safely for a Markdown table."""
    if value is None or value == "":
        return "unknown"
    if isinstance(value, bool):
        value = str(value).lower()
    return str(value).replace("|", "\\|").replace("\n", " ")


def load_experiment_config(outdir, manifest, args):
    """Load the launcher's resolved config, with legacy-compatible defaults.

    Older runs did not persist a config.  The fallback keeps summaries useful,
    while new runs get exact values from experiment_config.json.
    """
    path = Path(outdir) / "experiment_config.json"
    config = {}
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        pass
    experiment_ids = sorted(set(manifest["experiment_id"].dropna().astype(str))) if "experiment_id" in manifest else []
    config.setdefault("experiment_id", ", ".join(experiment_ids) or "unknown")
    config.setdefault("variants", ", ".join(sorted(set(manifest["variant"].dropna().astype(str)))) if "variant" in manifest else "unknown")
    config.setdefault("seeds", ", ".join(sorted(set(manifest["seed"].dropna().astype(str)), key=int)) if "seed" in manifest else "unknown")
    config.setdefault("aggregation", f"First {args.top_k} ranked sampled rows per seed")
    config.setdefault("bbs", args.bbs)
    config.setdefault("proxy_eval_lower_is_better", bool(args.lower_is_better))
    return config


def write_markdown_summary(summary, path, top_k, manifest, args):
    metrics = metric_names(top_k)
    config = load_experiment_config(path.parent, manifest, args)
    experiment_ids = sorted(set(manifest["experiment_id"].dropna().astype(str))) if "experiment_id" in manifest else []
    variants = sorted(set(manifest["variant"].dropna().astype(str))) if "variant" in manifest else []
    seeds = sorted(set(manifest["seed"].dropna().astype(str)), key=int) if "seed" in manifest else []
    lines = [
        f"# {args.title}",
        "",
        "## Experiment parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    parameter_labels = [
        ("experiment_id", "Experiment ID"),
        ("variants", "Variants"),
        ("seeds", "Seeds"),
        ("aggregation", "Aggregation"),
        ("bbs", "Building blocks"),
        ("deepdel", "DeepDEL checkpoint"),
        ("deepdel_conditioned", "Conditioned DeepDEL checkpoint"),
        ("deepdel_fixed", "Fixed-threshold DeepDEL checkpoint"),
        ("autodock_model", "Docking-proxy model"),
        ("clusters", "Hierarchical clusters"),
        ("size1", "Library shape SIZE1"),
        ("size2", "Library shape SIZE2"),
        ("size3", "Library shape SIZE3"),
        ("bb_pool_size", "Building-block pool size"),
        ("reaction_mode", "Reaction mode"),
        ("gfn_reward_source", "GFlowNet reward source"),
        ("proxy_eval_reward_mode", "Proxy reward mode"),
        ("proxy_eval_threshold", "Proxy threshold"),
        ("threshold", "Fixed threshold"),
        ("threshold_min", "Conditioned threshold min"),
        ("threshold_max", "Conditioned threshold max"),
        ("fourier_n_freqs", "Fourier n-freqs"),
        ("pretrain_steps", "Pretraining steps"),
        ("total_steps", "Total steps"),
        ("specialization_mode", "Specialization mode"),
        ("proxy_eval_weight_source", "Proxy weight source"),
        ("proxy_eval_max_weight", "Proxy maximum weight"),
        ("proxy_eval_alpha", "Proxy alpha"),
        ("proxy_eval_n_top", "Proxy top-N"),
        ("steps", "Training steps"),
        ("batch_trajectories", "Batch trajectories"),
        ("lr", "Learning rate"),
        ("logz_lr", "LogZ learning rate"),
        ("beta", "Beta"),
    ]
    for key, label in parameter_labels:
        value = config.get(key)
        if key in {"size1", "size2", "size3"} and value is not None:
            value = f"{value}"
        if key == "size1":
            value = f"SIZE1={value}, SIZE2={config.get('size2', 'unknown')}, SIZE3={config.get('size3', 'unknown')}"
            label = "Library shape"
        elif key in {"size2", "size3"}:
            continue
        lines.append(f"| {label} | `{_markdown_value(value)}` |")
    lines.extend([
        "",
        "## Aggregation parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Experiment ID | `{', '.join(experiment_ids) or 'unknown'}` |",
        f"| Variants | `{', '.join(variants) or 'unknown'}` |",
        f"| Seeds | `{', '.join(seeds) or 'unknown'}` |",
        f"| Building-block file | `{args.bbs}` |",
        f"| Ranked rows per seed | `{top_k}` |",
        f"| Lower-is-better oracle direction | `{bool(args.lower_is_better)}` |",
        f"| Minimum valid seeds required | `{args.min_seeds}` |",
    ])
    exp1_arch_variants = [v for v in variants if v in {"deepsets", "hierarchical", "multihot"}]
    if exp1_arch_variants:
        lines.extend([
            "",
            "## Model dimensions",
            "",
            "| Variant | Dimensions | Width / dimension | Estimated trainable parameters |",
            "|---|---|---:|---:|",
        ])
        for variant, dimensions, width, params in experiment1_model_dimensions(config):
            lines.append(f"| {variant} | `{dimensions}` | {width} | {params:,} |")
        lines.append("")
        if "deepsets_params" in config:
            lines.append(
                f"These are the resolved Experiment 1 dimensions for `BB_POOL_SIZE={config.get('bb_pool_size', 'unknown')}`; the DeepSets dimension is solved to match the multihot/hierarchical parameter budget."
            )
        else:
            lines.append(
                "These are the historical Experiment 1 defaults (`BB_POOL_SIZE=3000`); the DeepSets dimension is solved to match the multihot/hierarchical parameter budget."
            )
    lines.extend([
        "",
        "## Mean wall-clock training time",
        "",
        "| Variant | Mean wall-clock time (seconds) | Mean wall-clock time (minutes) | Runs with timing |",
        "|---|---:|---:|---:|",
    ])
    for variant in variants:
        times = [read_wall_clock_seconds(rec["topm_csv"]) for rec in manifest.to_dict("records") if rec.get("variant") == variant]
        times = [value for value in times if np.isfinite(value)]
        mean_time = float(np.mean(times)) if times else np.nan
        lines.append(f"| {variant} | {mean_time:.2f} | {mean_time / 60.0:.2f} | {len(times)} |" if np.isfinite(mean_time) else f"| {variant} | NA | NA | 0 |")
    lines.extend([
        "",
        f"Summary over the first **{top_k}** ranked sampled rows per seed. Values are `mean ± std`.",
        "",
        "| Variant | " + " | ".join(f"`{metric}`" for metric in metrics) + " |",
        "|---|" + "---:|" * len(metrics),
    ])
    values = {
        (rec["variant"], rec["metric"]): rec
        for rec in summary.to_dict("records")
    }
    for variant in variants:
        cells = []
        for metric in metrics:
            rec = values.get((variant, metric))
            if rec is None or pd.isna(rec["mean"]):
                cells.append("NA")
            else:
                std = rec["std"]
                cells.append(
                    f"{rec['mean']:.6g} ± {std:.6g}"
                    if pd.notna(std)
                    else f"{rec['mean']:.6g} ± NA"
                )
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "`top<N>_diversity` is the mean pairwise generalized-Tanimoto distance of the top-N library representations, using the concatenated cycle-level mean 2048-bit Morgan (radius 2) fingerprints. Duplicate sampled libraries are retained and contribute zero distance to each other.",
    ])
    path.write_text("\n".join(lines) + "\n")


def metric_row(manifest_row, topm, proxy, lower, top_k, bb_fps):
    """Compute seed-level metrics for one (variant, seed).

    Parameters
    ----------
    lower:
        Whether lower proxy values are better. This controls the best proxy
        value reported over the DeepDEL top-k shortlist.
    """
    missing = (set(KEY) - set(topm)) | (set(KEY) - set(proxy))
    if missing:
        raise ValueError(f"missing candidate columns: {sorted(missing)}")
    if "autodock_proxy_value" not in proxy:
        raise ValueError("proxy CSV is missing autodock_proxy_value")
    if "reward" not in topm and "deepdel_reward" not in topm:
        raise ValueError("topm CSV is missing reward/deepdel_reward")

    topm = topm.copy()
    proxy = proxy.copy()
    topm[KEY] = topm[KEY].astype(str)
    proxy[KEY] = proxy[KEY].astype(str)
    if "rank" not in topm:
        raise ValueError("topm CSV must contain rank")

    # A sample is a ranked terminal record, not a unique library identity.
    # Multihot can sample the same library repeatedly; those repetitions are
    # intentionally retained in the top-100 sample-level estimand.
    topm["rank"] = finite(topm["rank"])
    topm = topm.sort_values(["rank"] + KEY, ascending=True).head(top_k).reset_index(drop=True)
    if len(topm) < top_k:
        raise ValueError(f"only {len(topm)} sampled candidates; need {top_k}")
    if topm["rank"].isna().any():
        raise ValueError("topm rank contains non-finite values")

    # Proxy evaluation (F1) covers the entire DeepDEL (F2) top-k shortlist.
    joined = topm.merge(
        proxy[[*KEY, "autodock_proxy_value"]],
        on=KEY,
        how="left",
        validate="many_to_one",
    )
    deep = finite(joined["reward"] if "reward" in joined.columns else joined["deepdel_reward"])  # F2
    if deep.isna().any():
        raise ValueError(f"DeepDEL reward contains non-finite values in top {top_k}")
    dock = finite(joined["autodock_proxy_value"])  # F1
    valid = dock.dropna()

    # F1 metrics: evaluate the complete DeepDEL (F2) top-k shortlist, then
    # report the best F1 proxy reward (not the value at F2 rank 1).
    dock_top1 = float(valid.min() if lower else valid.max()) if len(valid) else np.nan
    dock_topk_mean = float(valid.mean()) if len(valid) else np.nan

    # Optional diagnostics: rerank the same top-k samples by F1 library reward.
    # `lower` controls whether lower docking proxy values are better (raw proxy
    # scores) or higher values are better (threshold rewards).
    oracle = valid.sort_values(ascending=bool(lower)).head(top_k)
    oracle_top1 = float(oracle.iloc[0]) if len(oracle) else np.nan
    oracle_top100_mean = float(oracle.mean()) if len(oracle) else np.nan

    row = dict(manifest_row)
    row.update(
        {
            "n_candidates": int(len(joined)),
            "n_valid_docking": int(len(valid)),
            "deepdel_top1": float(deep.iloc[0]),                                                # F2
            f"deepdel_top{top_k}_mean": float(deep.mean()),                                    # F2
            f"docking_proxy_top1_on_deepdel_top{top_k}": dock_top1,                            # F1
            f"docking_proxy_top{top_k}_mean_on_deepdel_top{top_k}": dock_topk_mean,            # F1
            f"top{top_k}_diversity": top_k_diversity(topm, bb_fps),
            "status": "ok",
        }
    )

    joined = joined.copy()
    joined.insert(0, "seed", manifest_row["seed"])
    joined.insert(0, "variant", manifest_row["variant"])
    joined["deepdel_rank"] = (
        finite(joined["rank"]).astype("Int64")
        if "rank" in joined.columns
        else pd.Series(np.arange(1, len(joined) + 1), index=joined.index)
    )
    joined["deepdel_reward"] = deep
    joined["docking_proxy_value"] = dock
    joined["docking_valid"] = dock.notna()
    if "experiment_id" in manifest_row:
        joined.insert(0, "experiment_id", manifest_row["experiment_id"])
    return row, joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--lower-is-better",
        type=int,
        default=0,
        help=(
            "Direction for oracle docking metrics only. Use 0 (default) when "
            "autodock_proxy_value is a library-level reward to maximize "
            "(e.g. threshold mode); use 1 when it is a raw docking score to minimize."
        ),
    )
    ap.add_argument("--min-seeds", type=int, default=1)
    ap.add_argument("--bbs", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--title", type=str, default="Experiment Summary",
                    help="Markdown title (e.g. 'Experiment 1 Summary' or 'Experiment 2 Summary').")
    args = ap.parse_args()
    if args.top_k < 2:
        raise ValueError("--top-k must be at least 2")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    bb_fps = load_bb_fingerprints(args.bbs)

    # Prefer successful/pending rows from the original launcher manifest; ignore
    # prior aggregate failure annotations when re-running.
    if {"variant", "seed"}.issubset(manifest.columns):
        manifest = manifest.drop_duplicates(subset=["variant", "seed"], keep="first")

    rows, candidates, failures = [], [], []
    for rec in manifest.to_dict("records"):
        try:
            topm = pd.read_csv(rec["topm_csv"])
            proxy_path = rec.get("proxy_csv", rec.get("docking_csv"))
            proxy = pd.read_csv(proxy_path)
            row, cand = metric_row(rec, topm, proxy, bool(args.lower_is_better), args.top_k, bb_fps)
            rows.append(row)
            candidates.append(cand)
        except Exception as exc:
            failed = dict(rec)
            failed["status"] = "failed"
            failed["failure_reason"] = str(exc)
            failures.append(failed)

    per_seed = pd.DataFrame(rows)
    if not per_seed.empty:
        per_seed.to_csv(out / "per_seed_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "per_seed_metrics.csv", index=False)

    if candidates:
        pd.concat(candidates, ignore_index=True).to_csv(out / "candidate_level.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "candidate_level.csv", index=False)

    # Manifest with final statuses.
    status_rows = []
    ok_keys = {(r.get("variant"), r.get("seed")) for r in rows}
    for rec in manifest.to_dict("records"):
        key = (rec.get("variant"), rec.get("seed"))
        if key in ok_keys:
            rec = dict(rec)
            rec["status"] = "ok"
            rec.pop("failure_reason", None)
            status_rows.append(rec)
    status_rows.extend(failures)
    pd.DataFrame(status_rows).drop_duplicates(subset=["variant", "seed"], keep="last").to_csv(
        out / "run_manifest.csv", index=False
    )

    summary = []
    if not per_seed.empty:
        for variant, group in per_seed.groupby("variant"):
            for metric in metric_names(args.top_k):
                vals = finite(group[metric]).dropna()
                n = len(vals)
                mean = float(vals.mean()) if n else np.nan
                std = float(vals.std(ddof=1)) if n >= 2 else np.nan
                stderr = float(std / np.sqrt(n)) if n >= 2 else np.nan
                summary.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "n_seeds": n,
                        "mean": mean,
                        "std": std,
                        "stderr": stderr,
                        "ci95_low": mean - 1.96 * stderr if n >= 2 else np.nan,
                        "ci95_high": mean + 1.96 * stderr if n >= 2 else np.nan,
                    }
                )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out / "summary_mean_std.csv", index=False)
    write_markdown_summary(summary_df, out / "summary_mean_std.md", args.top_k, manifest, args)

    if len(per_seed) < args.min_seeds:
        raise SystemExit(f"Only {len(per_seed)} valid runs; required {args.min_seeds}")


if __name__ == "__main__":
    main()
