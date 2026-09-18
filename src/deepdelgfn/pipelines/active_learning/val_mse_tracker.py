#!/usr/bin/env python3
"""Track the DeepDEL model's external validation MSE across an active-learning
campaign.

The external validation set is a *separate* held-out CSV (distinct from the
internal train/val split used for early stopping during DeepDEL training). It is
evaluated at each campaign "step":

  - step 0: immediately after dd-init (the initial DeepDEL checkpoint);
  - step k: immediately after the k-th DeepDEL update.

Each step's ``external_val_mse`` (read from the per-step ``--stats-json`` file)
is appended to ``RUN_ROOT/deepdel_extval_mse_by_step.csv``, and the evolution
plot ``RUN_ROOT/deepdel_extval_mse_evolution.png`` is regenerated so the final
campaign step produces the final plot.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import RUN_ID, RUN_ROOT, _csv_data_row_count, _plot_title_metadata


_DEEPDEL_EXT_VAL_COLUMNS = [
    "step",
    "phase",
    "outer_loop",
    "inner_loop",
    "timestamp",
    "run_id",
    "external_val_mse",
    "last_val_mse",
    "best_val_mse",
    "epochs_completed",
]


def _read_stats(stats_json_path: Path) -> Optional[dict]:
    if stats_json_path is None:
        return None
    stats_json_path = Path(stats_json_path)
    if not stats_json_path.exists():
        print(
            f"[deepdel-extval] WARNING: Stats JSON not found at {stats_json_path}; "
            "skipping recording."
        )
        return None
    try:
        with stats_json_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(
            f"[deepdel-extval] WARNING: Could not read stats JSON {stats_json_path}: {e}"
        )
        return None


def _update_extval_csv(stats: dict) -> Path:
    """Upsert one external-validation step into the run-level CSV."""
    out_csv = RUN_ROOT / "deepdel_extval_mse_by_step.csv"
    row_df = pd.DataFrame([stats], columns=_DEEPDEL_EXT_VAL_COLUMNS)
    if out_csv.exists() and _csv_data_row_count(out_csv) > 0:
        try:
            old_df = pd.read_csv(out_csv)
        except Exception as e:
            print(
                f"[deepdel-extval] WARNING: Could not read existing CSV {out_csv}; "
                f"rewriting it: {e}"
            )
            old_df = pd.DataFrame(columns=_DEEPDEL_EXT_VAL_COLUMNS)
        for col in _DEEPDEL_EXT_VAL_COLUMNS:
            if col not in old_df.columns:
                old_df[col] = np.nan
        old_df = old_df[_DEEPDEL_EXT_VAL_COLUMNS]
        same_step = pd.to_numeric(old_df["step"], errors="coerce").eq(int(stats["step"]))
        out_df = pd.concat([old_df.loc[~same_step], row_df], ignore_index=True)
    else:
        out_df = row_df
    out_df["step"] = pd.to_numeric(out_df["step"], errors="coerce")
    out_df = out_df.sort_values(by="step").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out_df.to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)
    print(f"[deepdel-extval] Updated external-validation CSV: {out_csv}")
    return out_csv


def write_deepdel_val_mse_plot(stats_csv: Path) -> None:
    """Plot step vs external validation MSE over the active-learning campaign."""
    try:
        if not stats_csv.exists() or _csv_data_row_count(stats_csv) <= 0:
            return
        df = pd.read_csv(stats_csv)
        if len(df) == 0:
            return
        if "external_val_mse" not in df.columns:
            print(
                "[deepdel-extval] WARNING: Cannot plot external-validation MSE; "
                "missing column: external_val_mse"
            )
            return
        df["step"] = pd.to_numeric(df["step"], errors="coerce")
        df["external_val_mse"] = pd.to_numeric(df["external_val_mse"], errors="coerce")
        valid = df.dropna(subset=["step", "external_val_mse"]).sort_values(by="step")
        if len(valid) == 0:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(
            valid["step"],
            valid["external_val_mse"],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label="external val MSE",
        )
        ax.set_xlabel("campaign step (0 = dd-init)")
        ax.set_ylabel("external validation MSE")
        ax.set_title(
            _plot_title_metadata(
                prefix="DeepDEL external validation MSE over active learning | "
            )
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_png = RUN_ROOT / "deepdel_extval_mse_evolution.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[plot] Saved external-validation MSE plot to {out_png}")
    except Exception as e:
        print(f"[WARN] Failed to generate external-validation plot: {e}")


def record_deepdel_val_mse(
    stats_json_path: Path,
    *,
    step: int,
    phase: str,
    outer_loop: int,
    inner_loop: Optional[int] = None,
) -> None:
    """Record one campaign step's external validation MSE into the CSV and plot."""
    ts = _read_stats(stats_json_path)
    if ts is None:
        return
    external_val_mse = ts.get("external_val_mse")
    if external_val_mse is None or not np.isfinite(float(external_val_mse)):
        print(
            f"[deepdel-extval] WARNING: external_val_mse missing/invalid in "
            f"{stats_json_path}; skipping step {step} recording."
        )
        return
    stats = {
        "step": int(step),
        "phase": str(phase),
        "outer_loop": int(outer_loop),
        "inner_loop": ("" if inner_loop is None else int(inner_loop)),
        "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "run_id": RUN_ID,
        "external_val_mse": float(external_val_mse),
        "last_val_mse": float(ts.get("last_val_mse", float("nan"))),
        "best_val_mse": float(ts.get("best_val_mse", float("nan"))),
        "epochs_completed": int(ts.get("epochs_completed", 0)),
    }
    print(
        f"[deepdel-extval] step={stats['step']} phase={stats['phase']} "
        f"external_val_mse={stats['external_val_mse']:.8g}"
    )
    stats_csv = _update_extval_csv(stats)
    write_deepdel_val_mse_plot(stats_csv)
