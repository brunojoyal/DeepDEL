#!/usr/bin/env python3
"""Main dispatch entry point for the active learning pipeline.

This module was extracted from active_learning_stage.py.  It reads the
``STAGE``, ``OUTER_LOOP``, and ``INNER_LOOP`` environment variables and
delegates to the appropriate stage function from this subpackage.
"""

import os
import time

from .config import CFG, BBS_CSV, RUN_ID, active_learning_target
from .prepare import (
    prepare_outer_loop,
    prepare_proxy_outer_loop,
    prepare_dataset_outer_loop,
    prepare_dataset_batch_outer_loop,
    prepare_dataset_finalize_outer_loop,
    prepare_deepdel_outer_loop,
)
from .inner import (
    inner_train_outer_loop,
    eval_topm_batch_outer_loop,
    eval_topm_finalize_outer_loop,
    deepdel_update_outer_loop,
)
from .docking_stage import (
    dock_outer_loop,
    dock_prepare_outer_loop,
    dock_launch_outer_loop,
    dock_batch_outer_loop,
    dock_finalize_outer_loop,
)


def main():
    stage = os.environ.get("STAGE", "prepare")
    outer_loop = int(os.environ.get("OUTER_LOOP", "0"))
    inner_loop_raw = os.environ.get("INNER_LOOP")
    inner_loop = int(inner_loop_raw) if inner_loop_raw not in (None, "") else None
    print(
        f"[stage] STAGE={stage} OUTER_LOOP={outer_loop} INNER_LOOP={inner_loop} "
        f"RUN_ID={RUN_ID}"
    )
    print(f"[stage] TARGET={active_learning_target()}")
    print(f"[stage] BBs (combined): {BBS_CSV}")
    stages_requiring_inner = {
        "inner_train",
        "eval_topm_batch",
        "eval_topm_finalize",
        "deepdel_update",
    }
    if stage in stages_requiring_inner and inner_loop is None:
        raise RuntimeError(f"INNER_LOOP is required when STAGE={stage}")
    if stage == "prepare":
        prepare_outer_loop(outer_loop)
    elif stage == "prepare_proxy":
        prepare_proxy_outer_loop(outer_loop)
    elif stage == "prepare_dataset":
        prepare_dataset_outer_loop(outer_loop)
    elif stage == "prepare_dataset_batch":
        prepare_dataset_batch_outer_loop(outer_loop)
    elif stage == "prepare_dataset_finalize":
        prepare_dataset_finalize_outer_loop(outer_loop)
    elif stage == "prepare_deepdel":
        prepare_deepdel_outer_loop(outer_loop)
    elif stage == "inner_train":
        inner_train_outer_loop(outer_loop, int(inner_loop))
    elif stage == "eval_topm_batch":
        eval_topm_batch_outer_loop(outer_loop, int(inner_loop))
    elif stage == "eval_topm_finalize":
        eval_topm_finalize_outer_loop(outer_loop, int(inner_loop))
    elif stage == "deepdel_update":
        deepdel_update_outer_loop(outer_loop, int(inner_loop))
    elif stage == "dock":
        dock_outer_loop(outer_loop, inner_loop)
    elif stage == "dock_prepare":
        dock_prepare_outer_loop(outer_loop, inner_loop)
    elif stage == "dock_launch":
        dock_launch_outer_loop(outer_loop, inner_loop)
    elif stage == "dock_batch":
        dock_batch_outer_loop(outer_loop, inner_loop)
    elif stage == "dock_finalize":
        dock_finalize_outer_loop(outer_loop, inner_loop)
    else:
        raise ValueError(f"Unknown STAGE={stage}")
    time.sleep(float(CFG["sleep_between_cycles"]))


if __name__ == "__main__":
    main()