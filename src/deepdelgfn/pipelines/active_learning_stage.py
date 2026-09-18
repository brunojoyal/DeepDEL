#!/usr/bin/env python3
"""
Unified worker for the split CPU/GPU active learning pipeline.

This script is invoked by `jobs/active_learning_job.sh` (which submits one
sbatch job per stage). The submitter (`scripts/active_learning_submit.py`)
writes a single config JSON under RUN_ROOT and selects which Slurm partition
each stage runs on; this worker reads STAGE/OUTER_LOOP/DEL_GFN_CONFIG from
the environment and dispatches accordingly.

Stages: ``prepare`` → ``inner`` × M → ``dock`` (legacy) or
``dock_prepare`` → ``dock_launch`` → ``dock_batch`` array → ``dock_finalize``.
The prepare stage builds the initial DeepDEL checkpoint; each inner stage runs
one GFlowNet training/evaluation cycle and finetunes that checkpoint in-place.

.. note::

    This module is a thin shim that imports the real implementation from the
    ``active_learning`` subpackage.  The subpackage was created by splitting
    the original monolithic ``active_learning_stage.py`` into focused modules:

    - ``config.py`` -- global constants, CFG setup, path/resource helpers
    - ``prepare.py`` -- proxy / dataset / DeepDEL prepare stages
    - ``inner.py`` -- GFN inner-loop training, eval, and DeepDEL finetuning
    - ``docking_candidates.py`` -- top-N and leader-based docking selection
    - ``docking_stage.py`` -- dock_prepare / dock_batch / dock_finalize / dock_launch
    - ``main.py`` -- stage dispatch

    The public API is unchanged: ``from deepdelgfn.pipelines.active_learning
    import main`` and ``python -m deepdelgfn.pipelines.active_learning_stage``
    continue to work.
"""

from .active_learning.main import main

if __name__ == "__main__":
    main()