#!/usr/bin/env python3
"""
Diagnostic script: dock aspirin against AmpC two ways and compare.

Test A: bypass ligbuild — feed the pre-built bundle_aspirin_000.tgz directly
        to our _extract_db2 / _build_indock / _run_dock64 / _parse_outdock_score.
        Working dir is inside $SLURM_TMPDIR so the dockfiles copy lands on local
        NVMe (the fix for Lustre Fortran binary I/O mis-scoring).
        Expected result: -67.85

Test B: full pipeline (ligbuild + dock64), also inside $SLURM_TMPDIR.
        Expected result: comparable to -67.85 (possibly different pose but
        in the same range)
"""
import sys
import os
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deepdelgfn.mols.vina_scorer import Dock3Scorer

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"

PROJECT      = Path(__file__).resolve().parent.parent
INDOCK       = PROJECT / "ampc_dockfiles" / "INDOCK"
DOCKFILES    = PROJECT / "ampc_dockfiles"
PREBUILT_TGZ = PROJECT / "docking_runs" / "aspirin_ampc" / "lig_bundles" / "bundle_aspirin_000.tgz"

# Use SLURM_TMPDIR so dockfiles copies land on local NVMe
slurm_tmp = os.environ.get("SLURM_TMPDIR")
if slurm_tmp is None:
    raise RuntimeError("SLURM_TMPDIR is not set. This script must run inside a Slurm allocation.")
BASE = Path(slurm_tmp)

DEBUG_DIR_A = BASE / "debug_dock_A"
DEBUG_DIR_B = BASE / "debug_dock_B"

# ── Test A: pre-built bundle (bypass ligbuild) ────────────────────────────────
print("=" * 60)
print("TEST A: pre-built bundle (bypass ligbuild)")
print(f"  Working dir: {DEBUG_DIR_A}  (local NVMe)")
print(f"  Pre-built tgz: {PREBUILT_TGZ}")
print("=" * 60)

shutil.rmtree(DEBUG_DIR_A, ignore_errors=True)
DEBUG_DIR_A.mkdir(parents=True)

scorer_a = Dock3Scorer(indock_template=INDOCK, dockfiles_dir=DOCKFILES)
td_a = DEBUG_DIR_A / "workdir"
td_a.mkdir()

try:
    db2_a = scorer_a._extract_db2(PREBUILT_TGZ, td_a)
    print(f"  Extracted db2: {db2_a}  ({db2_a.stat().st_size} bytes)")

    run_dir_a = td_a / "run"
    run_dir_a.mkdir()
    indock_a = scorer_a._build_indock(db2_a, run_dir_a)

    scorer_a._run_dock64(run_dir_a, indock_a)

    outdock_a = run_dir_a / "OUTDOCK"
    lines_a = outdock_a.read_text().splitlines()
    print(f"  OUTDOCK ({len(lines_a)} lines) — scoring section:")
    in_scores = False
    for ln in lines_a:
        if "mol#" in ln and "id_num" in ln:
            in_scores = True
        if in_scores:
            print(f"    {ln}")

    score_a = scorer_a._parse_outdock_score(outdock_a)
    print(f"\n[Test A result] Score from pre-built bundle: {score_a}")
    print(f"  Expected: -67.85")

except Exception as e:
    print(f"  [Test A ERROR] {e}")
    import traceback; traceback.print_exc()

# ── Test B: full pipeline (ligbuild + dock64) ─────────────────────────────────
print()
print("=" * 60)
print("TEST B: full fresh-ligbuild pipeline")
print(f"  Working dir: {DEBUG_DIR_B}  (local NVMe, preserved for inspection)")
print("=" * 60)

shutil.rmtree(DEBUG_DIR_B, ignore_errors=True)
DEBUG_DIR_B.mkdir(parents=True)

scorer_b = Dock3Scorer(
    indock_template=INDOCK,
    dockfiles_dir=DOCKFILES,
    tmp_dir=DEBUG_DIR_B,   # preserved — inspect afterwards
)

score_b = scorer_b.score_smiles(ASPIRIN_SMILES, name="aspirin")
print(f"\n[Test B result] Score from fresh ligbuild: {score_b}")

fresh_db2s = list(DEBUG_DIR_B.rglob("*.db2"))
if fresh_db2s:
    print(f"  Fresh db2: {fresh_db2s[0]}  ({fresh_db2s[0].stat().st_size} bytes)")

# Print Test B OUTDOCK scoring section too
outdock_b_files = list(DEBUG_DIR_B.rglob("OUTDOCK"))
if outdock_b_files:
    outdock_b = outdock_b_files[0]
    lines_b = outdock_b.read_text().splitlines()
    print(f"  OUTDOCK ({len(lines_b)} lines) — scoring section:")
    in_scores = False
    for ln in lines_b:
        if "mol#" in ln and "id_num" in ln:
            in_scores = True
        if in_scores:
            print(f"    {ln}")
