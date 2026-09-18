#!/usr/bin/env python3
"""Dock a single SMILES with the repo's Dock3 AmpC setup.

This is a lightweight command-line wrapper around
``deepdelgfn.mols.vina_scorer.Dock3Scorer`` using the prepared AmpC dockfiles
bundled in this repository.

Typical usage on the cluster:

    source /project/rrg-mailhoto/share/dockingpackages/dockenv.sh
    export PYTHONPATH=/absolute/path/to/DEL-GFN-2/src:$PYTHONPATH
    python scripts/dock_smiles_ampc.py \
        --smiles 'CC(=O)Oc1ccccc1C(=O)O' \
        --name aspirin

If you want to preserve intermediates (ligbuild output, patched INDOCK, OUTDOCK,
copied dockfiles, etc.), pass ``--work-dir`` to point at a persistent directory.
Otherwise the scorer will use a temporary directory, preferring ``$SLURM_TMPDIR``
when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from deepdelgfn.mols.vina_scorer import Dock3Scorer


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Dock a single SMILES against the AmpC Dock3 setup in this repo."
    )
    ap.add_argument("--smiles", required=True, help="SMILES string to dock.")
    ap.add_argument(
        "--name",
        default="lig",
        help="Ligand name used for temporary file naming (default: %(default)s).",
    )
    ap.add_argument(
        "--indock",
        default=str(PROJECT_ROOT / "ampc_dockfiles" / "INDOCK"),
        help="Path to INDOCK template (default: repo AmpC INDOCK).",
    )
    ap.add_argument(
        "--dockfiles",
        default=str(PROJECT_ROOT / "ampc_dockfiles"),
        help="Path to dockfiles directory (default: repo ampc_dockfiles/).",
    )
    ap.add_argument(
        "--dockenv-sh",
        default="/project/rrg-mailhoto/share/dockingpackages/dockenv.sh",
        help="Path to dockenv.sh used to expose ligbuild and its environment.",
    )
    ap.add_argument(
        "--dock64",
        default="/project/rrg-mailhoto/share/dock64",
        help="Path to dock64 executable.",
    )
    ap.add_argument(
        "--ligbuild",
        default="ligbuild",
        help="ligbuild executable name/path (default: %(default)s).",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help=(
            "Persistent working directory for debugging/artifact inspection. "
            "If omitted, a temporary directory is used."
        ),
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds for ligbuild/dock64 subprocesses (default: %(default)s).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON object instead of a plain score.",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    scorer = Dock3Scorer(
        indock_template=args.indock,
        dockfiles_dir=args.dockfiles,
        dockenv_sh=args.dockenv_sh,
        dock64_exe=args.dock64,
        ligbuild_exe=args.ligbuild,
        tmp_dir=args.work_dir,
        timeout=args.timeout,
    )
    score = scorer.score_smiles(args.smiles, name=args.name)

    if args.json:
        print(
            json.dumps(
                {
                    "name": args.name,
                    "smiles": args.smiles,
                    "docking_score": score,
                    "indock": str(Path(args.indock).resolve()),
                    "dockfiles": str(Path(args.dockfiles).resolve()),
                    "work_dir": None if args.work_dir is None else str(Path(args.work_dir).resolve()),
                }
            )
        )
    else:
        print(score)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())