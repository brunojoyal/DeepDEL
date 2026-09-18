# DEL-GFN

GFlowNet-based framework for designing **DNA-Encoded Libraries (DELs)**. It
models a DEL as a three-cycle (trimeric) library built from building-block (BB)
pools, scores candidate products with a learned surrogate of docking, and trains
a GFlowNet to sample high-quality libraries.

---

## Repository layout

```
.
├── data/                       # building blocks + receptor PDBQTs
│   ├── bbs.csv                 # building-block table (ID, SMILES, Name)
│   └── targets/                # Vina receptors (Mpro, TBLR1, ClpP, sEH)
└── src/deepdelgfn/
    ├── autodock_proxy/         # surrogate docking model (RF / NN)
    │   ├── train_rf.py         #   Random Forest trainer  -> .joblib
    │   ├── train_nn.py         #   PyTorch MLP trainer    -> .pt
    │   └── model.py            #   unified loader/predictor
    ├── deepdel/                # DeepDEL reward model + dataset generation
    │   ├── generate_dataset.py #   enumerate BB triples, score with proxy
    │   ├── train_offline.py    #   pure offline DeepDEL training
    │   └── fourier.py          #   threshold conditioning (NeRF Fourier)
    ├── gfn/                    # GFlowNet samplers
    │   ├── train_gfn.py            # RxnFlow-style (DeepSets state)
    │   ├── train_gfn_hierarchical.py  # H-DEL-GFlowNet (cycle/cluster/BB)
    │   ├── train_gfn_multihot.py  # flat multi-hot state
    │   └── sample_mcmc.py        # MCMC baseline over DeepDEL reward
    ├── mols/                   # chemistry, library enumeration, docking
    │   ├── dels.py             #   trimer chemistry / reaction modes
    │   ├── vina_scorer.py      #   Vina & DOCK3 scorers
    │   ├── score_library.py    #   enumerate + score a BB block set
    │   ├── dock_library_parallel.py  # dock a pre-generated CSV
    │   └── eval_autodock_proxy_topm_scores.py  # proxy-score top-m libraries
    ├── models/                 # DeepSets TripleDeepSet + checkpoint helpers
    ├── rewards.py              # library-level reward aggregation
    └── utils/                  # scoring/plotting/weight utilities
```

---

## Requirements

### Python packages

| Package        | Used for                                  |
| -------------- | ----------------------------------------- |
| `rdkit`        | SMILES parsing, standardization, ECFP/Morgan fingerprints, reactions |
| `torch`        | NN proxy, DeepDEL, GFlowNet training      |
| `numpy`        | array/fingerprint math                    |
| `pandas`       | CSV I/O, BB tables, dataset frames        |
| `scikit-learn` | RandomForest proxy, train/test splits     |
| `joblib`       | RF proxy persistence                      |
| `matplotlib`   | training/dataset plots                    |
| `scipy`        | Kendall's tau (optional, in `train_offline`) |

Example install (adjust for your environment):

```bash
conda create -n deepdel python=3.10 -y
conda activate deepdel
conda install -c conda-forge rdkit numpy pandas scipy matplotlib -y
pip install torch scikit-learn joblib
```

### External docking tools (only needed for docking/scoring)

Docking is optional — dataset generation, proxy training, and GFlowNet training
never call an external binary. You only need these when you run the docking
components.

**AutoDock Vina backend** (`--docking-backend vina`):

* `vina` or `qvina02` — the docking engine (on `PATH` or via `--engine`).
* `prepare_ligand` (ADFR) or `mk_prepare_ligand.py` (Meeko) — ligand PDBQT
  preparation (the scorer tries both).
* `obabel` (Open Babel) — fallback ligand PDBQT preparation.

**DOCK3 backend** (`--docking-backend dock3`):

* `ligbuild` and `dock64` on a prepared DOCK3 installation (typically sourced
  via a `dockenv.sh`), plus a target `INDOCK` template and a `dockfiles/`
  directory.

---

## Setup

```bash
git clone https://github.com/brunojoyal/DeepDEL.git
cd DeepDEL
# The package lives under src/; put it on PYTHONPATH.
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

All library entry points are invoked as modules, e.g.:

```bash
python -m deepdelgfn.deepdel.train_offline --help
```

---

## Building-block data

`data/bbs.csv` is the canonical building-block table. Required columns are
`ID` and `SMILES`; `Name` is optional. An **optional** `pool` column (values
`1`/`2`/`3`) tags each BB's cycle for three-pool setups; when absent, all three
cycles draw from the same BB universe.

The building blocks in `data/bbs.csv` are from the **Enamine** screening-library
catalog (see the `EN300-…` catalog IDs in the `Name` column).

```csv
ID,SMILES,Name
1,Nc1cccc(C#Cc2cccc(C(=O)O)c2)c1,EN300-27576
2,C#CC[C@@H](N)C(=O)O,EN300-100145
...
```

### Reaction modes

Chemistry lives in `deepdelgfn/mols/dels.py`. Three modes are supported:

| Mode                  | Step 1          | Step 2              | Target |
| --------------------- | --------------- | ------------------- | ------ |
| `amide_sulfonamide`   | amide (BB1-NH₂ + BB2-COOH) | sulfonamide (BB3-SO₂Cl + NH₂) | AmpC β-lactamase |
| `amide_amide`         | amide           | amide (BB3 = amino acid) | sEH |
| `amide_amide_legacy` (default) | BB1 acid + BB2 NH₂ | BB2 acid + BB3 NH₂ | sEH |

---

## Pipeline components

Every component below can be run **independently** with
`python -m deepdelgfn.<module>`. Together they form the pipeline:

```
docking scores ──▶ AutoDock proxy (RF/NN) ──▶ DeepDEL dataset
                                                     │
DeepDEL reward model ◀───────────────────────────────┘
        │
        └──▶ GFlowNet / MCMC samplers ──▶ top-m candidates
                                                  │
                            proxy eval / real docking ◀┘
```

### 1. Train an AutoDock proxy

The proxy learns `SMILES → docking score` from previously docked molecules
(CSVs with a SMILES column and a docking-score column). It exists so that the
rest of the pipeline can score arbitrarily many products cheaply.

**Random Forest** (`.joblib`):

```bash
python -m deepdelgfn.autodock_proxy.train_rf \
  --inputs data/scored_libraries \
  --smiles_col smiles --target_col docking_score \
  --model_out models/autodock_model.joblib \
  --preds_out outputs/autodock_proxy/predictions.csv \
  --n_estimators 300
```

**Neural network** (`.pt`):

```bash
python -m deepdelgfn.autodock_proxy.train_nn \
  --inputs data/scored_libraries \
  --smiles_col smiles --target_col docking_score \
  --model_out models/autodock_model_nn.pt \
  --preds_out outputs/autodock_proxy/predictions_nn.csv \
  --n_bits 4096 --hidden_dim 512 --n_layers 3 \
  --standardize_y --dedup_smiles
```

`--inputs` accepts files and/or directories (searched recursively).

**Predict a single molecule** with either artifact:

```bash
python -m deepdelgfn.utils.score_mol \
  --model models/autodock_model_nn.pt \
  --smiles 'CC(=O)Oc1ccccc1C(=O)O'
```

### 2. Generate a DeepDEL dataset

Enumerate random BB triples, build the resulting trimer products, score them
with the AutoDock proxy, aggregate each library into the **threshold reward**
(the default library reward), and stream an ID-only CSV
(`B1_id,B2_id,B3_id,threshold,y`) for DeepDEL training. The threshold reward is
`1 + Σ 1 / (1 + exp((score − threshold) / α))` over a library's products:

```bash
python -m deepdelgfn.deepdel.generate_dataset \
  --bbs data/bbs.csv \
  --autodock-model models/autodock_model_nn.pt \
  --reward threshold --threshold -8 --alpha 1.0 \
  --min-size1 6 --max-size1 6 \
  --min-size2 6 --max-size2 6 \
  --min-size3 6 --max-size3 6 \
  --num-triplets 10000 \
  --reaction-mode amide_amide_legacy \
  --out outputs/deepdel/deepdel_dataset.csv
```

To learn a threshold-conditioned reward model, sample thresholds over an
interval and store log-rewards (recommended for a GFlowNet downstream):

```bash
python -m deepdelgfn.deepdel.generate_dataset \
  --bbs data/bbs.csv \
  --autodock-model models/autodock_model_nn.pt \
  --reward threshold --threshold-min -12 --threshold-max -6 --alpha 1.0 \
  --min-size1 6 --max-size1 6 \
  --min-size2 6 --max-size2 6 \
  --min-size3 6 --max-size3 6 \
  --num-triplets 10000 \
  --reaction-mode amide_amide_legacy \
  --log-target \
  --out outputs/deepdel/deepdel_dataset_threshold.csv
```

Useful knobs: `--device cuda` (GPU proxy inference), `--n-threads` (CPU worker
processes), `--max-weight` / `--weight-source` (exclude heavy products),
`--bad-builds-csv` (log chemistry failures instead of aborting).

### 3. Train DeepDEL (offline)

The DeepDEL model is a `TripleDeepSet` that predicts a library's aggregated
reward from the (permutation-invariant) fingerprints of its building blocks. It
is later frozen and used as the GFlowNet reward oracle.

```bash
python -m deepdelgfn.deepdel.train_offline \
  --bbs data/bbs.csv \
  --dataset outputs/deepdel/deepdel_dataset.csv \
  --bb-fp-bits 2048 \
  --hidden-dim 512 --rho-dim 512 \
  --epochs 20 --batch-size 64 --lr 1e-4 \
  --save-last models/deepdel.pt \
  --log-target
```

Common options: `--pooling mean|sum`, `--shared-phi`, `--output-head
linear|sigmoid_scaled|softplus_scaled` (non-linear heads require `--lib-size`),
`--patience` (early stopping), `--resume` (continue from a checkpoint), and the
`--fourier-*` flags (threshold conditioning). For a dataset whose `y` column is
`log R`, pass `--log-target` so downstream consumers interpret the output as
log-reward.

### 4. Sample libraries with a GFlowNet

**RxnFlow-style GFlowNet** (DeepSets state, action subsampling, Trajectory
Balance). The frozen DeepDEL checkpoint supplies both the φ state embedding and
the terminal reward:

```bash
python -m deepdelgfn.gfn.train_gfn \
  --bbs data/bbs.csv \
  --deepdel models/deepdel.pt \
  --size1 6 --size2 6 --size3 6 \
  --beta 100 --steps 2000 --lr 1e-4 --logz-lr 1.0 \
  --batch-trajectories 8 \
  --subsample-ratio-start 0.0 --subsample-ratio-end 0.02 \
  --topm 100 \
  --outdir outputs/gfn \
  --policy-ckpt outputs/gfn/gfn_policy.pt
```

The phi↔policy coupling is controlled by `--phi-train-mode
frozen_table|trainable_table|trainable_network` and `--action-repr ecfp|phi`.
To drive the reward directly from the AutoDock proxy instead of DeepDEL, use
`--gfn-reward-source autodock_proxy --autodock-model models/autodock_model_nn.pt`.

**Hierarchical GFlowNet** (H-DEL-GFlowNet: cycle → cluster → BB). Requires a
cluster-assignment JSON (`{cycle: {bb_idx: cluster_id}}`):

```bash
python -m deepdelgfn.gfn.train_gfn_hierarchical \
  --bbs data/bbs.csv \
  --clusters data/clusters.json \
  --deepdel models/deepdel.pt \
  --size1 6 --size2 6 --size3 6 \
  --beta 50 --steps 2000 \
  --outdir outputs/gfn_hierarchical
```

**Multi-hot (flat) GFlowNet** — a flat bit-flip policy over the concatenated BB
pools:

```bash
python -m deepdelgfn.gfn.train_gfn_multihot \
  --bbs data/bbs.csv \
  --deepdel models/deepdel.pt \
  --size1 6 --size2 6 --size3 6 \
  --beta 50 --steps 2000 \
  --outdir outputs/gfn_multihot
```

**MCMC baseline** (Metropolis-Hastings over the frozen DeepDEL reward, optional
parallel tempering) — produced for reproducible comparison against the GFlowNet
samplers:

```bash
python -m deepdelgfn.gfn.sample_mcmc \
  --bbs data/bbs.csv \
  --deepdel models/deepdel.pt \
  --size1 6 --size2 6 --size3 6 \
  --beta 200 --mcmc-steps 100000 --burn-in 10000 \
  --topm 100 --outdir outputs/mcmc
```

All three GFlowNet trainers and the MCMC sampler write a `topm_rewards.csv`
(`rank,reward,yhat,B1_id,B2_id,B3_id,threshold`) of their top-m candidates.

### 5. Evaluate top-m candidates with the proxy

Re-score the top-m libraries selected by a sampler using the AutoDock proxy, and
reduce each library to a scalar reward:

```bash
python -m deepdelgfn.mols.eval_autodock_proxy_topm_scores \
  --bbs data/bbs.csv \
  --topm-csv outputs/gfn/topm_rewards.csv \
  --autodock-model models/autodock_model_nn.pt \
  --out outputs/gfn/topm_actual_scores.csv \
  --reward threshold --threshold -8 --alpha 1.0 \
  --reaction-mode amide_amide_legacy
```

`--dataset-out` can also emit the enumerated per-molecule rows in DeepDEL dataset
format, and `--shard-index`/`--num-shards` split the work across array jobs.

### 6. Enumerate, generate, and dock libraries

`score_library` turns an explicit set of building-block IDs (or a random block
set) into enumerated trimer products, then either emits them unscored or docks
them. Three pool sources are supported: separate `--pool1/--pool2/--pool3` CSVs,
a single `--pool` CSV, or a `--bbs-combined` CSV with a `pool` column.

**Generate only (no docking)** — produces an unscored CSV with the product
SMILES, which you can hand to `dock_library_parallel`:

```bash
python -m deepdelgfn.mols.score_library \
  --pool data/bbs.csv \
  --reaction-mode amide_amide_legacy \
  --blocks '1|2,3,4|5' \
  --generate-only \
  --out-dir outputs/library --out-csv outputs/library/unscored.csv
```

`--blocks 'b1_ids,b2_ids,b3_ids'` lists pipe-separated IDs per cycle (enumerates
the Cartesian product); `--random N` samples N random triples instead.

**Dock a block set with Vina:**

```bash
python -m deepdelgfn.mols.score_library \
  --pool data/bbs.csv \
  --reaction-mode amide_amide_legacy \
  --docking-backend vina --target sEH \
  --blocks '1|2,3,4|5' \
  --exhaustiveness 8 --jobs 8 \
  --out-dir outputs/library --out-csv outputs/library/scored.csv
```

`--target` selects a receptor from `data/targets/<target>.pdbqt` and its
pre-configured box (`Mpro`, `TBLR1`, `ClpP`, `sEH`); `--receptor` + `--box` let
you override either.

**Dock a pre-generated CSV** (molecule-level parallelism):

```bash
python -m deepdelgfn.mols.dock_library_parallel \
  --unscored-csv outputs/library/unscored.csv \
  --out-csv outputs/library/scored.csv \
  --backend vina --target sEH --n-proc 8
```

**DOCK3 backend** — supply the `INDOCK` template and `dockfiles/` directory
instead (`--docking-backend dock3 --indock-template ... --dockfiles ...` in
`score_library`, or `--backend dock3 --indock ... --dockfiles ...` in
`dock_library_parallel`).

