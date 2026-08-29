# Decision-Aware Trajectory Utility in Offline RL

[![CI](https://github.com/xuantrung-DA/Decision-Aware-Trajectory-Utility-in-Offline-RL/actions/workflows/ci.yml/badge.svg)](https://github.com/xuantrung-DA/Decision-Aware-Trajectory-Utility-in-Offline-RL/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research code for identifying trajectories that are important because they support consequential,
weakly represented decisions—not merely because they have high return or improve global coverage.

For an offline RL algorithm \(A\), dataset \(D\), and trajectory group \(G\), the empirical target is

$$
U_A(G \mid D) = J(A,D) - J(A,D \setminus G).
$$

Decision-Aware Trajectory Utility (DATU) estimates this signed, algorithm-conditional marginal
utility using only frozen offline data. It combines decision sensitivity, epistemic uncertainty,
and local support contribution. Environment rollouts are used only for final policy evaluation;
they never define the critical region, intervention groups, utility scores, or checkpoints.

> **Status:** the protocol and data interventions are frozen, and policy training is in progress.
> No empirical performance result is claimed in this repository yet.

## Research questions

- **C1:** Can datasets with matched global statistics but different decision-critical support lead
  to different policy performance?
- **C2:** Does helpful low-return data have greater marginal utility than harmful low-return data?
- **C3:** Does DATU predict marginal utility better than reward density, rarity, and global coverage?
- **C4:** Does the phenomenon persist beyond IQL?

The frozen study uses `D4RL/pointmaze/medium-v2` and 86 full-training jobs:

| Block | Design | Jobs |
| --- | --- | ---: |
| Core phenomenon | 4 matched conditions × BC/IQL/ReBRAC × 3 seeds, 200k updates | 36 |
| Marginal utility | 6 predefined removals × IQL/ReBRAC × 3 seeds, 200k updates | 36 |
| CQL diagnostic | 2 diagnostic conditions × 3 seeds, 300k updates | 6 |
| Headline precision | 4 IQL conditions × 2 additional seeds, 200k updates | 8 |
| **Total** | **17.8M gradient updates** | **86** |

The component ablation is fixed to DATU full, no uncertainty, no sensitivity, and no local
support. Exhaustive leave-one-group-out retraining, subset-selection benchmarks, MuJoCo/AntMaze
breadth, and the former 884-job Cartesian matrix are outside this study.

## Repository layout

```text
configs/       Frozen PointMaze, algorithm, experiment, and ablation definitions
scripts/       Data preparation, DATU, training, evaluation, and aggregation entry points
src/datu/      Algorithms, trajectory data model, utility estimators, and evaluation code
tests/         Unit and matrix-integrity tests
```

Generated datasets, preregistrations, checkpoints, logs, figures, and internal notes are excluded
from Git. Dataset and run manifests contain content hashes so local/Kaggle artifacts can still be
audited.

## Installation

Python 3.11 is the reference environment.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,robotics]"
```

Verify the code without downloading data:

```bash
python -m ruff check src scripts tests
python -m pytest -q
python scripts/smoke_test.py
python scripts/run_experiment.py --config configs/experiments/lab_review_core_v1.yaml
```

## Reproduction

The code never downloads a dataset implicitly. Download the single active dataset manually:

```bash
python scripts/download_datasets.py D4RL/pointmaze/medium-v2
minari download D4RL/pointmaze/medium-v2
```

Then prepare and freeze all offline artifacts in this order:

```bash
python scripts/prepare_dataset.py --config configs/datasets/pointmaze_medium.yaml --output data/processed/pointmaze_medium
python scripts/inspect_pointmaze.py --dataset data/processed/pointmaze_medium --output-dir results/dataset_audit/pointmaze_medium
python scripts/build_matched_datasets.py --config configs/matched/pointmaze.yaml --require-match
python scripts/compute_datu.py --config configs/experiments/pointmaze_iql.yaml --output-dir results/datu/pointmaze_core_reference --ensemble-cache results/reference_critics/pointmaze_core_E5 --critic-steps 100000 --ensemble-size 5 --knn-k 50 --aggregation topk_mean dataset.path=data/processed/matched/pointmaze_medium/reference_critical_high
python scripts/freeze_preregistration.py --reference-dataset data/processed/matched/pointmaze_medium/reference_critical_high --matched-manifest data/manifests/lab_review_core_v1/matched_pointmaze.json --datu-components results/datu/pointmaze_core_reference/datu_components.npz --removal-dir data/processed/removals/pointmaze_core --output data/manifests/lab_review_core_v1/pre_registration.json
python scripts/run_experiment.py --config configs/experiments/lab_review_core_v1.yaml --preflight-data
```

Training requires an explicit `--execute`; without it the runner only validates and writes a plan:

```bash
python scripts/run_experiment.py --config configs/experiments/lab_review_core_v1.yaml --execute
```

For a bounded Kaggle session, use the prioritized resumable runner:

```bash
python scripts/run_kaggle_session.py --execute --walltime-hours 10.5
```

Each progress line reports step, throughput, ETA, sampler/update latency, VRAM, and current losses.
NaN/Inf aborts immediately, interrupted runs preserve a checkpoint, and completed stable job IDs
are skipped on resume. Use `--retry-failed` only after fixing an infrastructure-level failure.

## License

Released under the [MIT License](LICENSE).
