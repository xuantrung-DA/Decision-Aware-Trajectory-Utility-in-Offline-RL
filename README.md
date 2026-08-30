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

> **Status (2026-08-30):** the frozen C1 analysis has a confirmatory IQL result from
> 20/86 valid official runs. Training for the remaining cross-algorithm and marginal-utility
> claims is in progress. Results below are an interim protocol-locked snapshot, not a pilot.

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

## Interim confirmatory result: C1

**Claim C1:** offline datasets with similar global statistics but different decision-critical
support can produce materially different downstream policy performance.

This snapshot uses IQL trained for 200k updates on each frozen condition. Every policy seed is
evaluated on the same predefined 20-episode, 250-step protocol; the final checkpoint is selected
without online validation. Seeds 0–2 belong to `A_CORE`, while preregistered headline seeds 3–4
belong to `D_HEADLINE_EXTRA_SEEDS`. These are 20 official matrix jobs, not smoke-test or pilot
runs.

### Matched-dataset audit

The comparison passed every frozen matching check before policy training:

| Audit item | Frozen observation | Result |
| --- | ---: | :---: |
| Trajectories | 180 vs. 180 | Pass |
| Transition-count mismatch | 1.34% | Pass |
| Mean-return mismatch | 0.00% | Pass |
| Global-coverage mismatch | 2.65% | Pass |
| Median-length mismatch | 0.45% | Pass |
| Standardized action-mean error | 0.0125 | Pass |
| Critical-coverage separation | 12.27% vs. 9.91% (+2.35 pp) | Pass |

The reference and critical-low dataset hashes are respectively
`ab1b0f9006e9b6dd4ee786b88134fddbc5b2802eb62f2a637fb0a507f3767c95` and
`9ce0d9fe7f9b9b59da9f1c38885d752a3314daa0c8c90ec4a6d91e4465792dc7`.

### Five-seed policy result

| Condition / paired contrast | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference critical-high success | 70% | 40% | 60% | 65% | 50% | 57.0% ± 12.0% |
| Critical-low success | 45% | 40% | 50% | 55% | 45% | 47.0% ± 5.7% |
| Paired difference | +25 pp | 0 pp | +10 pp | +10 pp | +5 pp | **+10.0 pp** |

| Preregistered analysis | Result |
| --- | ---: |
| Mean paired success difference | **+10.0 percentage points** |
| BCa 95% bootstrap interval | **[+4.0, +18.0] percentage points** |
| Paired Cohen's d | **1.07** |
| Seed directions | **4 positive, 1 tied, 0 negative** |
| Frozen C1 acceptance decision | **Pass** |

Within the frozen PointMaze–IQL setting, this supports C1: matching conventional dataset-level
statistics did not remove the performance gap associated with decision-critical support. This
result does **not** yet establish DATU's marginal-utility ranking or cross-algorithm generality.

### Claim status

| Claim | Current status | Required next evidence |
| --- | --- | --- |
| C1: critical support beyond global statistics | **IQL pass** | ReBRAC/BC mechanism context |
| C2: helpful vs. harmful marginal utility | Pending | Six removal interventions |
| C3: DATU vs. return/rarity/global coverage | Pending | Paired marginal-utility validation |
| C4: not an IQL-only artifact | Pending | ReBRAC plus focused CQL diagnostics |

The snapshot was recovered from a checkpoint archive with SHA-256
`9034a51f11db2567bd18e029cb9afb2a1048587b1fbe5456d72dcaa104fd3e9d`. All 20 final IQL
checkpoints loaded successfully; their training metrics and 2,320 serialized tensors were finite.

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
