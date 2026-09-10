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

Decision-Aware Trajectory Utility (DATU) is a **nonnegative offline ranking proxy** for this
signed empirical target, not a calibrated signed-utility predictor. It combines decision
sensitivity, ensemble disagreement, and local support contribution. The original scorer uses
one fixed IQL reference ensemble for both downstream learners. Environment rollouts measure
final policy performance and removal utility; they do not select the original groups or checkpoints.

> **Status (2026-09-10):** all **86/86 original full-training jobs** are complete, not pilot runs.
> Critical-support benefits are algorithm-dependent. Removing top-DATU groups hurts IQL and
> ReBRAC, but DATU does not consistently beat reward density. Six additional post-hoc ReBRAC
> random controls are prepared; their training results have **not yet been verified locally**.

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

The planned component ablations are DATU full, no uncertainty, no sensitivity, and no local
support. Descriptive **score-only** ablations on existing groups are available locally;
ablation-specific policy retraining is not complete. Exhaustive leave-one-group-out retraining,
subset-selection benchmarks, MuJoCo/AntMaze breadth, and the former 884-job matrix are outside
this study. Fitting the five-member reference scorer costs a further 500k updates, separate
from the 17.8M downstream updates.

## Execution progress

| Block | Complete | Planned | Status |
| --- | ---: | ---: | --- |
| Core phenomenon | **36** | 36 | Complete for BC, IQL, and ReBRAC |
| Headline precision | **8** | 8 | Complete |
| Marginal utility | **36** | 36 | Six removals, IQL and ReBRAC |
| CQL diagnostic | **6** | 6 | Complete |
| **Total** | **86** | **86** | **Original study complete** |

## Audited original-study results

Intervals below use corrected SciPy BCa bootstrap with 10,000 resamples, RNG seed 0, and
**paired training seeds** as the resampling unit. Probability differences are rounded to 12
decimals before resampling to stabilize floating-point ties. These small-sample intervals are
conditional on fixed datasets and the shared 20-episode evaluation panel; they are not adjusted
for multiple comparisons. `pp` means percentage points; paired effect size is Cohen's `d_z`.

### C1: decision-critical support under IQL

**Claim C1:** offline datasets with similar global statistics but different decision-critical
support can produce materially different downstream policy performance.

This analysis uses IQL trained for 200k updates on each frozen condition. Every policy seed is
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

| Frozen-protocol analysis | Result |
| --- | ---: |
| Mean paired success difference | **+10.0 percentage points** |
| BCa 95% bootstrap interval | **[+4.0, +18.0] percentage points** |
| Paired Cohen's d_z | **1.07** |
| Seed directions | **4 positive, 1 tied, 0 negative** |
| Within-study C1 threshold | **Met** |
| Strong-pass threshold (+15 pp mean) | Not met |

Within the frozen PointMaze–IQL setting, this supports C1: matching conventional dataset-level
statistics did not remove the performance gap associated with decision-critical support. This
result alone does **not** establish DATU's ranking superiority or cross-algorithm generality.

### Cross-algorithm mechanism boundary

All 36 `A_CORE` jobs are now complete. The preregistered reference-versus-critical-low contrast
is positive under IQL, but it changes sign under ReBRAC and BC:

| Algorithm | Seeds | Reference critical-high | Critical-low | Paired mean difference | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| IQL | 5 | **57.0%** | **47.0%** | **+10.0 pp** | Positive within this construction |
| ReBRAC | 3 | 41.7% | 45.0% | -3.3 pp | Does not replicate IQL |
| BC | 3 | 36.7% | 41.7% | -5.0 pp | Mechanism boundary |
| CQL | 3 | 53.3% | 38.3% | +15.0 pp | Wide CI: [-5.0, +35.0] pp |

For ReBRAC, the three paired differences are +15, -15, and -10 percentage points; their BCa
95% interval is [-13.3, +15.0] percentage points. The frozen cross-algorithm C4 criterion is
not met. CQL is directionally positive but imprecise; these results do not establish a universal
critical-support benefit or a formal causal mechanism.

The helpful-versus-harmful matched-condition contrast is also inconsistent across algorithms:

| Algorithm | Helpful low-return | Harmful low-return | Difference |
| --- | ---: | ---: | ---: |
| IQL, 5 seeds | 49.0% | 50.0% | -1.0 pp |
| ReBRAC, 3 seeds | 46.7% | 45.0% | +1.7 pp |
| BC, 3 seeds | 43.3% | 48.3% | -5.0 pp |

For IQL, the paired BCa 95% interval is [-8.0, +20.0] percentage points with `d_z = -0.066`.
The core comparison does not support a stable helpful-over-harmful advantage. The completed
controlled removals below also have algorithm-dependent signs.

### Group-removal utility

All 36 `B_MARGINAL` runs are complete. Each intervention removes 40 of 180 trajectories;
retained transition counts differ across original groups. Positive utility means removal hurts.

| Removed group | IQL utility, mean [BCa 95% CI], pp | ReBRAC utility, mean [BCa 95% CI], pp |
| --- | ---: | ---: |
| Top DATU | **+13.3 [+5.0, +30.0]** | **+11.7 [+5.0, +20.0]** |
| Bottom DATU | +8.3 [-10.0, +20.0] | 0.0 [-15.0, +8.3] |
| Top reward density | +15.0 [-10.0, +28.3] | -3.3 [-15.0, +3.3] |
| Bottom reward density | -8.3 [-30.0, +3.3] | -8.3 [-10.0, -5.0] |
| Helpful controlled | +8.3 [-15.0, +20.0] | -1.7 [-15.0, +10.0] |
| Harmful controlled | -1.7 [-25.0, +20.0] | +1.7 [-5.0, +5.0] |

Each cell uses three paired training seeds. IQL top-DATU effects are `[30, 5, 5]` pp;
their corrected CI is **[5, 30]**, not the [5, 21.7] interval from an earlier bootstrap
implementation. ReBRAC effects are `[20, 5, 10]` pp. This supports positive removal utility
for the top-scoring group, but does not by itself show superiority to random removal.

### Utility ranking against baselines

| Offline score | IQL pairwise accuracy | ReBRAC pairwise accuracy |
| --- | ---: | ---: |
| DATU | 10/14 (71.4%) | **10/15 (66.7%)** |
| Reward density | **14/14 (100%)** | 9/15 (60.0%) |
| Rarity | 7/14 (50.0%) | 7/15 (46.7%) |
| Global coverage | 39.3% | 23.3% |

These are descriptive rankings over six overlapping intervention groups, not independent
pairwise samples. Utility ties are excluded and prediction ties receive half credit.
DATU leads these baselines for ReBRAC, but reward density is stronger for IQL; universal
ranking superiority is not supported.

### Claim status

| Claim | Audited interpretation | Remaining limitation |
| --- | --- | --- |
| C1: critical support beyond global statistics | Positive within IQL | One fixed dataset construction |
| C2: helpful vs. harmful utility | No consistent cross-learner advantage | Mixed full-condition and removal results |
| C3: DATU vs. baseline scores | Algorithm-dependent; not universal superiority | Overlapping groups; random controls pending |
| C4: not an IQL-only artifact | Original replication criterion not met | CQL direction is positive but imprecise |

Additional limitations: selected trajectories all have return 1, so reward density is inverse
episode length; names such as `low_return` and `remove_top_return` are retained legacy IDs.
The frozen rarity implementation uses external neighbor ranks 2–21 for k=20, and local-support
search is approximate. These behaviors are disclosed, not silently changed after observing
outcomes. A fixed IQL scorer is not itself an algorithm-conditioned DATU implementation.

Source of this report: local final archive `datu_phase3_final_86jobs_kaggle_posix.zip`, SHA-256
`244f59cad14b4410a5590c4dada200a6c40b8cf1f57fa41f718b3f28d4ebbd4e`, audited into
`paper/audit/numbers.json`, `all_runs.csv`, `removals.csv`, and `ranking.csv`.
The original matrix signature is
`a6ec37a53ef4b60a66520252a44625b50399982e402f6137383ea725310c1168`.
Those local research artifacts are **not distributed in this source-only repository**.

## Repository layout

```text
configs/       Frozen PointMaze, algorithm, experiment, and ablation definitions
scripts/       Data preparation, DATU, training, evaluation, and aggregation entry points
src/datu/      Algorithms, trajectory data model, utility estimators, and evaluation code
tests/         Unit and matrix-integrity tests
```

Generated datasets, preregistrations, checkpoints, ZIP/TAR archives, logs, figures, the working
paper and internal notes (including local `ABOUT.md`) are excluded from Git. A clone contains
the implementation, not the historical run artifacts. Dataset and run manifests contain hashes;
exact replay requires the corresponding frozen artifacts rather than only regenerating paths.

## Installation

Python 3.11 is the reference environment.

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Then:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,robotics]"
```

Python 3.11 is used in CI; historical Kaggle policy runs used Python 3.12 and Torch 2.10.0.
Dependencies have lower bounds, not a full lockfile. Preserve run-level software versions when
reproducing results. Load only trusted checkpoints: model loading uses PyTorch serialization.

Verify without downloading data or launching official training (tests include tiny synthetic updates):

```bash
python -m ruff check src scripts tests
python -m pytest -q
python scripts/run_experiment.py --config configs/experiments/lab_review_core_v1.yaml --plan-dir results/plans/RELEASE_CHECK
```

## Reproduction

For a **new reproduction**, download the single active dataset manually. To resume existing
research, restore its verified artifacts and plan instead; do not rebuild the completed study.
The code does not download data implicitly:

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

The **matrix runner** requires explicit `--execute`; without it the runner only validates and
writes a plan. Other entrypoints differ: `train.py` starts training directly and `compute_datu.py`
can fit the reference ensemble when its cache is missing. The command below launches the
original full study and is not needed to inspect or resume supplemental work:

```bash
python scripts/run_experiment.py --config configs/experiments/lab_review_core_v1.yaml --execute
```

For a bounded Kaggle session, use the prioritized resumable runner:

```bash
python scripts/run_kaggle_session.py --execute --walltime-hours 10.5
```

Each progress line reports step, throughput, ETA, sampler/update latency, VRAM, and current losses.
NaN/Inf aborts immediately, interrupted runs preserve a checkpoint, and completed stable job IDs
are skipped on resume. This is queue-level recovery, not guaranteed exact optimizer/RNG resume
for an interrupted job. Use `--retry-failed` only after inspecting the failure.

## Post-hoc supplement

`IJMLC_SUPP_V1` is separate from the completed 86-run study:

- Two length-stratified random removal groups × three seeds × ReBRAC, 200k updates: six jobs.
- Match the removed transition budget to within 1%; freeze the first acceptable draws.
- Fresh 100-episode evaluation panel; average random draws **within each training seed** (n=3).
- Descriptive score-only component ablations reuse stored components and original groups.

Prepare the small input bundle from local frozen data, preregistration and the final archive:

```bash
python scripts/build_ijmlc_bundle.py --archive datu_phase3_final_86jobs_kaggle_posix.zip
```

This writes `dist/datu_ijmlc_supplement_input.zip`, `kaggle_ijmlc_cells.md` and a notebook.
Use those generated cells with expanded original checkpoint inputs on Kaggle.
`setup_ijmlc_kaggle.py` locates files by content and verifies hashes; `run_ijmlc_supplement.py`
performs CUDA/evaluation preflight, bounded training, reevaluation and output backups.
Updating GitHub does not update an existing Kaggle input. Never switch source underneath an
active session. Prepared configurations are not completed jobs, and no supplemental performance
claim is made here before its output has been audited.

## License

Released under the [MIT License](LICENSE).
