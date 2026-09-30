# CODA: Closed-Loop Online Diagnostic-Aware AutoRL

This repository contains the implementation, experimental configuration,
archived results, held-out evaluation data, and analysis code associated with
**CODA (Closed-Loop Online Diagnostic-Aware AutoRL)**.

CODA extends population-based online hyperparameter optimization with two
learner--optimizer communication channels:

- **I2O (Inside-to-Outside):** learner diagnostics are included in the
  population-level surrogate context.
- **O2I (Outside-to-Inside):** optimizer uncertainty is mapped to a bounded
  learner-side exploration actuator.

The experimental artifact covers PPO and SAC on four MuJoCo continuous-control
environments:

- `HalfCheetah-v5`
- `Hopper-v5`
- `Swimmer-v5`
- `Walker2d-v5`

The primary comparator is PB2. Additional baselines and ablations include PBT,
ASHA, CODA-I2O, and CODA-O2I.

---

## Artifact scope

The repository is intended to support two levels of reproducibility.

### Analysis-level reproducibility

The archived training logs and held-out evaluation outputs are sufficient to
reproduce the reported statistical summaries, tables, figures, channel audits,
factorial interaction analysis, and computational-cost analysis without
rerunning the original training experiments.

### Training-level reproducibility

Training code and configuration files are included for PPO, SAC, PBT, PB2,
ASHA, and the CODA variants.

The original full policy checkpoints are **not included in this artifact**.
Consequently, the held-out evaluation scripts are provided for transparency and
future reruns, but the archived held-out episode-level outputs are the inputs
used by the paper-analysis scripts in this release.

---

## Repository structure

```text
coda-autorl/
├── analysis/                  # Paper figures, tables, audits, and statistics
├── baselines/
│   ├── ppo/                   # PPO PBT/PB2/ASHA runners
│   └── sac/                   # SAC PBT/PB2/ASHA runners
├── configs/                   # Environments, seeds, PPO and SAC configuration
├── evaluation/                # Held-out evaluation scripts
├── experiments/
│   ├── ppo/                   # CODA PPO training entry point
│   └── sac/                   # CODA SAC training entry point
├── src/
│   └── coda/                  # CODA scheduler and implementation components
├── results/
│   ├── ppo/                   # PPO training/scheduler/held-out archives
│   ├── sac/                   # SAC training/scheduler/held-out archives
│   └── analysis/              # Reproduced analysis artifacts
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
└── .gitignore
```

Archived result directories contain raw metrics, scheduler logs, metadata, run
status information, champion-selection records, and held-out evaluation
outputs.

---

## Environment used for the experiments

The archived run metadata records the following principal software versions:

| Component | Version |
|---|---:|
| Python | 3.11.15 |
| Ray / RLlib | 2.55.1 |
| PyTorch | 2.5.1+cu121 |
| NumPy | 2.4.4 |
| pandas | 3.0.2 |
| SciPy | 1.17.1 |
| scikit-learn | 1.8.0 |
| Gymnasium | 1.2.2 |
| MuJoCo | 3.8.0 |

The original experiments were executed on a workstation with an Intel Core
i9-12900KF CPU, approximately 31 GiB RAM, and an NVIDIA RTX 3090 with 24 GiB
VRAM.

Two environment specifications are provided:

- `requirements.txt`: lightweight environment for reproducing the archived
  statistical analyses and figures.
- `environment.yml`: fuller environment intended for training/evaluation,
  including Ray/RLlib, PyTorch, Gymnasium, and MuJoCo.

Exact versions of `statsmodels` and `matplotlib` were not stored in the archived
run metadata, so the analysis requirements use bounded compatible versions for
those two packages rather than claiming an unrecorded exact version.

---

## Quick start: reproduce the analyses

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the analysis scripts:

```bash
python analysis/Tab04_held-out.py
python analysis/Tab05_paired_held-out.py
python analysis/Fig03_paired_heldout.py
python analysis/Fig04_training_convergence.py
python analysis/Tab06_terminal_training.py
python analysis/Tab07_paired_held-out.py
python analysis/Fig05_diagnostic_surrogate.py
python analysis/Tab8_comm_channels.py
python analysis/coda_factorial_interaction.py
python analysis/computational_cost.py
```

The standardized analysis scripts write generated artifacts under:

```text
results/analysis/
```

Each analysis directory contains numerical CSV outputs in addition to the
figure or LaTeX table body, allowing the reported values to be audited without
extracting numbers from rendered figures.

---

## Paper-to-script map

| Paper artifact | Script | Primary inputs | Default output |
|---|---|---|---|
| Table 4: held-out performance | `analysis/Tab04_held-out.py` | PPO/SAC held-out episode CSVs | `results/analysis/heldout_performance/` |
| Table 5: primary CODA--PB2 paired analysis | `analysis/Tab05_paired_held-out.py` | PPO/SAC held-out episode CSVs | `results/analysis/primary_paired_heldout/` |
| Figure 3: paired held-out effects | `analysis/Fig03_paired_heldout.py` | PPO/SAC held-out episode CSVs | `results/analysis/paired_heldout/` |
| Figure 4: training convergence | `analysis/Fig04_training_convergence.py` | training metric archives | `results/analysis/training_convergence/` |
| Table 6: terminal `tAUC100k` | `analysis/Tab06_terminal_training.py` | training metrics + held-out champion records | `results/analysis/terminal_training/` |
| Table 7: directional ablations | `analysis/Tab07_paired_held-out.py` | PPO/SAC held-out episode CSVs | `results/analysis/directional_ablation/` |
| Figure 5: diagnostic-surrogate controls | `analysis/Fig05_diagnostic_surrogate.py` | `analysis/diagnostic_surrogate_results.csv` | `results/analysis/diagnostic_surrogate/` |
| Table 8: communication-channel activity | `analysis/Tab8_comm_channels.py` | Full-CODA scheduler logs | `results/analysis/communication_channels/` |
| 2x2 factorial interaction | `analysis/coda_factorial_interaction.py` | PPO/SAC held-out episode CSVs | `results/analysis/factorial_interaction/` |
| Computational and interaction cost | `analysis/computational_cost.py` | PPO/SAC training metrics | `results/analysis/computational_cost/` |
| Execution-consistency audit | `analysis/audit_execution_consistency.py` | training + scheduler logs | user-specified output directory |

`analysis/Fig_training-heldout.py` is retained only as an additional/legacy
analysis and is not required to reproduce the main reported figures and tables.

---

## Execution-consistency audit

The execution audit requires both the training metrics and scheduler archives.

With the archive names included in the original artifact snapshot:

```bash
python analysis/audit_execution_consistency.py \
  --ppo-train results/ppo/ppo_train.zip \
  --ppo-scheduler results/ppo/ppo_scheduler.zip \
  --sac-train results/sac/sac_train.zip \
  --sac-scheduler results/sac/sac_scheduler.zip \
  --output-dir results/analysis/execution_consistency
```

The audit distinguishes reconstructable inheritance events from events for
which a compatible exported learner report can be linked. Unmatched exported
events are excluded from the effective-configuration audit denominator rather
than counted as mismatches.

---

## Training-metric and scheduler archive names

Some archived snapshots use the legacy names:

```text
results/ppo/ppo_train.zip
results/sac/sac_train.zip
results/ppo/ppo_scheduler.zip
results/sac/sac_scheduler.zip
```

The standardized analysis scripts may instead use the shorter aliases:

```text
results/ppo/metrics.zip
results/sac/metrics.zip
results/ppo/scheduler.zip
results/sac/scheduler.zip
```

Before creating a public release, use **one naming convention consistently**.
If the standardized scripts are retained unchanged, rename the four archives to
the shorter names above (or update the path constants in those scripts). Do not
keep duplicate copies solely to satisfy both names.

The extracted `results/<learner>/metrics/` and
`results/<learner>/scheduler/` directories are also retained as provenance
data where present.

---

## Held-out evaluation protocol

The paper-analysis scripts treat the **training seed** as the independent
experimental replicate.

For each method/environment/training-seed combination, the training-selected
champion is summarized using 100 held-out episodes. The episode-level
evaluations are not treated as 100 independent algorithm-level replicates.

The held-out data are stored under:

```text
results/ppo/heldout_reward_ppo_final/
results/sac/heldout_reward_sac_final/
```

Important files include:

```text
heldout_test_episodes.csv
heldout_test_seed_summary.csv
heldout_test_method_environment_summary.csv
training_champion_candidates.csv
heldout_episode_seeds.csv
test_protocol.json
```

---

## Statistical analysis

The principal analyses use matched training seeds.

The primary CODA--PB2 held-out family contains eight
learner--environment comparisons:

```text
2 learners x 4 environments = 8 comparisons
```

The directional ablation family contains 16 contrasts:

```text
2 learners x 4 environments x 2 directional contrasts = 16 comparisons
```

The separate exploratory 2x2 method-level interaction analysis uses:

```text
I = CODA - CODA-I2O - CODA-O2I + PB2
```

for each matched training seed and applies Holm correction across the eight
learner--environment interaction hypotheses.

Bootstrap confidence intervals use 10,000 matched-seed resamples in the
corresponding analysis scripts.

---

## Diagnostic-surrogate analysis

`analysis/Fig05_diagnostic_surrogate.py` summarizes the offline common-history
diagnostic-surrogate controls.

The primary figure uses the 70% learner-horizon cutoff. If the manuscript also
reports 60% and 80% sensitivity analyses, the released
`analysis/diagnostic_surrogate_results.csv` should contain those cutoffs as
well, so that all sensitivity claims can be regenerated directly from the
artifact.

---

## Computational cost

`analysis/computational_cost.py` computes, for each complete HPO run,

```text
N_steps = sum of final sampled interactions across participating trials
t_agg   = sum of final recorded execution time across participating trials
C_comp  = 1e5 * t_agg / N_steps
```

CODA relative computational cost is formed **within matched training seeds**
before aggregation.

The script uses the final exported report in causal order for each participant.
If an earlier row has a larger `time_total_s` value because of checkpoint
inheritance or logging behavior, this condition is retained in the participant
audit output rather than silently replacing the final value with the historical
maximum.

---

## Historical absolute paths

Some archived metadata contain absolute paths from the original execution
machine, for example paths beginning with `/home/...`.

These paths are retained as provenance metadata. Reproduction scripts should
resolve files relative to the repository and must not depend on the historical
absolute paths.

---

## Raw data and generated outputs

Raw archived logs and held-out outputs should be treated as immutable
provenance data.

Generated analysis artifacts belong under:

```text
results/analysis/
```

When regenerating a paper result, prefer writing a new derived CSV/figure/table
there rather than modifying the underlying raw log files.

---

## Checkpoints

Full training checkpoints are not included in this repository because of their
storage footprint.

Therefore:

- the archived held-out episode-level outputs can be reanalyzed directly;
- the provided evaluation scripts document the evaluation procedure;
- exact policy reevaluation requires the corresponding original checkpoints.



---

## Re-running training

A full training environment can be created with:

```bash
conda env create -f environment.yml
conda activate coda-autorl
```

Training experiments are computationally expensive and will not necessarily be
bitwise deterministic across different GPU drivers, CUDA versions, Ray/RLlib
versions, or MuJoCo installations.

Configuration sources are under:

```text
configs/
```

Baseline runners are under:

```text
baselines/ppo/
baselines/sac/
```

CODA experiment entry points are under:

```text
experiments/ppo/
experiments/sac/
```

Implementation components are under:

```text
src/coda/
```


---

## License

Unless a source file states otherwise, original code in this repository is
released under the MIT License; see `LICENSE`.

Third-party software, copied/adapted upstream code, and external datasets remain
subject to their respective licenses. Preserve upstream copyright notices,
license headers, and NOTICE files where applicable.

---

## Citation

A formal citation should be added here once the manuscript has a stable
bibliographic record.

For review-stage repositories, a minimal placeholder is preferable to inventing
publication metadata:

```text
CODA: Closed-Loop Online Diagnostic-Aware AutoRL with Bidirectional
Learner-Optimizer Communication. Manuscript under review.
```
