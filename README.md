# CODA: Closed-Loop Online Diagnostic-Aware AutoRL

**CODA** is a population-based framework for **online hyperparameter optimization in deep reinforcement learning**. Its main contribution is a bidirectional communication mechanism between the inner reinforcement-learning learner and the outer hyperparameter optimizer:

- **Inner-to-Outer (I2O):** learner diagnostics condition the outer optimization process.
- **Outer-to-Inner (O2I):** uncertainty from the outer surrogate is communicated back to the learner through a learner-specific actuator.

The framework is evaluated with two reinforcement-learning learners:

- **Proximal Policy Optimization (PPO)** — on-policy.
- **Soft Actor-Critic (SAC)** — off-policy.

The repository contains CODA implementations, external baselines, directional ablations, training scripts, champion selection, held-out evaluation, control-quality metrics, statistical analysis, and processed experimental results.

---

## Method Overview

CODA implements a closed-loop interaction of the form

\[
\text{learner diagnostics}
\rightarrow
\text{I2O state}
\rightarrow
\text{outer optimizer}
\rightarrow
\text{O2I uncertainty}
\rightarrow
\text{learner actuator}.
\]

The communication architecture is shared across learners, while the diagnostic signal, hyperparameter coordinates, and actuator are learner-specific.

### PPO instantiation

The PPO instantiation uses:

- an approximate-policy-KL diagnostic state;
- four online hyperparameter coordinates;
- entropy regularization as the O2I actuator.

### SAC instantiation

The SAC instantiation uses:

- a TD-error-based diagnostic state;
- four online hyperparameter coordinates;
- target entropy as the O2I actuator.

SAC continues to optimize its temperature parameter internally.

---

## Compared Methods

CODA is evaluated against three external hyperparameter-optimization baselines:

| Method | Online HP adaptation | Population | Bayesian surrogate | Checkpoint inheritance | Early stopping |
|---|---:|---:|---:|---:|---:|
| PBT | Yes | Yes | No | Yes | No |
| PB2 | Yes | Yes | Yes | Yes | No |
| ASHA | No | No | No | No | Yes |
| CODA | Yes | Yes | Yes | Yes | No |

Directional variants are also available:

- **CODA-I2O:** diagnostic-aware outer optimization without O2I actuation.
- **CODA-O2I:** O2I actuation without diagnostic conditioning.

---

## Benchmarks

Experiments use continuous-control tasks from **Gymnasium MuJoCo v5**:

- `Hopper-v5`
- `HalfCheetah-v5`
- `Walker2d-v5`
- `Swimmer-v5`

The same benchmark set is used for PPO and SAC to support a controlled cross-learner comparison.

---

## Repository Structure

```text
CODA/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── environments.py
│   ├── seeds.py
│   ├── ppo_config.py
│   └── sac_config.py
│
├── src/
│   └── coda/
│       ├── schedulers/
│       │   ├── coda_ppo_scheduler.py
│       │   └── coda_sac_scheduler.py
│       ├── diagnostics/
│       │   ├── ppo_diagnostics.py
│       │   └── sac_td_diagnostics.py
│       └── utils/
│           ├── checkpoint_utils.py
│           ├── metrics_utils.py
│           └── reproducibility.py
│
├── experiments/
│   ├── ppo/
│   │   ├── train_pbt.py
│   │   ├── train_pb2.py
│   │   ├── train_asha.py
│   │   ├── train_coda.py
│   │   ├── train_coda_i2o.py
│   │   └── train_coda_o2i.py
│   └── sac/
│       ├── train_pbt.py
│       ├── train_pb2.py
│       ├── train_asha.py
│       ├── train_coda.py
│       ├── train_coda_i2o.py
│       └── train_coda_o2i.py
│
├── evaluation/
│   ├── heldout_test_ppo.py
│   ├── heldout_test_sac.py
│   └── stability_metrics.py
│
├── analysis/
│   ├── training_curves.py
│   ├── heldout_statistics.py
│   ├── stability_analysis.py
│   ├── ablation_analysis.py
│   ├── runtime_analysis.py
│   └── generate_tables.py
│
├── results/
│   ├── ppo/
│   │   ├── training/
│   │   ├── scheduler/
│   │   ├── heldout/
│   │   ├── statistics/
│   │   └── runtime/
│   └── sac/
│       ├── training/
│       ├── scheduler/
│       ├── heldout/
│       ├── statistics/
│       └── runtime/
│
├── figures/
├── tables/
├── checkpoints/
│   └── README.md
└── docs/
    ├── experimental_protocol.md
    ├── reproducibility.md
    └── result_files.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<USERNAME>/<REPOSITORY>.git
cd <REPOSITORY>
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- Ray / RLlib
- Ray Tune
- Gymnasium
- MuJoCo
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn

> **Important:** the experiments use RLlib's legacy API stack. Exact package versions should be frozen in `requirements.txt` for reproducibility.

---

## Reproducibility

### Training seeds

The experimental protocol uses the following independent training seeds:

```text
1042
2854
3910
4721
5603
6198
7433
8256
9107
9845
```

### Held-out evaluation seeds

Held-out evaluation uses a disjoint block of reset seeds:

```text
100000–100099
```

Champion selection is performed using **training data only**. Held-out returns are never used for checkpoint selection.

The statistical unit is the **independent training seed**, not the individual held-out episode.

---

## Running the Experiments

Run scripts from the repository root so custom schedulers and callbacks remain importable during checkpoint creation and restoration.

### PPO

```bash
python experiments/ppo/train_pbt.py
python experiments/ppo/train_pb2.py
python experiments/ppo/train_asha.py
python experiments/ppo/train_coda.py
```

Directional variants:

```bash
python experiments/ppo/train_coda_i2o.py
python experiments/ppo/train_coda_o2i.py
```

### SAC

```bash
python experiments/sac/train_pbt.py
python experiments/sac/train_pb2.py
python experiments/sac/train_asha.py
python experiments/sac/train_coda.py
```

Directional variants, when included:

```bash
python experiments/sac/train_coda_i2o.py
python experiments/sac/train_coda_o2i.py
```

---

## SAC Configuration

The SAC search space is

| Hyperparameter | Range | Sampling |
|---|---:|---|
| Training batch size | `[256, 2048]` | Integer uniform |
| `tau` | `[1e-3, 2e-2]` | Log-uniform |
| Actor learning rate | `[1e-5, 3e-4]` | Log-uniform |
| Critic learning rate | `[1e-4, 1e-3]` | Log-uniform |

Common fixed settings:

```text
gamma                         = 0.99
initial_alpha                 = 1.0
entropy_learning_rate         = 3e-4
n_step                        = 1
learning_starts               = 1500
training_intensity            = 4
replay_buffer_capacity        = 250000
num_env_runners               = 4
num_envs_per_env_runner       = 8
rollout_fragment_length       = 8
network                       = [256, 256], ReLU
```

PBT, PB2, and CODA use a population of four workers. The SAC campaign uses a maximum of **1,000,000 environment interactions per worker**.

ASHA uses static sampled configurations with asynchronous resource allocation and is configured to approximately match the nominal aggregate interaction budget of the population-based methods.

---

## PPO Configuration

The PPO study uses the same PPO implementation across PBT, PB2, ASHA, and CODA.

Representative fixed settings include:

```text
gamma                  = 0.99
gradient clipping      = 0.5
minibatch size         = 512
SGD iterations         = 10
network                = [512, 512], tanh
policy/value networks  = separate
observation filter     = MeanStdFilter
```

See `docs/experimental_protocol.md` and `configs/ppo_config.py` for the complete final PPO protocol.

---

## Champion Selection

One champion is selected independently for each

```text
(method, learner, environment, training seed)
```

using training information only.

The selection protocol uses:

- the final causal training branch;
- a time-weighted mean return over up to the final 200k interactions;
- at least 100k interactions of terminal support;
- at least three finite terminal observations;
- at least 100k interactions after the last executed configuration change.

For ASHA, only trials reaching the full per-trial budget are eligible.

---

## Held-Out Evaluation

Each selected champion is evaluated using:

```text
100 held-out episodes
explore = False
reset seeds = 100000–100099
```

The same held-out reset seeds are used across methods for matched comparisons.

### Control-quality metrics

In addition to held-out return, the evaluation records:

- root orientation RMS deviation;
- root angular-velocity RMS;
- normalized action smoothness;
- normalized action effort;
- episode completion fraction;
- full-horizon survival;
- termination and truncation indicators;
- forward displacement;
- mean forward speed when available.

No composite stability score is constructed. Return remains the primary performance outcome.

---

## Statistical Analysis

Comparisons are conducted at the training-seed level.

The analysis includes:

- median and interquartile range;
- paired Wilcoxon signed-rank tests;
- Holm correction for multiple comparisons;
- paired effect sizes;
- average ranks across learner-environment blocks;
- sample-efficiency and training-trajectory analyses.

A non-significant result is not interpreted as evidence of equivalence.

---

## Results

Processed results are stored separately for PPO and SAC:

```text
results/
├── ppo/
│   ├── training/
│   ├── scheduler/
│   ├── heldout/
│   ├── statistics/
│   └── runtime/
└── sac/
    ├── training/
    ├── scheduler/
    ├── heldout/
    ├── statistics/
    └── runtime/
```

The objective is that all tables and figures in the associated manuscript can be regenerated from the processed data without repeating the full training campaign.

---

## Checkpoints

SAC checkpoints can be large because replay-buffer state is preserved for checkpoint inheritance.

Large checkpoints and raw Ray Tune directories should therefore not be stored directly in the standard Git repository. Final champion checkpoints can instead be archived using a research-data service such as Zenodo, OSF, or an institutional repository.

Add the final archive link here:

```text
Checkpoint archive: <DOI_OR_URL>
```

---

## Reproducing Evaluation

PPO:

```bash
python evaluation/heldout_test_ppo.py
```

SAC:

```bash
python evaluation/heldout_test_sac.py
```

The evaluation scripts:

1. validate training outputs;
2. select champions using training data only;
3. restore the selected checkpoint;
4. evaluate fixed held-out reset seeds with exploration disabled;
5. save episode-level results;
6. generate training-seed summaries;
7. generate method/environment summaries.

---

## Reproducing Figures and Tables

```bash
python analysis/training_curves.py
python analysis/heldout_statistics.py
python analysis/stability_analysis.py
python analysis/runtime_analysis.py
python analysis/generate_tables.py
```

Detailed reproduction instructions should be documented in:

```text
docs/reproducibility.md
```

---

## Experimental Metadata

Each experiment records metadata such as:

- method and learner;
- environment;
- training seed;
- population or trial count;
- hyperparameter search domain;
- fixed learner configuration;
- checkpointing settings;
- package versions;
- restore-time hyperparameter audits where applicable.

For PBT, PB2, and CODA, desired and effective hyperparameters are audited after checkpoint inheritance to detect restore-time configuration mismatches.

---

## Files Excluded from Git

Recommended `.gitignore` entries include:

```gitignore
# Ray
ray_results/
results/ray_tune_logs/

# Large checkpoints
results/champions/
checkpoints/**/*.pkl
checkpoints/**/*.pt
checkpoints/**/*.pth

# Python
__pycache__/
*.pyc
*.pyo

# Logs / temporary files
*.log
*.tmp

# Environments
.venv/
venv/

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db
```

---

## Citation

If you use CODA or this experimental framework, please cite the associated paper.

```bibtex
@article{coda2026,
  title   = {Closed-Loop Online Diagnostic-Aware AutoRL},
  author  = {<AUTHORS>},
  journal = {<JOURNAL>},
  year    = {2026}
}
```

A machine-readable citation file can also be provided in `CITATION.cff`.

---

## License

This project is distributed under the license specified in the `LICENSE` file.

---

## Contact

**<AUTHOR NAME>**  
<INSTITUTION>  
<EMAIL OR PERSONAL WEBSITE>
