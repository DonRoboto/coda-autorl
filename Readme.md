# CODA: Closed-Loop Online Diagnostic-Aware AutoRL

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray_Tune-028CF0.svg?style=flat&logo=ray&logoColor=white)](https://docs.ray.io/en/latest/tune/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Official implementation of the framework **Closed-Loop Online Diagnostic-Aware AutoRL (CODA)**.

CODA is a population-based meta-optimization framework that extends the Population-Based Bandits (PB2) architecture by establishing a bidirectional communication interface between the population-level optimizer and the evolving inner reinforcement learning (RL) learner.

## 📖 Overview

Standard population-based AutoRL methods often treat the inner learner as a black box, optimizing hyperparameters based solely on interval-level performance. CODA introduces two complementary channels:

1. **Inner-to-Outer (I2O) Communication:** Incorporates learner-state diagnostics (derived from policy KL divergence and value-function explained variance) as pre-decision contextual information for a Time-Varying Gaussian Process (TV-GP) surrogate.
2. **Outer-to-Inner (O2I) Communication:** Measures the normalized multidimensional magnitude of the hyperparameter intervention (HIM) and communicates this back to the inner learner by modulating the Proximal Policy Optimization (PPO) entropy actuator.

## 📂 Repository Structure

The codebase is organized to strictly separate the framework's core logic from the experimental execution and statistical evaluation.

```text
coda-autorl/
├── src/coda/               # Core CODA framework (TV-GP, Lineage Bookkeeping, Normalizers)
├── baselines/              # Wrappers for PBT, PB2, and ASHA baselines
├── experiments/            # Execution scripts and YAML configurations for MuJoCo
├── analysis/               # Statistical evaluation (Wilcoxon tests, Holm correction)
└── results/                # Output directories for Ray Tune logs and metrics
