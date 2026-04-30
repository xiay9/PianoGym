# RhythmGym / PianoGym

RhythmGym is a simulation codebase for fatigue-constrained piano rhythm training. The repository contains the PianoGym environment, the PianoMPC controller, comparison baselines, revision experiments, and scripts used to regenerate the result tables and figures for our manuscript.

The experiments are simulation-only. No human-subject data, private data, or external datasets are required.

## PLOS ONE Reproducibility Notes

This repository is organized to support the PLOS requirements for code, software, and data sharing:

- Author-generated code needed to reproduce the manuscript findings is included in `src/`, `experiments/`, and `scripts/`.
- Dependencies are listed in `requirements.txt`.
- Generated result datasets are written under `output/data/`.
- Generated manuscript figures are written under `output/paper1/figures/` and synchronized to `paper/figures/`.
- Generated LaTeX tables are written to `output/paper1/tabel.tex`.
- The simulation outputs are generated from fixed seeds where deterministic reruns are required.
- The code runs on CPU only; no GPU, CUDA, or proprietary software is required.

For the final public release, include the generated `output/data/` files or archive them together with the source code so that readers can regenerate all reported figures and tables. PLOS guidance expects code and data needed to replicate findings to be available without access restrictions at publication.

PLOS policy references:

- Data and open-science policy: https://plos.org/open-science-policies/
- Materials, software, and code sharing: https://journals.plos.org/plosone/s/materials-software-and-code-sharing

## Repository Structure

```text
RhythmGym/
+-- configs.py                  # Global simulator parameters
+-- src/
|   +-- env.py                  # PianoGym environment
|   +-- agents.py               # PianoMPC and baseline agents
|   +-- safety.py               # Environment-side safety guard
|   +-- metrics.py              # Evaluation and safety metrics
|   +-- data_logging.py         # Trajectory logging utilities
|   +-- ope.py                  # Offline policy evaluation utilities
|   +-- suite.py                # Task-suite construction
|   +-- utils.py                # Shared utilities
|   +-- __init__.py
+-- experiments/
|   +-- compare.py              # Main algorithm comparison
|   +-- pianoMPC_horizon.py     # Planning-horizon sweep
|   +-- safety_ablation.py      # Safety-module ablation
|   +-- safety_consistency.py   # Guard diagnostic analysis
|   +-- suite_robustness.py     # Task-suite robustness
|   +-- dynamics_mismatch.py    # Dynamics mismatch analysis
|   +-- scoped_mismatch.py      # Scoped simulator perturbations
|   +-- threshold_window_robustness.py
|   +-- guard_dependence.py     # Shared/weaker/no-wrapper analysis
|   +-- guard_sensitivity.py    # Guard-margin sensitivity
+-- scripts/
|   +-- generate_tables.py      # Generate manuscript tables
|   +-- plot_figures.py         # Generate manuscript figures
+-- paper/figures/              # Figure files used by the LaTeX manuscript
+-- output/data/                # Generated result datasets
+-- output/paper1/figures/      # Generated figure files
```

## Installation

Use Python 3.10 or newer. Python 3.12 was used for the revision experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Required packages:

```text
numpy
pandas
matplotlib
scipy
```

## Quick Smoke Test

Run a short deterministic subset to verify that the environment and plotting pipeline work:

```bash
python experiments/pianoMPC_horizon.py --num-runs 2
python experiments/safety_ablation.py --num-runs 2
python scripts/plot_figures.py
```

Expected outputs include:

```text
output/data/pianoMPC_horizon.json
output/data/safety_ablation.json
output/paper1/figures/e3_pianoMPC_horizon.png
output/paper1/figures/e4_safety_tradeoff.png
```

## Reproducing the Main Results

From the repository root:

```bash
python experiments/compare.py --num-runs 10
python experiments/pianoMPC_horizon.py --num-runs 10
python experiments/safety_ablation.py --num-runs 10
python experiments/safety_consistency.py
python experiments/suite_robustness.py --runs 10
python experiments/dynamics_mismatch.py
python experiments/scoped_mismatch.py --num-runs 10
python experiments/threshold_window_robustness.py
python scripts/generate_tables.py
python scripts/plot_figures.py
```

The convenience runner executes the main experiment suite and post-processing:

```bash
python run_all_experiments.py
```

Revision-specific guard analyses can be reproduced with:

```bash
python experiments/guard_dependence.py --num-runs 10
python experiments/guard_sensitivity.py --num-runs 10
```

## Figure and Table Outputs

`scripts/plot_figures.py` writes manuscript-ready figures to `output/paper1/figures/` and synchronizes the same files to `paper/figures/`, which is the directory read by the LaTeX manuscript.

```text
output/paper1/figures/e1_comparison.png
output/paper1/figures/e2_learning_curves.png
output/paper1/figures/e3_pianoMPC_horizon.png
output/paper1/figures/e4_safety_tradeoff.png
output/paper1/figures/e5_safety_consistency.png
output/paper1/figures/e6_dynamics_mismatch.png
output/paper1/figures/e7_threshold_window.png
```

`scripts/generate_tables.py` writes:

```text
output/paper1/tabel.tex
```

## Result Dataset Files

Important generated dataset files include:

```text
output/data/comparison_data.json
output/data/comparison_profiles.json
output/data/pianoMPC_horizon.json
output/data/safety_ablation.json
output/data/p1_safety/safety_consistency_summary.csv
output/data/p1_suite/e5_suite_summary.csv
output/data/p1_misspec/dynamics_mismatch_summary.csv
output/data/p1_mismatch/scoped_mismatch_summary.csv
output/data/p1_stability/e7_robustness_summary.csv
output/data/p1_guard/guard_dependence_summary.csv
output/data/p1_guard/guard_sensitivity_summary.csv
```

`output/data/p1_candidate/` contains exploratory candidate runs and is not required for reproducing the manuscript figures.

## Metrics

Primary evaluation metrics are implemented in `src/metrics.py`.

- `TTM`: time-to-mastery in steps.
- `FeasibleRate`: `1 - OverloadRate`.
- `OverloadRate`: fraction of recorded steps where fatigue exceeds the current fatigue threshold.
- `AvgViolation`: mean `max(0, fatigue - fatigue_threshold)`.
- Guard replacement and false-negative rates are diagnostic quantities for guard behavior, not real-world safety guarantees.

Fatigue feasibility is evaluated under the simulator's surrogate fatigue model. The repository does not claim physiological validation in real piano learners.

## License

This project is released under the MIT License. See `LICENSE`.

## Citation

If you use this repository, please cite the associated PLOS ONE manuscript once published.
