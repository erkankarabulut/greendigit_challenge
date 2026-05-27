# GreenForecast: Digital Infrastructure Sustainability Forecasting with Tabular Foundation Models

---

This repository contains our source code for the [GreenDIGIT](https://gd2.lab.uvalight.net/) challenge
at ECML PKDD 2026.

The source code is an addition of our own models and experiments to the original challenge repository:
https://github.com/GreenDIGIT-project/greendigit-ecml-pkdd-2026-challenge.

---

## Submission INDElab

### Participants

| Name            | Email              | Affiliation             |
|-----------------|--------------------|-------------------------|
| Zeyu Zhang      | z.zhang2@uva.nl    | University of Amsterdam |
| Erkan Karabulut | e.karabulut@uva.nl | University of Amsterdam |

### Reproducibility

This section provides the commands to reproduce our experiments, the default is the best performing method,
GreenForecast, which is built on top of TabPFN-TS.

#### Training

Provide the command used to train the model using the provided public training/development data.
Example:
python train.py --data path/to/public/training/data --output path/to/model

#### Testing / Inference

Provide the command used to run the trained model on a test dataset.
The testing command must allow the organising team to specify the path to the private test dataset used for final
evaluation.
Example:
python test.py --model path/to/model --test-data path/to/private/test/data --output path/to/predictions

---

## Methods

We run several methods for each task.

**Task A — Forecasting energy usage and carbon footprint:**
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/) — zero-shot tabular foundation model for time series, used out-of-the-box (`--forecast-backend tabpfn-ts`)
- **GreenForecast** (our contribution) — TabPFN-TS augmented with slot statistics (median/MAD per hour × weekday) as known future covariates (`--forecast-backend tabpfn-ts-feat`)
- [Prophet](https://facebook.github.io/prophet/) — additive decomposition model with daily and weekly seasonality (`--forecast-backend prophet`)
- [N-HiTS](https://arxiv.org/abs/2201.12886) — neural hierarchical interpolation for time series forecasting via [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/) (`--forecast-backend nhits`)

**Task A.1 — Missing/Invalid Signal Detection & Task A.2 — Peak Detection:**
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/) — treats the binary label sequence as a time series (`--clf-backend tabpfn-ts`)
- **GreenForecast** (our contribution) — TabPFN-TS with our non-overlapping engineered features (rolling statistics, slot deviation, cross-series load share, zero streaks, extremity signals) injected as covariates (`--clf-backend tabpfn-ts-feat`)
- [XGBoost](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785) — gradient-boosted trees on the same 33 engineered features (`--clf-backend xgb`)
- [ROCKET](https://arxiv.org/abs/1910.13051) — random convolutional kernel transform on raw 24 h multivariate windows via [aeon](https://www.aeon-toolkit.org/) (`--clf-backend rocket`)

**Task B — Forecast-Driven Sustainable Job Scheduling:**
- **FCFS** — First-Come-First-Served, dispatches every ready job immediately to the site with the most available slots; no forecast use (`--scheduler fcfs`)
- **GreedyCarbon** — dispatches to the lowest-carbon site now, or defers up to 6 h if a ≥15% greener window is available within deadline slack (`--scheduler greedy_carbon`)
- **GreedyEnergy** — same logic as GreedyCarbon but optimises energy consumption instead of carbon (`--scheduler greedy_energy`)
- **MultiObjective** — scores every (site, time) candidate in a 6 h lookahead window as a weighted sum of normalised energy, carbon, and dispatch delay; weights are driven by the declared primary objective (`--scheduler multi_objective`)
