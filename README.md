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

We evaluated TabPFN-TS, TabPFN-TS with engineered features (tabpfn-ts-feat), XGBoost, N-HiTS, and Prophet across all Task A subtasks. XGBoost achieved the best scores for signal and peak detection; tabpfn-ts-feat achieved the best scores for forecasting.

#### Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU build; omit for CPU
pip install -e ./task-a -e ./task-b
```

#### Training

The pipeline trains and generates all submission files in one step. Pass the full dataset (train + test rows); `--cutoff` separates training data from the prediction period.

```bash
python task-a/scripts/run_pipeline.py \
  --input path/to/data \
  --cutoff 2026-02-18T14:00:00+00:00 \
  --clf-backend xgb \
  --forecast-backend tabpfn-ts-feat
```

#### Testing / Inference

The same script handles inference. Provide the full dataset path (training rows + private test rows); the `--cutoff` splits them automatically:

```bash
python task-a/scripts/run_pipeline.py \
  --input path/to/private/test/data \
  --cutoff 2026-02-18T14:00:00+00:00 \
  --clf-backend xgb \
  --forecast-backend tabpfn-ts-feat
```

Submission files are written to `task-a/outputs/`:
- `forecast_submission.csv`
- `detection_submission.csv`
- `peak_submission.csv`

**Task B — Scheduling (uses Task A forecasts):**
```bash
python task-b/examples/exp_scheduling.py \
  --jobs data/job_trace.csv \
  --sites data/site_config.json \
  --forecast-csv task-a/outputs/forecast_submission.csv \
  --start 2025-11-19T23:00:00 \
  --end 2026-03-12T17:00:00 \
  --output task-b/output/exp_scheduling.csv
```

Results are written to `task-b/output/exp_scheduling.csv`. Each row is one (scheduler × objective) combination scored against the FCFS baseline. Our primary submission uses the `multi_objective` scheduler; look for rows where `scheduler=multi_objective` for our best results.

For a quick smoke test (first 100 jobs, matches default simulation start):
```bash
python task-b/examples/exp_scheduling.py \
  --jobs data/job_trace.csv \
  --sites data/site_config.json \
  --forecast-csv task-a/outputs/forecast_submission.csv \
  --max-jobs 100 \
  --output task-b/output/exp_scheduling_test.csv
```

---

## Methods

We experiment with several methods for each task.

**Task A — Forecasting energy usage and carbon footprint:**
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/) — zero-shot tabular foundation model for time series, used out-of-the-box (`--forecast-backend tabpfn-ts`)
- **GreenForecast** (our contribution) — TabPFN-TS augmented with slot statistics (median/MAD per hour × weekday) as known future covariates (`--forecast-backend tabpfn-ts-feat`)
- [Prophet](https://facebook.github.io/prophet/) — additive decomposition model with daily and weekly seasonality (`--forecast-backend prophet`)
- [N-HiTS](https://arxiv.org/abs/2201.12886) — neural hierarchical interpolation for time series forecasting via [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/) (`--forecast-backend nhits`)

**Task A.1 — Missing/Invalid Signal Detection & Task A.2 — Peak Detection:**
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/) — treats the binary label sequence as a time series (`--clf-backend tabpfn-ts`)
- **GreenForecast** (our contribution) — TabPFN-TS with our non-overlapping engineered features (rolling statistics, slot deviation, cross-series load share, zero streaks, extremity signals) injected as covariates (`--clf-backend tabpfn-ts-feat`)
- [XGBoost](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785) — gradient-boosted trees on the same 33 engineered features (`--clf-backend xgb`)

**Task B — Forecast-Driven Sustainable Job Scheduling:**
- **FCFS** — First-Come-First-Served, dispatches every ready job immediately to the site with the most available slots; no forecast use
- **GreedyCarbon** — dispatches to the lowest-carbon site now, or defers up to 6 h if a ≥15% greener window is available within deadline slack
- **GreedyEnergy** — same logic as GreedyCarbon but optimises energy consumption instead of carbon
- **MultiObjective** *(primary submission)* — scores every (site, time) candidate in a 6 h lookahead window as a weighted sum of normalised energy, carbon, and dispatch delay; weights are driven by the declared primary objective
- **TemporalCarbon** — scans `site.get_carbon(t)` in 15-min steps over a 6 h window and defers if a slot with ≥15% lower carbon intensity is available within the job's deadline slack
