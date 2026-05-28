# GreenForecast: Digital Infrastructure Sustainability Forecasting with Tabular Foundation Models

This repository contains our source code for the [GreenDIGIT](https://gd2.lab.uvalight.net/) challenge
at ECML PKDD 2026.

The source code extends the original challenge repository:
https://github.com/GreenDIGIT-project/greendigit-ecml-pkdd-2026-challenge

---

## Table of Contents

- [Submission INDElab](#submission-indelab)
  - [Participants](#participants)
  - [Reproducibility](#reproducibility)
    - [Installation](#installation)
    - [Training](#training)
    - [Testing / Inference](#testing--inference)
    - [Cluster / HPC Execution (Optional)](#cluster--hpc-execution-optional)
- [Methods](#methods)

---

## Submission INDElab

### Participants

| Name            | Email              | Affiliation             |
|-----------------|--------------------|-------------------------|
| Zeyu Zhang      | z.zhang2@uva.nl    | University of Amsterdam |
| Erkan Karabulut | e.karabulut@uva.nl | University of Amsterdam |

### Reproducibility

#### Installation

```bash
pip install -r requirements.txt
```

> **TabPFN-TS requires accepting PriorLabs' terms of service.**
> On first run the library will prompt you interactively.
> Alternatively, obtain a token from <https://ux.priorlabs.ai/account> and
> set it before running: `export TABPFN_TOKEN=<your_token>`

#### Training

Train on the provided public data. `--cutoff` separates training rows from
the prediction period; rows at or before the cutoff are used for training.

**Task A:**

```bash
python task-a/scripts/run_pipeline.py \
  --input path/to/data \
  --cutoff 2026-02-18T14:00:00+00:00 \
  --clf-backend xgb \
  --forecast-backend tabpfn-ts-feat \
  --output-dir task-a/outputs
```

#### Testing / Inference

Pass the private test dataset (training rows + private test rows combined);
`--cutoff` splits them automatically. Submission files are written to
`--output-dir`.

**Task A:**

```bash
python task-a/scripts/run_pipeline.py \
  --input path/to/private/test/data \
  --cutoff 2026-02-18T14:00:00+00:00 \
  --clf-backend xgb \
  --forecast-backend tabpfn-ts-feat \
  --output-dir task-a/outputs
```

Outputs written to `task-a/outputs/`:
- `forecast_submission.csv`
- `detection_submission.csv`
- `peak_submission.csv`

**Task B** (uses the Task A forecast output from the step above):

```bash
python task-b/examples/exp_scheduling.py \
  --jobs data/job_trace.csv \
  --sites data/site_config.json \
  --forecast-csv task-a/outputs/forecast_submission.csv \
  --start 2026-02-18T14:00:00 \
  --end 2026-03-12T17:00:00 \
  --output task-b/output/exp_scheduling.csv
```

#### Cluster / HPC Execution (Optional)

For faster execution on a SLURM-managed HPC cluster, two helper scripts are
provided.

**`setup_environment.sh`** — creates the `greendigit` conda environment with
all dependencies, including CUDA-enabled PyTorch. Run this once on the cluster:

```bash
sbatch setup_environment.sh
```

**`run_experiment.sh`** — submits any experiment script as a SLURM job on a
GPU node. Pass the Python script path and any of its arguments directly:

```bash
sbatch run_experiment.sh task-a/scripts/run_pipeline.py \
  --input path/to/data \
  --cutoff 2026-02-18T14:00:00+00:00 \
  --clf-backend xgb \
  --forecast-backend tabpfn-ts-feat \
  --output-dir task-a/outputs
```

Logs are written to `exec_logs/slurm_<script>_<job_id>.out`.

---

## Methods

### Task A

**Forecasting (energy usage and carbon footprint):**
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/) — zero-shot tabular foundation model for time series (`--forecast-backend tabpfn-ts`)
- **GreenForecast** (our contribution) — TabPFN-TS augmented with slot statistics (median/MAD per hour × weekday) as known future covariates (`--forecast-backend tabpfn-ts-feat`)
- [Prophet](https://facebook.github.io/prophet/) — additive decomposition model with daily and weekly seasonality (`--forecast-backend prophet`)
- [N-HiTS](https://arxiv.org/abs/2201.12886) — neural hierarchical interpolation via [NeuralForecast](https://nixtlaverse.nixtla.io/neuralforecast/) (`--forecast-backend nhits`)

**A.1 — Missing/Invalid Signal Detection & A.2 — Peak Detection:**
- [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series/) — treats the binary label sequence as a time series (`--clf-backend tabpfn-ts`)
- **GreenForecast** (our contribution) — TabPFN-TS with engineered features (rolling statistics, slot deviation, cross-series load share, zero streaks, extremity signals) as covariates (`--clf-backend tabpfn-ts-feat`)
- [XGBoost](https://dl.acm.org/doi/pdf/10.1145/2939672.2939785) — gradient-boosted trees on the same engineered features (`--clf-backend xgb`)

### Task B

**Forecast-Driven Sustainable Job Scheduling:**
- **FCFS** — First-Come-First-Served organizer baseline; no forecast use
- **GreedyCarbon** — organizer baseline; dispatches to the lowest-carbon site, deferring up to 6 h if a ≥15% greener window is available
- **MultiObjective** *(primary submission)* — scores every (site, time) candidate in a 6 h lookahead window as a weighted sum of normalised energy, carbon, and dispatch delay; weights driven by the declared primary objective
- **TemporalCarbon** — scans the site model's rolling carbon signal at 15-min steps over a 6 h window and defers if a slot with ≥15% lower carbon intensity is available within the job's deadline slack