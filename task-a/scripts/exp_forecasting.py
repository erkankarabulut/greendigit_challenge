"""
Forecasting experiments for Task A.

Experiments
-----------
1. Backend comparison: tabpfn-ts, tabpfn-ts-feat, nhits, prophet
   (context_length=None for all)

2. Context-length sweep: tabpfn-ts and tabpfn-ts-feat
   context_lengths=[96, 336, 672, 2688, None]

Results are appended to --output CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_A_SRC = ROOT / "task-a" / "src"
TASK_B_SRC = ROOT / "task-b"
if str(TASK_A_SRC) not in sys.path:
    sys.path.insert(0, str(TASK_A_SRC))

from task_a.dataio import default_data_path, load_series_csv, split_temporal
from task_a.evaluation import (
    compose_task_a_score,
    load_forecasts,
    score_forecasts,
    score_detection,
    score_peaks,
)
from task_a.models.model import MyModel
from task_a.schemas import parse_timestamp
from task_a.submission import write_forecasts, write_detections, write_peaks

BACKENDS = ["tabpfn-ts", "tabpfn-ts-feat", "nhits", "prophet"]
CONTEXT_SWEEP_BACKENDS = ["tabpfn-ts", "tabpfn-ts-feat"]
CONTEXT_LENGTHS = [96, 672, 2880, 5760, None]
HORIZONS = (4, 96)

CSV_FIELDS = [
    "experiment", "backend", "context_length",
    "S_A_4", "smape_energy_4", "smape_cfp_4",
    "S_A_96", "smape_energy_96", "smape_cfp_96",
    "ScoreTA", "elapsed_s",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=default_data_path())
    p.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    p.add_argument("--output", type=Path, default=ROOT / "task-a" / "outputs" / "exp_forecasting.csv")
    p.add_argument("--tabpfn-mode", default="LOCAL", choices=["CLIENT", "LOCAL"])
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "task-a" / "outputs" / "exp_tmp")
    return p.parse_args()


def _write_row(writer, row: dict) -> None:
    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_forecast_config(
        backend: str,
        context_length: int | None,
        train,
        test,
        series_ids,
        test_origins,
        tabpfn_mode: str,
        tmp_dir: Path,
) -> dict:
    t0 = time.time()
    model = MyModel.fit(train, forecast_backend=backend, tabpfn_mode=tabpfn_mode)
    forecast_path = tmp_dir / f"fc_{backend}_{context_length}.csv"
    forecasts = model.predict_forecasts(
        series_ids, test_origins, HORIZONS, tabpfn_mode, context_length
    )
    write_forecasts(forecasts, forecast_path)
    preds = load_forecasts(forecast_path)
    parts = {}
    parts.update(score_forecasts(test, preds, require_complete=True))
    parts["ScoreTA"] = compose_task_a_score(parts)
    elapsed = time.time() - t0
    return {
        "S_A_4": parts.get("S_A_4", ""),
        "smape_energy_4": parts.get("smape_energy_4", ""),
        "smape_cfp_4": parts.get("smape_cfp_4", ""),
        "S_A_96": parts.get("S_A_96", ""),
        "smape_energy_96": parts.get("smape_energy_96", ""),
        "smape_cfp_96": parts.get("smape_cfp_96", ""),
        "ScoreTA": parts.get("ScoreTA", ""),
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = load_series_csv(args.input)
    train, test = split_temporal(rows, args.cutoff)
    series_ids = sorted({r.series_id for r in rows})
    test_origins = sorted({
        r.bucket_15m for r in rows if r.bucket_15m > parse_timestamp(args.cutoff)
    })
    print(f"Loaded {len(train)} train / {len(test)} test rows, {len(series_ids)} series")

    write_header = not args.output.exists()
    fh = args.output.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    # --- Experiment 1: backend comparison (default context length) ---
    for backend in BACKENDS:
        print(f"\n[backend-comparison] {backend}  context_length=None")
        try:
            result = run_forecast_config(
                backend, None, train, test, series_ids, test_origins,
                args.tabpfn_mode, args.tmp_dir
            )
            result.update({"experiment": "backend_comparison", "backend": backend, "context_length": "None"})
            _write_row(writer, result)
            fh.flush()
            print(f"  ScoreTA={result['ScoreTA']:.4f}  ({result['elapsed_s']}s)")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # --- Experiment 2: context-length sweep ---
    for backend in CONTEXT_SWEEP_BACKENDS:
        # Train once, sweep context at inference
        print(f"\n[context-sweep] training {backend} ...")
        t0 = time.time()
        model = MyModel.fit(train, forecast_backend=backend, tabpfn_mode=args.tabpfn_mode)
        print(f"  trained in {time.time() - t0:.1f}s")

        for ctx in CONTEXT_LENGTHS:
            print(f"  context_length={ctx}")
            t1 = time.time()
            try:
                forecast_path = args.tmp_dir / f"fc_ctx_{backend}_{ctx}.csv"
                forecasts = model.predict_forecasts(
                    series_ids, test_origins, HORIZONS, args.tabpfn_mode, ctx
                )
                write_forecasts(forecasts, forecast_path)
                preds = load_forecasts(forecast_path)
                parts = {}
                parts.update(score_forecasts(test, preds, require_complete=True))
                parts["ScoreTA"] = compose_task_a_score(parts)
                elapsed = time.time() - t1
                row = {
                    "experiment": "context_sweep",
                    "backend": backend,
                    "context_length": str(ctx),
                    "S_A_4": parts.get("S_A_4", ""),
                    "smape_energy_4": parts.get("smape_energy_4", ""),
                    "smape_cfp_4": parts.get("smape_cfp_4", ""),
                    "S_A_96": parts.get("S_A_96", ""),
                    "smape_energy_96": parts.get("smape_energy_96", ""),
                    "smape_cfp_96": parts.get("smape_cfp_96", ""),
                    "ScoreTA": parts.get("ScoreTA", ""),
                    "elapsed_s": round(elapsed, 1),
                }
                _write_row(writer, row)
                fh.flush()
                print(f"    ScoreTA={parts['ScoreTA']:.4f}  ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"    ERROR: {exc}")

    fh.close()
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
