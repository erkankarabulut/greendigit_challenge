"""
Full Task A pipeline: train → forecast → detect → peaks → evaluate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure task_a is importable when the script is run from the project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
TASK_A_SRC = ROOT / "task-a" / "src"
if str(TASK_A_SRC) not in sys.path:
    sys.path.insert(0, str(TASK_A_SRC))

from task_a.dataio import default_data_path, load_series_csv, split_temporal
from task_a.evaluation import (
    compose_task_a_score,
    load_detections,
    load_forecasts,
    load_peaks,
    score_detection,
    score_forecasts,
    score_peaks,
)
from task_a.models.my_model import MyModel
from task_a.schemas import parse_timestamp
from task_a.submission import (
    validate_detection_csv,
    validate_forecast_csv,
    validate_peak_csv,
    write_detections,
    write_forecasts,
    write_peaks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task A end-to-end pipeline")
    p.add_argument("--input", type=Path, default=default_data_path())
    p.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    p.add_argument("--output-dir", type=Path, default=ROOT / "task-a" / "outputs")
    p.add_argument(
        "--clf-backend", default="tabpfn",
        choices=["rf", "tabpfn", "tabpfn-ts", "rocket"],
        help=(
            "rf: RandomForest + manual/rolling features. "
            "tabpfn: TabPFNClassifier, full training table as context. "
            "tabpfn-ts: TabPFN-TS treating binary labels as a time series."
        ),
    )
    p.add_argument("--tabpfn-mode", default="LOCAL", choices=["CLIENT", "LOCAL"])
    p.add_argument(
        "--forecast-backend", default="tabpfn-ts",
        choices=["tabpfn-ts", "nhits", "prophet"],
        help="Forecasting backend: tabpfn-ts (default), nhits, or prophet.",
    )
    p.add_argument(
        "--max-context-length", type=int, default=None,
        help="Rows of history per series for forecasting. None = full (GPU).",
    )
    p.add_argument("--horizons", type=int, nargs="+", default=[4, 96])
    return p.parse_args()


def _stamp(msg: str, t0: float) -> float:
    elapsed = time.time() - t0
    print(f"[{elapsed:6.1f}s] {msg}", flush=True)
    return time.time()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.output_dir / "my_model.json"
    forecast_path = args.output_dir / "forecast_submission.csv"
    detection_path = args.output_dir / "detection_submission.csv"
    peaks_path = args.output_dir / "peak_submission.csv"
    metrics_path = args.output_dir / "metrics.json"

    print("=" * 60)
    print(f"Task A pipeline  |  clf={args.clf_backend}  |  forecast={args.forecast_backend}  |  tabpfn_mode={args.tabpfn_mode}")
    print("=" * 60)
    t = time.time()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    rows = load_series_csv(args.input)
    train, test = split_temporal(rows, args.cutoff)
    series_ids = sorted({r.series_id for r in rows})
    test_origins = sorted({r.bucket_15m for r in rows if r.bucket_15m > parse_timestamp(args.cutoff)})
    t = _stamp(f"loaded {len(train)} train / {len(test)} test rows across {len(series_ids)} series", t)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = MyModel.fit(
        train,
        clf_backend=args.clf_backend,
        tabpfn_mode=args.tabpfn_mode,
        forecast_backend=args.forecast_backend,
    )
    model.save(model_path)
    t = _stamp(f"trained MyModel ({args.clf_backend}) → {model_path}", t)

    # ------------------------------------------------------------------
    # Forecast  (TabPFN-TS)
    # ------------------------------------------------------------------
    forecasts = model.predict_forecasts(
        series_ids=series_ids,
        test_origins=test_origins,
        horizons=tuple(args.horizons),
        max_context_length=args.max_context_length,
    )
    write_forecasts(forecasts, forecast_path)
    t = _stamp(f"forecast: {len(forecasts)} rows → {forecast_path}", t)

    # ------------------------------------------------------------------
    # A.1 — valid-signal detection
    # ------------------------------------------------------------------
    detections = model.predict_detection(test)
    write_detections(detections, detection_path)
    t = _stamp(f"detection: {len(detections)} rows → {detection_path}", t)

    # ------------------------------------------------------------------
    # A.2 — peak-event detection
    # ------------------------------------------------------------------
    peaks = model.predict_peaks(test)
    write_peaks(peaks, peaks_path)
    t = _stamp(f"peaks: {len(peaks)} rows → {peaks_path}", t)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    validate_forecast_csv(forecast_path)
    validate_detection_csv(detection_path)
    validate_peak_csv(peaks_path)
    t = _stamp("all submission files validated", t)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    forecast_preds = load_forecasts(forecast_path)
    detection_preds = load_detections(detection_path)
    peak_preds = load_peaks(peaks_path)

    parts: dict = {}
    parts.update(score_forecasts(test, forecast_preds, require_complete=True))
    parts.update(score_detection(test, detection_preds))
    parts.update(score_peaks(train, test, predictions=peak_preds))
    parts["ScoreTA"] = compose_task_a_score(parts)

    with metrics_path.open("w") as fh:
        json.dump(parts, fh, indent=2)
        fh.write("\n")

    _stamp(f"metrics saved → {metrics_path}", t)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    for k, v in parts.items():
        print(f"  {k:<25} {v:.6f}")
    print("=" * 60)
    print(f"  ScoreTA (lower=better): {parts['ScoreTA']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
