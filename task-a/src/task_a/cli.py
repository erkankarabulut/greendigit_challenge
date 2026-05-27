from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataio import default_data_path, load_series_csv, split_temporal, write_series_csv
from .evaluation import (
    compose_task_a_score,
    load_detections,
    load_forecasts,
    load_peaks,
    score_detection,
    score_forecasts,
    score_peaks,
)
from .labels import peak_label, peak_threshold, valid_signal_label
from .models.baseline import BaselineForecaster
from .models.model import MyModel
from .schemas import DetectionRow, PeakRow, parse_timestamp
from .submission import (
    validate_detection_csv,
    validate_forecast_csv,
    validate_peak_csv,
    write_detections,
    write_forecasts,
    write_peaks,
)


DEFAULT_OUTPUTS = Path("outputs")


def cmd_prepare_data(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    write_series_csv(rows, args.output)
    print(f"wrote {len(rows)} canonical rows to {args.output}")


def cmd_train_baseline(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    train, _ = split_temporal(rows, args.cutoff)
    model = BaselineForecaster.fit(train)
    model.save(args.model)
    print(f"trained baseline on {len(train)} rows and wrote {args.model}")


def cmd_predict(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    series_ids = sorted({row.series_id for row in rows})
    origins = sorted({
        row.bucket_15m for row in rows
        if row.bucket_15m > parse_timestamp(args.cutoff)
    })
    model = BaselineForecaster.load(args.model)
    predictions = []
    for origin in origins:
        predictions.extend(model.predict(series_ids, origin, args.horizons))
    write_forecasts(predictions, args.output)
    print(f"wrote {len(predictions)} forecasts to {args.output}")


def cmd_predict_detection(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    _, test = split_temporal(rows, args.cutoff)
    predictions = [
        DetectionRow(
            row.series_id,
            row.bucket_15m,
            float(valid_signal_label(row)),
            valid_signal_label(row),
        )
        for row in test
    ]
    write_detections(predictions, args.output)
    print(f"wrote {len(predictions)} detection rows to {args.output}")


def cmd_predict_peaks(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    train, test = split_temporal(rows, args.cutoff)
    thresholds = peak_threshold(train, args.peak_quantile)
    predictions = []
    for row in test:
        threshold = max(1e-9, thresholds.get(row.series_id, 1e-9))
        score = min(1.0, max(row.energy_wh, row.cfp_g) / threshold)
        predictions.append(PeakRow(row.series_id, row.bucket_15m, score, peak_label(row, thresholds)))
    write_peaks(predictions, args.output)
    print(f"wrote {len(predictions)} peak rows to {args.output}")



def cmd_train_model(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    train, _ = split_temporal(rows, args.cutoff)
    disabled = [g.strip() for g in args.disabled_feature_groups.split(",") if g.strip()] \
        if args.disabled_feature_groups else []
    model = MyModel.fit(
        train,
        clf_backend=args.clf_backend,
        tabpfn_mode=args.tabpfn_mode,
        forecast_backend=args.forecast_backend,
        disabled_feature_groups=disabled,
    )
    model.save(args.model)
    print(f"trained MyModel (clf={args.clf_backend}, forecast={args.forecast_backend}, "
          f"disabled={disabled or 'none'}) on {len(train)} rows → {args.model}")


def cmd_predict_model(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    series_ids = sorted({row.series_id for row in rows})
    origins = sorted({
        row.bucket_15m for row in rows
        if row.bucket_15m > parse_timestamp(args.cutoff)
    })
    model = MyModel.load(args.model)
    predictions = model.predict_forecasts(
        series_ids, origins, tuple(args.horizons), args.tabpfn_mode, args.max_context_length
    )
    write_forecasts(predictions, args.output)
    print(f"wrote {len(predictions)} forecasts to {args.output}")


def cmd_predict_detection_model(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    _, test = split_temporal(rows, args.cutoff)
    model = MyModel.load(args.model)
    predictions = model.predict_detection(test, args.max_context_length)
    write_detections(predictions, args.output)
    print(f"wrote {len(predictions)} detection rows to {args.output}")


def cmd_predict_peaks_model(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    _, test = split_temporal(rows, args.cutoff)
    model = MyModel.load(args.model)
    predictions = model.predict_peaks(test, args.max_context_length)
    write_peaks(predictions, args.output)
    print(f"wrote {len(predictions)} peak rows to {args.output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    rows = load_series_csv(args.input)
    train, test = split_temporal(rows, args.cutoff)
    validate_forecast_csv(args.forecasts)
    predictions = load_forecasts(args.forecasts)
    detections = None
    peaks = None
    if args.detections:
        validate_detection_csv(args.detections)
        detections = load_detections(args.detections)
    if args.peaks:
        validate_peak_csv(args.peaks)
        peaks = load_peaks(args.peaks)
    parts = {}
    parts.update(score_forecasts(test, predictions, args.eps, require_complete=True))
    parts.update(score_detection(test, detections))
    parts.update(score_peaks(train, test, args.peak_quantile, peaks))
    parts["ScoreTA"] = compose_task_a_score(parts)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w") as fh:
        json.dump(parts, fh, indent=2)
        fh.write("\n")
    print(json.dumps(parts, indent=2))


def cmd_validate_submission(args: argparse.Namespace) -> None:
    count = validate_forecast_csv(args.forecasts)
    print(f"validated {count} forecast rows")


def cmd_validate_detection(args: argparse.Namespace) -> None:
    count = validate_detection_csv(args.detections)
    print(f"validated {count} detection rows")


def cmd_validate_peaks(args: argparse.Namespace) -> None:
    count = validate_peak_csv(args.peaks)
    print(f"validated {count} peak rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="task-a")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--input", type=Path, default=default_data_path())
    prepare.add_argument("--output", type=Path, default=Path("../data/task_a_series.csv"))
    prepare.set_defaults(func=cmd_prepare_data)

    train = sub.add_parser("train-baseline")
    train.add_argument("--input", type=Path, default=default_data_path())
    train.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    train.add_argument("--model", type=Path, default=DEFAULT_OUTPUTS / "baseline_model.json")
    train.set_defaults(func=cmd_train_baseline)

    predict = sub.add_parser("predict")
    predict.add_argument("--input", type=Path, default=default_data_path())
    predict.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    predict.add_argument("--model", type=Path, default=DEFAULT_OUTPUTS / "baseline_model.json")
    predict.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "forecast_submission.csv")
    predict.add_argument("--horizons", type=int, nargs="+", default=[4, 96])
    predict.set_defaults(func=cmd_predict)

    detection = sub.add_parser("predict-detection")
    detection.add_argument("--input", type=Path, default=default_data_path())
    detection.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    detection.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "detection_submission.csv")
    detection.set_defaults(func=cmd_predict_detection)

    peaks = sub.add_parser("predict-peaks")
    peaks.add_argument("--input", type=Path, default=default_data_path())
    peaks.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    peaks.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "peak_submission.csv")
    peaks.add_argument("--peak-quantile", type=float, default=0.95)
    peaks.set_defaults(func=cmd_predict_peaks)

    train_model = sub.add_parser("train-model")
    train_model.add_argument("--input", type=Path, default=default_data_path())
    train_model.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    train_model.add_argument("--model", type=Path, default=DEFAULT_OUTPUTS / "my_model.json")
    train_model.add_argument(
        "--clf-backend", default="tabpfn-ts",
        choices=["rf", "xgb", "tabpfn-ts", "tabpfn-ts-feat", "rocket"],
        help=(
            "rf: RandomForest + engineered features. "
            "xgb: XGBoost + engineered features. "
            "tabpfn-ts: TabPFN-TS out-of-the-box (default). "
            "tabpfn-ts-feat: TabPFN-TS with non-overlapping engineered features as covariates. "
            "rocket: ROCKET on raw 24h multivariate windows."
        ),
    )
    train_model.add_argument("--tabpfn-mode", default="LOCAL", choices=["CLIENT", "LOCAL"])
    train_model.add_argument(
        "--forecast-backend", default="tabpfn-ts",
        choices=["tabpfn-ts", "tabpfn-ts-feat", "nhits", "prophet"],
        help=(
            "tabpfn-ts: TabPFN-TS out-of-the-box (default). "
            "tabpfn-ts-feat: TabPFN-TS with slot statistics as known future covariates. "
            "nhits: N-HiTS neural forecaster. "
            "prophet: Facebook Prophet per series."
        ),
    )
    train_model.add_argument(
        "--disabled-feature-groups", default="",
        metavar="GROUPS",
        help=(
            "Comma-separated feature groups to disable for rf/xgb/tabpfn-ts-feat clf backends. "
            "Choices: base,rolling,slot,cross,extremity. "
            "Example: --disabled-feature-groups rolling,cross"
        ),
    )
    train_model.set_defaults(func=cmd_train_model)

    predict_model = sub.add_parser("predict-model")
    predict_model.add_argument("--input", type=Path, default=default_data_path())
    predict_model.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    predict_model.add_argument("--model", type=Path, default=DEFAULT_OUTPUTS / "my_model.json")
    predict_model.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "forecast_submission.csv")
    predict_model.add_argument("--horizons", type=int, nargs="+", default=[4, 96])
    predict_model.add_argument("--tabpfn-mode", default="LOCAL", choices=["CLIENT", "LOCAL"])
    predict_model.add_argument(
        "--max-context-length", type=int, default=None,
        help="History rows per series passed to TabPFN. None = full history (GPU). Set ≤1000 on CPU."
    )
    predict_model.set_defaults(func=cmd_predict_model)

    predict_detection_model = sub.add_parser("predict-detection-model")
    predict_detection_model.add_argument("--input", type=Path, default=default_data_path())
    predict_detection_model.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    predict_detection_model.add_argument("--model", type=Path, default=DEFAULT_OUTPUTS / "my_model.json")
    predict_detection_model.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "detection_submission.csv")
    predict_detection_model.add_argument(
        "--max-context-length", type=int, default=None,
        help="Limit training labels per series passed to TabPFN-TS. None = full history."
    )
    predict_detection_model.set_defaults(func=cmd_predict_detection_model)

    predict_peaks_model = sub.add_parser("predict-peaks-model")
    predict_peaks_model.add_argument("--input", type=Path, default=default_data_path())
    predict_peaks_model.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    predict_peaks_model.add_argument("--model", type=Path, default=DEFAULT_OUTPUTS / "my_model.json")
    predict_peaks_model.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "peak_submission.csv")
    predict_peaks_model.add_argument(
        "--max-context-length", type=int, default=None,
        help="Limit training labels per series passed to TabPFN-TS. None = full history."
    )
    predict_peaks_model.set_defaults(func=cmd_predict_peaks_model)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--input", type=Path, default=default_data_path())
    evaluate.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    evaluate.add_argument("--forecasts", type=Path, default=DEFAULT_OUTPUTS / "forecast_submission.csv")
    evaluate.add_argument("--detections", type=Path)
    evaluate.add_argument("--peaks", type=Path)
    evaluate.add_argument("--output", type=Path, default=DEFAULT_OUTPUTS / "metrics.json")
    evaluate.add_argument("--eps", type=float, default=1e-9)
    evaluate.add_argument("--peak-quantile", type=float, default=0.95)
    evaluate.set_defaults(func=cmd_evaluate)

    validate = sub.add_parser("validate-submission")
    validate.add_argument("--forecasts", type=Path, required=True)
    validate.set_defaults(func=cmd_validate_submission)

    validate_detection = sub.add_parser("validate-detection")
    validate_detection.add_argument("--detections", type=Path, required=True)
    validate_detection.set_defaults(func=cmd_validate_detection)

    validate_peaks = sub.add_parser("validate-peaks")
    validate_peaks.add_argument("--peaks", type=Path, required=True)
    validate_peaks.set_defaults(func=cmd_validate_peaks)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
