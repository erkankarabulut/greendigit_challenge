"""
Peak detection experiments for Task A.2.

Experiments
-----------
1. Backend comparison: tabpfn-ts, tabpfn-ts-feat, xgb, rocket
   (context_length=None for all)

2. Context-length sweep: tabpfn-ts and tabpfn-ts-feat
   context_lengths=[96, 672, 2880, 5760, None]

3. Feature-group ablation: xgb backend, disable one group at a time
   groups: base, rolling, slot, cross, extremity

4. Raw vs enriched: xgb with only base features vs all 33 features

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
if str(TASK_A_SRC) not in sys.path:
    sys.path.insert(0, str(TASK_A_SRC))

from task_a.dataio import default_data_path, load_series_csv, split_temporal
from task_a.evaluation import load_peaks, score_peaks
from task_a.models.model import MyModel
from task_a.schemas import parse_timestamp
from task_a.submission import write_peaks

CLF_BACKENDS = ["tabpfn-ts", "tabpfn-ts-feat", "xgb", "rocket"]
CONTEXT_SWEEP_BACKENDS = ["tabpfn-ts", "tabpfn-ts-feat"]
CONTEXT_LENGTHS = [96, 672, 2880, 5760, None]
ABLATION_BACKEND = "tabpfn-ts"
FEATURE_GROUPS = ["base", "rolling", "slot", "cross", "extremity"]
XGB_RAW_DISABLED = ["rolling", "slot", "cross", "extremity"]  # only base features

CSV_FIELDS = [
    "experiment", "clf_backend", "context_length", "disabled_groups",
    "S_A2", "auroc_a2", "f1_a2", "elapsed_s",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=default_data_path())
    p.add_argument("--cutoff", default="2026-02-18T14:00:00+00:00")
    p.add_argument("--output", type=Path, default=ROOT / "task-a" / "outputs" / "exp_peak_detection.csv")
    p.add_argument("--tabpfn-mode", default="LOCAL", choices=["CLIENT", "LOCAL"])
    p.add_argument("--tmp-dir", type=Path, default=ROOT / "task-a" / "outputs" / "exp_tmp")
    return p.parse_args()


def _write_row(writer, row: dict) -> None:
    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_clf_config(
    clf_backend: str,
    context_length: int | None,
    disabled: list[str],
    train,
    test,
    tabpfn_mode: str,
    tmp_dir: Path,
    tag: str = "",
) -> dict:
    t0 = time.time()
    model = MyModel.fit(
        train,
        clf_backend=clf_backend,
        tabpfn_mode=tabpfn_mode,
        disabled_feature_groups=disabled,
    )
    peaks_path = tmp_dir / f"peaks_{clf_backend}_{context_length}_{tag}.csv"
    peaks = model.predict_peaks(test, max_context_length=context_length)
    write_peaks(peaks, peaks_path)
    preds = load_peaks(peaks_path)
    parts = score_peaks(train, test, predictions=preds)
    elapsed = time.time() - t0
    return {
        "S_A2": parts["S_A2"],
        "auroc_a2": parts["auroc_a2"],
        "f1_a2": parts["f1_a2"],
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = load_series_csv(args.input)
    train, test = split_temporal(rows, args.cutoff)
    print(f"Loaded {len(train)} train / {len(test)} test rows")

    write_header = not args.output.exists()
    fh = args.output.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    # --- Experiment 1: backend comparison ---
    for backend in CLF_BACKENDS:
        print(f"\n[backend-comparison] {backend}")
        try:
            result = run_clf_config(
                backend, None, [], train, test, args.tabpfn_mode, args.tmp_dir, "bc"
            )
            result.update({
                "experiment": "backend_comparison",
                "clf_backend": backend,
                "context_length": "None",
                "disabled_groups": "",
            })
            _write_row(writer, result)
            fh.flush()
            print(f"  S_A2={result['S_A2']:.4f}  f1={result['f1_a2']:.4f}  ({result['elapsed_s']}s)")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # --- Experiment 2: context-length sweep ---
    for backend in CONTEXT_SWEEP_BACKENDS:
        print(f"\n[context-sweep] training {backend} ...")
        t0 = time.time()
        model = MyModel.fit(train, clf_backend=backend, tabpfn_mode=args.tabpfn_mode)
        print(f"  trained in {time.time()-t0:.1f}s")

        for ctx in CONTEXT_LENGTHS:
            print(f"  context_length={ctx}")
            t1 = time.time()
            try:
                peaks_path = args.tmp_dir / f"peaks_ctx_{backend}_{ctx}.csv"
                peaks = model.predict_peaks(test, max_context_length=ctx)
                write_peaks(peaks, peaks_path)
                preds = load_peaks(peaks_path)
                parts = score_peaks(train, test, predictions=preds)
                elapsed = time.time() - t1
                row = {
                    "experiment": "context_sweep",
                    "clf_backend": backend,
                    "context_length": str(ctx),
                    "disabled_groups": "",
                    "S_A2": parts["S_A2"],
                    "auroc_a2": parts["auroc_a2"],
                    "f1_a2": parts["f1_a2"],
                    "elapsed_s": round(elapsed, 1),
                }
                _write_row(writer, row)
                fh.flush()
                print(f"    S_A2={parts['S_A2']:.4f}  ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"    ERROR: {exc}")

    # --- Experiment 3: feature-group ablation (xgb) ---
    print(f"\n[ablation] {ABLATION_BACKEND} — full model (no disabled groups)")
    try:
        result = run_clf_config(
            ABLATION_BACKEND, None, [], train, test, args.tabpfn_mode, args.tmp_dir, "abl_full"
        )
        result.update({
            "experiment": "ablation",
            "clf_backend": ABLATION_BACKEND,
            "context_length": "None",
            "disabled_groups": "",
        })
        _write_row(writer, result)
        fh.flush()
        print(f"  S_A2={result['S_A2']:.4f}  ({result['elapsed_s']}s)")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    for group in FEATURE_GROUPS:
        print(f"\n[ablation] {ABLATION_BACKEND} — disable={group}")
        try:
            result = run_clf_config(
                ABLATION_BACKEND, None, [group], train, test, args.tabpfn_mode, args.tmp_dir, f"abl_{group}"
            )
            result.update({
                "experiment": "ablation",
                "clf_backend": ABLATION_BACKEND,
                "context_length": "None",
                "disabled_groups": group,
            })
            _write_row(writer, result)
            fh.flush()
            print(f"  S_A2={result['S_A2']:.4f}  ({result['elapsed_s']}s)")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # --- Experiment 4: raw vs enriched (xgb) ---
    for variant, disabled in [("raw", XGB_RAW_DISABLED), ("enriched", [])]:
        print(f"\n[raw-vs-enriched] xgb-{variant}")
        try:
            result = run_clf_config(
                ABLATION_BACKEND, None, disabled, train, test,
                args.tabpfn_mode, args.tmp_dir, f"rve_{variant}"
            )
            result.update({
                "experiment": "raw_vs_enriched",
                "clf_backend": f"xgb-{variant}",
                "context_length": "None",
                "disabled_groups": ",".join(disabled),
            })
            _write_row(writer, result)
            fh.flush()
            print(f"  S_A2={result['S_A2']:.4f}  f1={result['f1_a2']:.4f}  ({result['elapsed_s']}s)")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    fh.close()
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
