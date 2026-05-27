from __future__ import annotations

import bisect
import json
import math
import pickle
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from task_a.labels import peak_label, peak_threshold, valid_signal_label
from task_a.schemas import DetectionRow, ForecastRow, PeakRow, SeriesRow, parse_timestamp

_WINDOW_SIZES = (4, 24, 96)   # 1h, 6h, 24h in 15-min steps
_ROCKET_WINDOW = 96            # 24h of context for ROCKET (2 channels × 96 steps)
_MAX_CFP_RATIO = 10.0          # g/Wh cap for cfp/energy ratio (avoids sensor-fault extremes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _naive_ts(dt: Any) -> Any:
    """Convert any datetime-like to tz-naive pandas Timestamp in UTC."""
    import pandas as pd
    ts = pd.Timestamp(dt)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def _compute_slot_stats(
    series_data: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-series, keyed by 'hour_weekday' → median & MAD for energy and cfp."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for sid, buckets in series_data.items():
        slot_buckets: dict[str, list[tuple[float, float]]] = {}
        for iso_ts, (e, c) in buckets.items():
            ts = parse_timestamp(iso_ts)
            key = f"{ts.hour}_{ts.weekday()}"
            slot_buckets.setdefault(key, []).append((e, c))
        result[sid] = {}
        for key, vals in slot_buckets.items():
            energies = [v[0] for v in vals]
            cfps = [v[1] for v in vals]
            med_e = _median(energies)
            med_c = _median(cfps)
            result[sid][key] = {
                "med_e": med_e,
                "mad_e": _median([abs(x - med_e) for x in energies]) + 1e-9,
                "med_c": med_c,
                "mad_c": _median([abs(x - med_c) for x in cfps]) + 1e-9,
            }
    return result


def _build_cross_sums(rows: list[SeriesRow]) -> dict[str, tuple[float, float]]:
    """iso_ts → (sum_energy, sum_cfp) across all series at that timestamp."""
    sums: dict[str, tuple[float, float]] = {}
    for row in rows:
        iso_ts = row.bucket_15m.isoformat()
        prev_e, prev_c = sums.get(iso_ts, (0.0, 0.0))
        sums[iso_ts] = (prev_e + row.energy_wh, prev_c + row.cfp_g)
    return sums


def _build_zero_streaks(
    series_data: dict[str, dict[str, tuple[float, float]]],
    rows: list[SeriesRow],
) -> dict[tuple[str, str], int]:
    """Consecutive zero-energy count strictly before each (sid, ts)."""
    combined: dict[tuple[str, str], float] = {}
    for sid, buckets in series_data.items():
        for ts, (e, _) in buckets.items():
            combined[(sid, ts)] = e
    for row in rows:
        combined[(row.series_id, row.bucket_15m.isoformat())] = row.energy_wh

    per_series: dict[str, list[tuple[str, float]]] = {}
    for (sid, ts), e in combined.items():
        per_series.setdefault(sid, []).append((ts, e))

    result: dict[tuple[str, str], int] = {}
    for sid, entries in per_series.items():
        streak = 0
        for ts, e in sorted(entries):
            result[(sid, ts)] = streak
            streak = (streak + 1) if e == 0.0 else 0
    return result


def _build_sorted_histories(
    series_data: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, list[tuple[str, float, float]]]:
    return {
        sid: sorted((ts, e, c) for ts, (e, c) in buckets.items())
        for sid, buckets in series_data.items()
    }


# ---------------------------------------------------------------------------
# Feature groups — each returns a fixed-length list[float]
# ---------------------------------------------------------------------------

def _base_features(row: SeriesRow, series_stats: dict) -> list[float]:
    # 8 features
    stats = series_stats.get(row.series_id, {
        "mean_energy": 0.0, "std_energy": 1.0,
        "mean_cfp": 0.0, "std_cfp": 1.0,
    })
    z_energy = (row.energy_wh - stats["mean_energy"]) / stats["std_energy"]
    z_cfp = (row.cfp_g - stats["mean_cfp"]) / stats["std_cfp"]
    return [
        row.energy_wh, row.cfp_g, float(row.records),
        z_energy, z_cfp,
        float(row.energy_wh == 0.0), float(row.cfp_g == 0.0),
        float(row.records > 0 and row.energy_wh == 0.0),
    ]


def _rolling_features(
    row: SeriesRow,
    sorted_history: list[tuple[str, float, float]],
) -> list[float]:
    # 12 features — mean/std energy, mean cfp, zero-count per window; no leakage
    ts_iso = row.bucket_15m.isoformat()
    keys = [t for t, _, _ in sorted_history]
    cutoff = bisect.bisect_left(keys, ts_iso)
    feats: list[float] = []
    for w in _WINDOW_SIZES:
        start = max(0, cutoff - w)
        window = sorted_history[start:cutoff]
        if window:
            energies = [e for _, e, _ in window]
            cfps = [c for _, _, c in window]
            n = len(window)
            mean_e = sum(energies) / n
            std_e = (sum((x - mean_e) ** 2 for x in energies) / n) ** 0.5
            mean_c = sum(cfps) / n
            zero_count = float(sum(1 for e in energies if e == 0.0))
            feats.extend([mean_e, std_e, mean_c, zero_count])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0])
    return feats


def _slot_deviation_features(row: SeriesRow, slot_stats: dict) -> list[float]:
    # 2 features — how far this reading is from the typical value at this (hour, weekday)
    key = f"{row.bucket_15m.hour}_{row.bucket_15m.weekday()}"
    stats = slot_stats.get(row.series_id, {}).get(
        key, {"med_e": 0.0, "mad_e": 1e-9, "med_c": 0.0, "mad_c": 1e-9}
    )
    return [
        (row.energy_wh - stats["med_e"]) / stats["mad_e"],
        (row.cfp_g - stats["med_c"]) / stats["mad_c"],
    ]


def _cross_series_features(row: SeriesRow, cross_sums: dict) -> list[float]:
    # 4 features — total load at this timestamp and this series' share
    iso_ts = row.bucket_15m.isoformat()
    sum_e, sum_c = cross_sums.get(iso_ts, (max(row.energy_wh, 1e-9), max(row.cfp_g, 1e-9)))
    return [
        sum_e, sum_c,
        row.energy_wh / (sum_e + 1e-9),
        row.cfp_g / (sum_c + 1e-9),
    ]


def _rolling_pct_features(
    row: SeriesRow,
    sorted_history: list[tuple[str, float, float]],
) -> list[float]:
    # 4 features — percentile rank and distance to rolling max (last 96 steps)
    ts_iso = row.bucket_15m.isoformat()
    keys = [t for t, _, _ in sorted_history]
    cutoff = bisect.bisect_left(keys, ts_iso)
    start = max(0, cutoff - 96)
    window = sorted_history[start:cutoff]
    if not window:
        return [0.5, 0.5, 0.0, 0.0]
    energies = sorted(e for _, e, _ in window)
    cfps = sorted(c for _, _, c in window)
    n = len(window)
    e_pct = sum(1 for x in energies if x <= row.energy_wh) / n
    c_pct = sum(1 for x in cfps if x <= row.cfp_g) / n
    return [e_pct, c_pct, row.energy_wh / (energies[-1] + 1e-9), row.cfp_g / (cfps[-1] + 1e-9)]


_FEATURE_GROUPS = frozenset({"base", "rolling", "slot", "cross", "extremity"})


def _all_features(
    row: SeriesRow,
    series_stats: dict,
    sorted_histories: dict,
    slot_stats: dict,
    cross_sums: dict,
    zero_streaks: dict,
    disabled: frozenset = frozenset(),
) -> list[float]:
    sh = sorted_histories.get(row.series_id, [])
    iso_ts = row.bucket_15m.isoformat()
    result: list[float] = []
    if "base" not in disabled:
        result += _base_features(row, series_stats)                     # 8
    if "rolling" not in disabled:
        result += _rolling_features(row, sh)                            # 12
    if "slot" not in disabled:
        result += _slot_deviation_features(row, slot_stats)             # 2
    if "cross" not in disabled:
        result += _cross_series_features(row, cross_sums)               # 4
    if "extremity" not in disabled:
        result += [float(zero_streaks.get((row.series_id, iso_ts), 0))] # 1
        result += [row.energy_wh / (row.records + 1)]                   # 1
        result += _rolling_pct_features(row, sh)                        # 4
        result += [row.energy_wh * row.cfp_g]                           # 1
    return result


# ---------------------------------------------------------------------------
# ROCKET time-series input builder
# ---------------------------------------------------------------------------

def _build_rocket_input(
    rows: list[SeriesRow],
    sorted_histories: dict[str, list[tuple[str, float, float]]],
    window: int = _ROCKET_WINDOW,
) -> "np.ndarray":  # type: ignore[name-defined]
    """Return float32 array of shape (n_samples, 2, window) for aeon RocketClassifier."""
    import numpy as np
    X = []
    for row in rows:
        sh = sorted_histories.get(row.series_id, [])
        ts_iso = row.bucket_15m.isoformat()
        keys = [t for t, _, _ in sh]
        cutoff = bisect.bisect_left(keys, ts_iso)
        start = max(0, cutoff - window)
        segment = sh[start:cutoff]
        pad = window - len(segment)
        energy_ts = [0.0] * pad + [e for _, e, _ in segment]
        cfp_ts = [0.0] * pad + [c for _, _, c in segment]
        X.append([energy_ts, cfp_ts])
    return np.array(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# TabPFN-TS classification helper
# ---------------------------------------------------------------------------

def _tabpfn_ts_classify(
    context_labels: dict[str, dict[str, int]],
    test: list[SeriesRow],
    tabpfn_mode: str,
    max_context_length: int | None = None,
    threshold: float = 0.5,
) -> tuple[list[float], list[int]]:
    import pandas as pd
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

    mode = TabPFNMode.LOCAL if tabpfn_mode.upper() == "LOCAL" else TabPFNMode.CLIENT
    pipeline = TabPFNTSPipeline(tabpfn_mode=mode)

    context_rows = []
    for sid, ts_labels in context_labels.items():
        items = sorted(ts_labels.items())
        if max_context_length is not None:
            items = items[-max_context_length:]
        for iso_ts, label in items:
            context_rows.append({"item_id": sid, "timestamp": _naive_ts(parse_timestamp(iso_ts)), "target": float(label)})
    future_rows = [{"item_id": r.series_id, "timestamp": _naive_ts(r.bucket_15m)} for r in test]

    pred_df = pipeline.predict_df(
        context_df=pd.DataFrame(context_rows),
        future_df=pd.DataFrame(future_rows),
        quantiles=[0.5],
    )

    lookup: dict[tuple[str, str], float] = {}
    for idx, row in pred_df.iterrows():
        item_id, ts = idx if isinstance(idx, tuple) else (None, idx)
        ts_iso = pd.Timestamp(ts).tz_localize("UTC").isoformat()
        lookup[(str(item_id), ts_iso)] = float(row["target"])

    scores, preds = [], []
    for r in test:
        raw = lookup.get((r.series_id, r.bucket_15m.isoformat()), 0.5)
        score = max(0.0, min(1.0, raw))
        scores.append(score)
        preds.append(int(score >= threshold))
    return scores, preds


def _tabpfn_ts_classify_with_features(
    context_labels: dict[str, dict[str, int]],
    test: list[SeriesRow],
    tabpfn_mode: str,
    series_data: dict[str, dict[str, tuple[float, float]]],
    series_stats: dict,
    slot_stats: dict,
    max_context_length: int | None = None,
    disabled: frozenset = frozenset(),
    threshold: float = 0.5,
) -> tuple[list[float], list[int]]:
    """TabPFN-TS classification with our non-overlapping engineered features injected as covariates."""
    import pandas as pd
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

    mode = TabPFNMode.LOCAL if tabpfn_mode.upper() == "LOCAL" else TabPFNMode.CLIENT
    pipeline = TabPFNTSPipeline(tabpfn_mode=mode)

    sorted_histories = _build_sorted_histories(series_data)

    # Cross-sums for context rows (rebuilt from stored training data)
    cross_sums_ctx: dict[str, tuple[float, float]] = {}
    for sid, buckets in series_data.items():
        for iso_ts, (e, c) in buckets.items():
            prev_e, prev_c = cross_sums_ctx.get(iso_ts, (0.0, 0.0))
            cross_sums_ctx[iso_ts] = (prev_e + e, prev_c + c)

    zero_streaks = _build_zero_streaks(series_data, test)
    cross_sums_test = _build_cross_sums(test)

    context_rows: list[dict] = []
    for sid, ts_labels in context_labels.items():
        items = sorted(ts_labels.items())
        if max_context_length is not None:
            items = items[-max_context_length:]
        for iso_ts, label in items:
            ts = parse_timestamp(iso_ts)
            e, c = series_data.get(sid, {}).get(iso_ts, (0.0, 0.0))
            # records not stored in series_data; use 1.0 as neutral default
            row = SeriesRow(sid, ts, 1.0, e, c)
            feats = _all_features(row, series_stats, sorted_histories, slot_stats, cross_sums_ctx, zero_streaks, disabled)
            context_rows.append({
                "item_id": sid,
                "timestamp": _naive_ts(ts),
                "target": float(label),
                **{f"feat_{i}": v for i, v in enumerate(feats)},
            })

    future_rows: list[dict] = []
    for r in test:
        feats = _all_features(r, series_stats, sorted_histories, slot_stats, cross_sums_test, zero_streaks, disabled)
        future_rows.append({
            "item_id": r.series_id,
            "timestamp": _naive_ts(r.bucket_15m),
            **{f"feat_{i}": v for i, v in enumerate(feats)},
        })

    pred_df = pipeline.predict_df(
        context_df=pd.DataFrame(context_rows),
        future_df=pd.DataFrame(future_rows),
        quantiles=[0.5],
    )

    lookup: dict[tuple[str, str], float] = {}
    for idx, row in pred_df.iterrows():
        item_id, ts = idx if isinstance(idx, tuple) else (None, idx)
        ts_iso = pd.Timestamp(ts).tz_localize("UTC").isoformat()
        lookup[(str(item_id), ts_iso)] = float(row["target"])

    scores, preds = [], []
    for r in test:
        raw = lookup.get((r.series_id, r.bucket_15m.isoformat()), threshold)
        score = max(0.0, min(1.0, raw))
        scores.append(score)
        preds.append(int(score >= threshold))
    return scores, preds


# ---------------------------------------------------------------------------
# Forecasting backends (module-level for clarity)
# ---------------------------------------------------------------------------

def _context_df(
    series_data: dict[str, dict[str, tuple[float, float]]],
    series_ids: list[str],
    target_col: str,
    max_context_length: int | None,
) -> list[dict]:
    rows = []
    for sid in series_ids:
        items = sorted(series_data.get(sid, {}).items())
        if max_context_length is not None:
            items = items[-max_context_length:]
        for iso_ts, (energy, cfp) in items:
            rows.append({
                "sid": sid,
                "ts": _naive_ts(parse_timestamp(iso_ts)),
                "val": energy if target_col == "energy_wh" else cfp,
            })
    return rows


def _forecast_tabpfn_ts(
    series_data: dict[str, dict[str, tuple[float, float]]],
    series_ids: list[str],
    forecast_ts_list: list,
    tabpfn_mode: str,
    max_context_length: int | None,
) -> tuple[dict, dict]:
    """
    Step 1: Predict energy (log1p scale).
    Step 2: Predict cfp/energy ratio (carbon intensity signal, log1p scale).
    Step 3: cfp = energy_pred × ratio_pred.
    """
    import pandas as pd
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

    mode = TabPFNMode.LOCAL if tabpfn_mode.upper() == "LOCAL" else TabPFNMode.CLIENT
    pipeline = TabPFNTSPipeline(tabpfn_mode=mode)

    future_df = pd.DataFrame([
        {"item_id": sid, "timestamp": _naive_ts(ts)}
        for sid in series_ids
        for ts in forecast_ts_list
    ])

    def _run(context_rows_raw: list[dict]) -> dict:
        context_df = pd.DataFrame([
            {"item_id": r["sid"], "timestamp": r["ts"], "target": r["val"]}
            for r in context_rows_raw
        ])
        pred_df = pipeline.predict_df(
            context_df=context_df,
            future_df=future_df.copy(),
            quantiles=[0.5],
        )
        out: dict[tuple[str, str], float] = {}
        for idx, row in pred_df.iterrows():
            item_id, ts = idx if isinstance(idx, tuple) else (series_ids[0], idx)
            ts_iso = pd.Timestamp(ts).tz_localize("UTC").isoformat()
            out[(str(item_id), ts_iso)] = float(row["target"])
        return out

    # --- energy (log1p) ---
    energy_ctx = _context_df(series_data, series_ids, "energy_wh", max_context_length)
    for r in energy_ctx:
        r["val"] = math.log1p(r["val"])
    energy_raw = _run(energy_ctx)
    energy_lookup = {k: max(0.0, math.expm1(v)) for k, v in energy_raw.items()}

    # --- cfp/energy ratio (carbon intensity, log1p) ---
    ratio_ctx: list[dict] = []
    for sid in series_ids:
        items = sorted(series_data.get(sid, {}).items())
        if max_context_length is not None:
            items = items[-max_context_length:]
        for iso_ts, (energy, cfp) in items:
            ratio = min(cfp / max(energy, 1e-9), _MAX_CFP_RATIO)
            ratio_ctx.append({"sid": sid, "ts": _naive_ts(parse_timestamp(iso_ts)), "val": math.log1p(ratio)})
    ratio_raw = _run(ratio_ctx)

    cfp_lookup: dict[tuple[str, str], float] = {}
    for k, ratio_log in ratio_raw.items():
        ratio_pred = max(0.0, math.expm1(ratio_log))
        cfp_lookup[k] = ratio_pred * energy_lookup.get(k, 0.0)

    return energy_lookup, cfp_lookup


def _forecast_nhits(
    series_data: dict[str, dict[str, tuple[float, float]]],
    series_ids: list[str],
    forecast_ts_list: list,
    max_context_length: int | None,
    max_steps: int = 300,
) -> tuple[dict, dict]:
    import pandas as pd
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'neuralforecast' package is required for --forecast-backend nhits. "
            "Install it with: pip install neuralforecast"
        ) from exc

    # Compute h: steps from end of training to last forecast timestamp
    all_train_ts = [
        _naive_ts(parse_timestamp(iso_ts))
        for sid in series_ids
        for iso_ts in series_data.get(sid, {}).keys()
    ]
    if not all_train_ts:
        return {}, {}
    last_train_ts = max(all_train_ts)
    last_forecast_ts = max(_naive_ts(ts) for ts in forecast_ts_list)
    h = int((last_forecast_ts - last_train_ts) / pd.Timedelta("15min")) + 1

    # input_size: aim for 2 weeks, capped by available training rows
    n_train_per_series = len(list(series_data.get(series_ids[0], {}).items()))
    if max_context_length:
        n_train_per_series = min(n_train_per_series, max_context_length)
    input_size = min(96 * 14, n_train_per_series // 2)

    energy_lookup: dict[tuple[str, str], float] = {}
    cfp_lookup: dict[tuple[str, str], float] = {}

    for target_col, lookup in (("energy_wh", energy_lookup), ("cfp_g", cfp_lookup)):
        ctx = _context_df(series_data, series_ids, target_col, max_context_length)
        df = pd.DataFrame({
            "unique_id": [r["sid"] for r in ctx],
            "ds": [r["ts"] for r in ctx],
            "y": [r["val"] for r in ctx],
        })
        model = NHITS(h=h, input_size=input_size, max_steps=max_steps)
        nf = NeuralForecast(models=[model], freq="15min")
        nf.fit(df)
        pred_df = nf.predict().reset_index()

        # column name varies by version; pick whatever is not uid/ds/cutoff
        val_col = next(c for c in pred_df.columns if c not in ("unique_id", "ds", "cutoff"))
        for _, row in pred_df.iterrows():
            ts_iso = pd.Timestamp(row["ds"]).tz_localize("UTC").isoformat()
            lookup[(str(row["unique_id"]), ts_iso)] = max(0.0, float(row[val_col]))

    return energy_lookup, cfp_lookup


def _forecast_prophet(
    series_data: dict[str, dict[str, tuple[float, float]]],
    series_ids: list[str],
    forecast_ts_list: list,
    max_context_length: int | None,
) -> tuple[dict, dict]:
    import pandas as pd
    try:
        from prophet import Prophet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'prophet' package is required for --forecast-backend prophet. "
            "Install it with: pip install prophet"
        ) from exc

    future = pd.DataFrame({"ds": [_naive_ts(ts) for ts in forecast_ts_list]})

    energy_lookup: dict[tuple[str, str], float] = {}
    cfp_lookup: dict[tuple[str, str], float] = {}

    for target_col, lookup in (("energy_wh", energy_lookup), ("cfp_g", cfp_lookup)):
        for sid in series_ids:
            ctx = _context_df(series_data, [sid], target_col, max_context_length)
            train_df = pd.DataFrame({"ds": [r["ts"] for r in ctx], "y": [r["val"] for r in ctx]})

            m = Prophet(
                seasonality_mode="multiplicative",
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
            )
            m.fit(train_df)
            forecast = m.predict(future.copy())
            for _, row in forecast.iterrows():
                ts_iso = pd.Timestamp(row["ds"]).tz_localize("UTC").isoformat()
                lookup[(sid, ts_iso)] = max(0.0, float(row["yhat"]))

    return energy_lookup, cfp_lookup


def _forecast_tabpfn_ts_feat(
    series_data: dict[str, dict[str, tuple[float, float]]],
    series_ids: list[str],
    forecast_ts_list: list,
    tabpfn_mode: str,
    max_context_length: int | None,
    series_stats: dict,
    slot_stats: dict,
) -> tuple[dict, dict]:
    """TabPFN-TS forecasting with slot-statistics injected as known future covariates.

    Slot statistics (median / MAD for each hour × weekday cell) are the only
    features computable for future timestamps without knowing the target value,
    so they are the natural covariates to add here.  Context rows get the same
    four columns so TabPFN-TS can learn the association between the covariate
    and the target during the in-context training phase.
    """
    import pandas as pd
    from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

    mode = TabPFNMode.LOCAL if tabpfn_mode.upper() == "LOCAL" else TabPFNMode.CLIENT
    pipeline = TabPFNTSPipeline(tabpfn_mode=mode)

    def _slot_cov(ts: Any, sid: str) -> dict:
        h = ts.hour if hasattr(ts, "hour") else ts.hour
        dow = ts.dayofweek if hasattr(ts, "dayofweek") else ts.weekday()
        key = f"{h}_{dow}"
        s = slot_stats.get(sid, {}).get(key, {"med_e": 0.0, "mad_e": 1e-9, "med_c": 0.0, "mad_c": 1e-9})
        return {"slot_med_e": s["med_e"], "slot_mad_e": s["mad_e"],
                "slot_med_c": s["med_c"], "slot_mad_c": s["mad_c"]}

    future_df = pd.DataFrame([
        {"item_id": sid, "timestamp": _naive_ts(ts), **_slot_cov(_naive_ts(ts), sid)}
        for sid in series_ids
        for ts in forecast_ts_list
    ])

    def _run(context_rows_raw: list[dict]) -> dict:
        rows = []
        for r in context_rows_raw:
            naive_ts = r["ts"]
            rows.append({"item_id": r["sid"], "timestamp": naive_ts, "target": r["val"],
                         **_slot_cov(naive_ts, r["sid"])})
        context_df = pd.DataFrame(rows)
        pred_df = pipeline.predict_df(
            context_df=context_df,
            future_df=future_df.copy(),
            quantiles=[0.5],
        )
        out: dict[tuple[str, str], float] = {}
        for idx, row in pred_df.iterrows():
            item_id, ts = idx if isinstance(idx, tuple) else (series_ids[0], idx)
            ts_iso = pd.Timestamp(ts).tz_localize("UTC").isoformat()
            out[(str(item_id), ts_iso)] = float(row["target"])
        return out

    energy_ctx = _context_df(series_data, series_ids, "energy_wh", max_context_length)
    for r in energy_ctx:
        r["val"] = math.log1p(r["val"])
    energy_raw = _run(energy_ctx)
    energy_lookup = {k: max(0.0, math.expm1(v)) for k, v in energy_raw.items()}

    ratio_ctx: list[dict] = []
    for sid in series_ids:
        items = sorted(series_data.get(sid, {}).items())
        if max_context_length is not None:
            items = items[-max_context_length:]
        for iso_ts, (energy, cfp) in items:
            ratio = min(cfp / max(energy, 1e-9), _MAX_CFP_RATIO)
            ratio_ctx.append({"sid": sid, "ts": _naive_ts(parse_timestamp(iso_ts)), "val": math.log1p(ratio)})
    ratio_raw = _run(ratio_ctx)

    cfp_lookup: dict[tuple[str, str], float] = {}
    for k, ratio_log in ratio_raw.items():
        ratio_pred = max(0.0, math.expm1(ratio_log))
        cfp_lookup[k] = ratio_pred * energy_lookup.get(k, 0.0)

    return energy_lookup, cfp_lookup


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class MyModel:
    series_data: dict[str, dict[str, tuple[float, float]]]
    peak_thresholds: dict[str, float]
    global_mean_energy: float
    global_mean_cfp: float
    series_stats: dict[str, dict[str, float]]
    clf_backend: str          # rf | xgb | tabpfn-ts | tabpfn-ts-feat | rocket
    tabpfn_mode: str          # LOCAL | CLIENT
    forecast_backend: str = "tabpfn-ts"    # tabpfn-ts | tabpfn-ts-feat | nhits | prophet
    slot_stats: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    train_detection_labels: dict[str, dict[str, int]] = field(default_factory=dict)
    train_peak_labels: dict[str, dict[str, int]] = field(default_factory=dict)
    disabled_feature_groups: list[str] = field(default_factory=list)
    detection_clf: Any = field(default=None, repr=False)
    peak_clf: Any = field(default=None, repr=False)

    @classmethod
    def fit(
        cls,
        train: list[SeriesRow],
        clf_backend: str = "tabpfn-ts",
        tabpfn_mode: str = "LOCAL",
        forecast_backend: str = "tabpfn-ts",
        disabled_feature_groups: list[str] | None = None,
    ) -> "MyModel":
        """
        clf_backend:
          "rf"             — RandomForest with engineered features.
          "xgb"            — XGBoost with engineered features.
          "tabpfn-ts"      — TabPFN-TS out-of-the-box (its own feature engineering).
          "tabpfn-ts-feat" — TabPFN-TS with our non-overlapping features as covariates.
          "rocket"         — ROCKET on raw 24h multivariate time series windows.

        forecast_backend:
          "tabpfn-ts"      — TabPFN-TS with log1p + ratio approach.
          "tabpfn-ts-feat" — TabPFN-TS with slot statistics as known future covariates.
          "nhits"          — N-HiTS via neuralforecast (fit at predict time).
          "prophet"        — Prophet per series (fit at predict time).

        disabled_feature_groups:
          Subset of {"base","rolling","slot","cross","extremity"} to skip.
          Only affects rf, xgb, and tabpfn-ts-feat clf backends.
        """
        disabled_feature_groups = disabled_feature_groups or []
        series_data: dict[str, dict[str, tuple[float, float]]] = {}
        for row in train:
            series_data.setdefault(row.series_id, {})[row.bucket_15m.isoformat()] = (
                row.energy_wh, row.cfp_g,
            )

        series_stats: dict[str, dict[str, float]] = {}
        for sid, buckets in series_data.items():
            energies = [e for e, _ in buckets.values()]
            cfps = [c for _, c in buckets.values()]
            n = len(energies)
            mean_e = sum(energies) / n
            std_e = (sum((x - mean_e) ** 2 for x in energies) / n) ** 0.5 + 1e-9
            mean_c = sum(cfps) / n
            std_c = (sum((x - mean_c) ** 2 for x in cfps) / n) ** 0.5 + 1e-9
            series_stats[sid] = {
                "mean_energy": mean_e, "std_energy": std_e,
                "mean_cfp": mean_c, "std_cfp": std_c,
            }

        all_energy = [e for b in series_data.values() for e, _ in b.values()]
        all_cfp = [c for b in series_data.values() for _, c in b.values()]
        global_mean_energy = sum(all_energy) / len(all_energy) if all_energy else 0.0
        global_mean_cfp = sum(all_cfp) / len(all_cfp) if all_cfp else 0.0

        thresholds = peak_threshold(train)
        slot_stats = _compute_slot_stats(series_data)
        sorted_histories = _build_sorted_histories(series_data)

        train_detection_labels: dict[str, dict[str, int]] = {}
        train_peak_labels: dict[str, dict[str, int]] = {}
        for row in train:
            iso_ts = row.bucket_15m.isoformat()
            train_detection_labels.setdefault(row.series_id, {})[iso_ts] = valid_signal_label(row)
            train_peak_labels.setdefault(row.series_id, {})[iso_ts] = peak_label(row, thresholds)

        y_det = [valid_signal_label(row) for row in train]
        y_peak = [peak_label(row, thresholds) for row in train]
        detection_clf = None
        peak_clf = None

        if clf_backend == "rocket":
            try:
                from aeon.classification.convolution_based import RocketClassifier
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "The 'aeon' package is required for --clf-backend rocket. "
                    "Install it with: pip install aeon"
                ) from exc
            X = _build_rocket_input(train, sorted_histories)
            detection_clf = RocketClassifier(n_kernels=10_000, random_state=42)
            peak_clf = RocketClassifier(n_kernels=10_000, random_state=42)
            import numpy as _np
            detection_clf.fit(X, _np.array(y_det))
            peak_clf.fit(X, _np.array(y_peak))

        elif clf_backend in ("rf", "xgb"):
            disabled = frozenset(disabled_feature_groups)
            cross_sums = _build_cross_sums(train)
            zero_streaks = _build_zero_streaks(series_data, [])
            X = [_all_features(row, series_stats, sorted_histories, slot_stats, cross_sums, zero_streaks, disabled)
                 for row in train]
            if clf_backend == "xgb":
                try:
                    from xgboost import XGBClassifier
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "The 'xgboost' package is required for --clf-backend xgb. "
                        "Install it with: pip install xgboost"
                    ) from exc
                detection_clf = XGBClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=sum(1 for y in y_det if y == 0) / max(1, sum(y_det)),
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                )
                peak_clf = XGBClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=sum(1 for y in y_peak if y == 0) / max(1, sum(y_peak)),
                    eval_metric="logloss", random_state=42, n_jobs=-1,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                detection_clf = RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
                )
                peak_clf = RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
                )
            from sklearn.dummy import DummyClassifier
            if len(set(y_det)) < 2:
                detection_clf = DummyClassifier(strategy="most_frequent")
            if len(set(y_peak)) < 2:
                peak_clf = DummyClassifier(strategy="most_frequent")
            detection_clf.fit(X, y_det)
            peak_clf.fit(X, y_peak)

        return cls(
            series_data=series_data,
            peak_thresholds=thresholds,
            global_mean_energy=global_mean_energy,
            global_mean_cfp=global_mean_cfp,
            series_stats=series_stats,
            clf_backend=clf_backend,
            tabpfn_mode=tabpfn_mode,
            forecast_backend=forecast_backend,
            slot_stats=slot_stats,
            train_detection_labels=train_detection_labels,
            train_peak_labels=train_peak_labels,
            disabled_feature_groups=disabled_feature_groups,
            detection_clf=detection_clf,
            peak_clf=peak_clf,
        )

    # ------------------------------------------------------------------
    # Forecasting — dispatches to backend
    # ------------------------------------------------------------------

    def predict_forecasts(
        self,
        series_ids: list[str],
        test_origins: list,
        horizons: tuple[int, ...] = (4, 96),
        tabpfn_mode: str | None = None,
        max_context_length: int | None = None,
    ) -> list[ForecastRow]:
        forecast_ts_list = sorted({
            origin + timedelta(minutes=15 * h)
            for origin in test_origins
            for h in horizons
        })

        backend = self.forecast_backend
        if backend == "nhits":
            energy_lookup, cfp_lookup = _forecast_nhits(
                self.series_data, series_ids, forecast_ts_list, max_context_length,
            )
        elif backend == "prophet":
            energy_lookup, cfp_lookup = _forecast_prophet(
                self.series_data, series_ids, forecast_ts_list, max_context_length,
            )
        elif backend == "tabpfn-ts-feat":
            energy_lookup, cfp_lookup = _forecast_tabpfn_ts_feat(
                self.series_data, series_ids, forecast_ts_list,
                tabpfn_mode or self.tabpfn_mode, max_context_length,
                self.series_stats, self.slot_stats,
            )
        else:
            energy_lookup, cfp_lookup = _forecast_tabpfn_ts(
                self.series_data, series_ids, forecast_ts_list,
                tabpfn_mode or self.tabpfn_mode, max_context_length,
            )

        output: list[ForecastRow] = []
        for origin in test_origins:
            for sid in series_ids:
                for h in horizons:
                    forecast_ts = origin + timedelta(minutes=15 * h)
                    ts_iso = forecast_ts.isoformat()
                    output.append(ForecastRow(
                        sid, forecast_ts, h,
                        energy_lookup.get((sid, ts_iso), 0.0),
                        cfp_lookup.get((sid, ts_iso), 0.0),
                    ))
        return output

    # ------------------------------------------------------------------
    # Shared input builder for rf / rocket
    # ------------------------------------------------------------------

    def _make_clf_input(self, test: list[SeriesRow]) -> Any:
        sorted_histories = _build_sorted_histories(self.series_data)
        if self.clf_backend == "rocket":
            return _build_rocket_input(test, sorted_histories)
        disabled = frozenset(self.disabled_feature_groups)
        cross_sums = _build_cross_sums(test)
        zero_streaks = _build_zero_streaks(self.series_data, test)
        return [
            _all_features(row, self.series_stats, sorted_histories,
                          self.slot_stats, cross_sums, zero_streaks, disabled)
            for row in test
        ]

    # ------------------------------------------------------------------
    # A.1 — valid-signal detection
    # ------------------------------------------------------------------

    def predict_detection(
        self, test: list[SeriesRow], max_context_length: int | None = None
    ) -> list[DetectionRow]:
        disabled = frozenset(self.disabled_feature_groups)
        all_det_labels = [l for s in self.train_detection_labels.values() for l in s.values()]
        det_threshold = sum(all_det_labels) / max(1, len(all_det_labels))
        if self.clf_backend == "tabpfn-ts":
            scores, preds = _tabpfn_ts_classify(
                self.train_detection_labels, test, self.tabpfn_mode, max_context_length, det_threshold
            )
            return [DetectionRow(row.series_id, row.bucket_15m, scores[i], preds[i])
                    for i, row in enumerate(test)]
        if self.clf_backend == "tabpfn-ts-feat":
            scores, preds = _tabpfn_ts_classify_with_features(
                self.train_detection_labels, test, self.tabpfn_mode,
                self.series_data, self.series_stats, self.slot_stats,
                max_context_length, disabled, det_threshold,
            )
            return [DetectionRow(row.series_id, row.bucket_15m, scores[i], preds[i])
                    for i, row in enumerate(test)]

        X = self._make_clf_input(test)
        proba = self.detection_clf.predict_proba(X)
        preds = self.detection_clf.predict(X)
        valid_col = list(self.detection_clf.classes_).index(1)
        return [
            DetectionRow(row.series_id, row.bucket_15m, float(proba[i][valid_col]), int(preds[i]))
            for i, row in enumerate(test)
        ]

    # ------------------------------------------------------------------
    # A.2 — peak-event detection
    # ------------------------------------------------------------------

    def predict_peaks(
        self, test: list[SeriesRow], max_context_length: int | None = None
    ) -> list[PeakRow]:
        disabled = frozenset(self.disabled_feature_groups)
        all_peak_labels = [l for s in self.train_peak_labels.values() for l in s.values()]
        pk_threshold = sum(all_peak_labels) / max(1, len(all_peak_labels))
        if self.clf_backend == "tabpfn-ts":
            scores, preds = _tabpfn_ts_classify(
                self.train_peak_labels, test, self.tabpfn_mode, max_context_length, pk_threshold
            )
            return [PeakRow(row.series_id, row.bucket_15m, scores[i], preds[i])
                    for i, row in enumerate(test)]
        if self.clf_backend == "tabpfn-ts-feat":
            scores, preds = _tabpfn_ts_classify_with_features(
                self.train_peak_labels, test, self.tabpfn_mode,
                self.series_data, self.series_stats, self.slot_stats,
                max_context_length, disabled, pk_threshold,
            )
            return [PeakRow(row.series_id, row.bucket_15m, scores[i], preds[i])
                    for i, row in enumerate(test)]

        X = self._make_clf_input(test)
        proba = self.peak_clf.predict_proba(X)
        preds = self.peak_clf.predict(X)
        peak_col = list(self.peak_clf.classes_).index(1)
        return [
            PeakRow(row.series_id, row.bucket_15m, float(proba[i][peak_col]), int(preds[i]))
            for i, row in enumerate(test)
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(
                {
                    "series_data": {
                        sid: {ts: list(v) for ts, v in buckets.items()}
                        for sid, buckets in self.series_data.items()
                    },
                    "peak_thresholds": self.peak_thresholds,
                    "global_mean_energy": self.global_mean_energy,
                    "global_mean_cfp": self.global_mean_cfp,
                    "series_stats": self.series_stats,
                    "clf_backend": self.clf_backend,
                    "tabpfn_mode": self.tabpfn_mode,
                    "forecast_backend": self.forecast_backend,
                    "slot_stats": self.slot_stats,
                    "train_detection_labels": self.train_detection_labels,
                    "train_peak_labels": self.train_peak_labels,
                    "disabled_feature_groups": self.disabled_feature_groups,
                },
                fh,
            )
        if self.clf_backend in ("rf", "xgb", "rocket"):
            with path.with_suffix(".pkl").open("wb") as fh:
                pickle.dump({"detection_clf": self.detection_clf, "peak_clf": self.peak_clf}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "MyModel":
        path = Path(path)
        with path.open() as fh:
            data = json.load(fh)
        detection_clf = peak_clf = None
        if data["clf_backend"] in ("rf", "xgb", "rocket"):
            with path.with_suffix(".pkl").open("rb") as fh:
                clfs = pickle.load(fh)
            detection_clf = clfs["detection_clf"]
            peak_clf = clfs["peak_clf"]
        return cls(
            series_data={
                sid: {ts: tuple(v) for ts, v in buckets.items()}
                for sid, buckets in data["series_data"].items()
            },
            peak_thresholds=data["peak_thresholds"],
            global_mean_energy=data["global_mean_energy"],
            global_mean_cfp=data["global_mean_cfp"],
            series_stats=data["series_stats"],
            clf_backend=data["clf_backend"],
            tabpfn_mode=data["tabpfn_mode"],
            forecast_backend=data.get("forecast_backend", "tabpfn-ts"),
            slot_stats=data.get("slot_stats", {}),
            train_detection_labels=data.get("train_detection_labels", {}),
            train_peak_labels=data.get("train_peak_labels", {}),
            disabled_feature_groups=data.get("disabled_feature_groups", []),
            detection_clf=detection_clf,
            peak_clf=peak_clf,
        )
