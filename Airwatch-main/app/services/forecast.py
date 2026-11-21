"""48-hour PM2.5 forecasting (direct multi-horizon gradient boosting with a calibrated 80% uncertainty band).

One model is trained across all areas and horizons (horizon h is a feature). It predicts the log-CHANGE versus
persistence ("no change from now"). Weather at the forecast target time is an input: in training it is the
observed weather, at inference it is the weather FORECAST rows that follow 'now' in the dataset.
"""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.config import ARTIFACTS_DIR, INDUSTRIAL, LOCALITIES, OUTSKIRTS

FEATURES = ["pm_now", "lag1", "lag3", "lag6", "lag12", "lag24", "mean24", "mean168", "h", "hour_sin", "hour_cos",
            "tw_temp", "tw_rh", "tw_ws", "tw_wd_sin", "tw_wd_cos", "tw_precip", "tw_pressure", "ws_now", "rh_now",
            "dens", "ind", "out"]
GROUPS = {  # friendly groups for explainability
    "Recent PM2.5 history": ["pm_now", "lag1", "lag3", "lag6", "lag12", "lag24", "mean24"],
    "7-day local baseline": ["mean168"], "Forecast horizon": ["h"], "Time of day": ["hour_sin", "hour_cos"],
    "Wind speed": ["tw_ws", "ws_now"], "Wind direction": ["tw_wd_sin", "tw_wd_cos"], "Humidity": ["tw_rh", "rh_now"],
    "Temperature": ["tw_temp"], "Rain": ["tw_precip"], "Pressure": ["tw_pressure"], "Area type": ["dens", "ind", "out"]}
HORIZONS = list(range(1, 49))
TRAIN_H = [1, 3, 6, 12, 18, 24, 30, 36, 42, 48]
MIN_HISTORY = 168


def _static(name):
    dens = LOCALITIES.get(name, (0, 0, 0.5))[2]
    return dens, float(name in INDUSTRIAL), float(name in OUTSKIRTS)


def _regular(g, t_index):
    return g.sort_values("time").drop_duplicates("time").set_index("time").reindex(t_index)


def _rows(g, name, origins, horizons, with_target=True):
    o = np.asarray(origins, int)
    pm = g["pm25"].to_numpy(float)
    hour = g.index.hour.to_numpy()
    roll24 = pd.Series(pm).rolling(24, min_periods=18).mean().to_numpy()
    roll168 = pd.Series(pm).rolling(168, min_periods=120).mean().to_numpy()
    wx = {c: g[c].to_numpy(float) for c in ["temp", "rh", "wind_speed", "wind_dir", "precip", "pressure"]}
    dens, ind, out = _static(name)
    parts = []
    for h in horizons:
        t = o + h
        d = pd.DataFrame({
            "pm_now": pm[o], "lag1": pm[o - 1], "lag3": pm[o - 3], "lag6": pm[o - 6], "lag12": pm[o - 12], "lag24": pm[o - 24],
            "mean24": roll24[o], "mean168": roll168[o], "h": float(h),
            "hour_sin": np.sin(2 * np.pi * hour[t] / 24), "hour_cos": np.cos(2 * np.pi * hour[t] / 24),
            "tw_temp": wx["temp"][t], "tw_rh": wx["rh"][t], "tw_ws": wx["wind_speed"][t],
            "tw_wd_sin": np.sin(np.radians(wx["wind_dir"][t])), "tw_wd_cos": np.cos(np.radians(wx["wind_dir"][t])),
            "tw_precip": wx["precip"][t], "tw_pressure": wx["pressure"][t],
            "ws_now": wx["wind_speed"][o], "rh_now": wx["rh"][o], "dens": dens, "ind": ind, "out": out})
        d["persist"] = pm[o]
        d["snaive"] = pm[o + h - 24 * int(np.ceil(h / 24))]      # same hour on the previous day(s)
        if with_target:
            d["y"] = pm[t]
        d["_h"], d["location"] = h, name
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _t_index(df):
    return pd.date_range(df.time.min(), df.time.max(), freq="h")


def build_dataset(df, t0):
    """Time-based split of the history: first 65% train, next 15% calibration (band width), last 20% test.
    Targets of one split never reach into the next."""
    t_index = _t_index(df)
    n_hist = int((t_index <= t0).sum())
    c1, c2 = int(0.65 * n_hist), int(0.80 * n_hist)
    tr, ca, te = [], [], []
    for name, g in df.groupby("location"):
        g = _regular(g, t_index)
        tr.append(_rows(g, name, np.arange(MIN_HISTORY, c1 - 48, 2), TRAIN_H))
        ca.append(_rows(g, name, np.arange(c1, c2 - 48, 2), TRAIN_H))
        te.append(_rows(g, name, np.arange(c2, n_hist - 48), HORIZONS))
    clean = lambda parts, cols: pd.concat(parts, ignore_index=True).dropna(subset=cols)
    return clean(tr, ["y", "persist"]), clean(ca, ["y", "persist"]), clean(te, ["y", "persist", "snaive"])


@lru_cache(maxsize=1)
def _load():
    p = ARTIFACTS_DIR / "forecast_models.joblib"
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def models_available():
    return _load() is not None


def load_metrics():
    p = ARTIFACTS_DIR / "forecast_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def band(models, X, k=None):
    """Log-space residual models: forecast = persistence * exp(predicted change). The band is widened by a factor k
    chosen on a calibration slice so the 80% interval covers ~80% of held-out observations."""
    f, base = X[FEATURES], np.log1p(X["persist"].to_numpy(float))
    m, q10, q90 = models["median"].predict(f), models["q10"].predict(f), models["q90"].predict(f)
    k = models.get("k", 1.0) if k is None else k
    lo_r, hi_r = m - k * (m - np.minimum(q10, m)), m + k * (np.maximum(q90, m) - m)
    return np.clip(np.expm1(base + m), 0, None), np.clip(np.expm1(base + lo_r), 0, None), np.expm1(base + hi_r)


def _predict(models, X):
    return band(models, X)


def forecast_all(df, t0, horizons=HORIZONS):
    """Forecast every area from 'now' (t0). Returns columns: location, h, time, pm25, lo, hi."""
    models = _load()
    t_index = _t_index(df)
    t0 = pd.Timestamp(t0)
    if t0 not in t_index:
        raise ValueError("t0 is not on the data's hourly grid")
    pos = int(np.where(t_index == t0)[0][0])
    if pos < MIN_HISTORY or pos + max(horizons) >= len(t_index):
        raise ValueError("Not enough history or future-weather rows around t0")
    rows = [_rows(_regular(g, t_index), name, [pos], horizons, with_target=False) for name, g in df.groupby("location")]
    X = pd.concat(rows, ignore_index=True)
    med, lo, hi = _predict(models, X)
    out = X[["location", "_h"]].rename(columns={"_h": "h"})
    out["time"] = t0 + pd.to_timedelta(out["h"], unit="h")
    out["pm25"], out["lo"], out["hi"] = med, lo, hi
    return out


def history(df, location, t0, hours=168):
    g = df[(df.location == location) & (df.time <= t0) & (df.time > pd.Timestamp(t0) - pd.Timedelta(hours=hours))]
    return g.sort_values("time")


def first_crossing(fc, threshold):
    """First forecast hour above the threshold (and the peak), or None."""
    above = fc[fc.pm25 > threshold]
    if above.empty:
        return None
    peak = fc.loc[fc.pm25.idxmax()]
    first = above.iloc[0]
    return {"first_time": first.time, "first_value": float(first.pm25), "peak_time": peak.time, "peak_value": float(peak.pm25)}
