"""Air-quality data access: a real Open-Meteo snapshot if present, otherwise a clearly-labelled simulation."""
import json
from functools import lru_cache

import pandas as pd

from app.config import DATA_DIR
from app.services.aq_sim import simulate

DATA_FILE = DATA_DIR / "air_quality.csv"
META_FILE = DATA_DIR / "air_quality_meta.json"

# CPCB National AQI concentration bands for PM2.5 (ug/m3, defined for 24-h averages; applied to hourly
# values here as an approximation).
PM25_BANDS = [(30, "Good", "#2ca02c"), (60, "Satisfactory", "#8bc34a"), (90, "Moderate", "#e6b800"),
              (120, "Poor", "#ff7f0e"), (250, "Very poor", "#d62728"), (float("inf"), "Severe", "#7f1d1d")]
CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def pm25_category(v):
    for upper, label, color in PM25_BANDS:
        if v <= upper:
            return label, color


def cardinal(deg):
    return CARDINALS[int((float(deg) % 360) / 22.5 + 0.5) % 16]


def now_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None)


@lru_cache(maxsize=4)
def _simulated(hour_key):
    now = pd.Timestamp(hour_key)
    meta = {"source": "simulated", "t0": now,
            "label": "SIMULATED air-quality data (structure is realistic, values are not measurements)"}
    return simulate(now), meta


def load_air_quality():
    """Returns (DataFrame, meta). meta['t0'] is 'now' for the dataset; rows after t0 hold FORECAST weather."""
    if DATA_FILE.exists() and META_FILE.exists():
        meta = json.loads(META_FILE.read_text())
        meta["t0"] = pd.Timestamp(meta["t0"])
        return pd.read_csv(DATA_FILE, parse_dates=["time"]), meta
    df, meta = _simulated(str(now_ist()))
    return df.copy(), dict(meta)


def is_real(meta):
    return meta.get("source") != "simulated"
