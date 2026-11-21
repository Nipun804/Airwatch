"""SIMULATED hourly air-quality + weather for the Delhi NCR localities.

Used only when no real data snapshot (data/air_quality.csv) exists. It has realistic structure
(diurnal cycle, wind/rain ventilation, multi-day pollution episodes, area differences, pollutant
signatures per land use) so the forecasting/hotspot/source pipelines can be demonstrated end to end,
but the numbers are NOT measurements. The UI labels it as simulated.
"""
import numpy as np
import pandas as pd

from app.config import INDUSTRIAL, LOCALITIES, OUTSKIRTS, landuse_prior


def _ar1(n, phi, sigma, rng):
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def simulate(now, days_back=92, hours_ahead=72, seed=None):
    now = pd.Timestamp(now).floor("h")
    times = pd.date_range(now - pd.Timedelta(days=days_back), now + pd.Timedelta(hours=hours_ahead), freq="h")
    n, hr = len(times), times.hour.values
    rng = np.random.default_rng(seed if seed is not None else int(now.strftime("%Y%m%d")))

    # regional weather shared by all areas (winter-like climatology)
    temp = 18 + 7 * np.cos(2 * np.pi * (hr - 15) / 24) + _ar1(n, .97, .35, rng)
    rh = np.clip(68 - 1.6 * (temp - 18) + _ar1(n, .97, 1.2, rng), 15, 100)
    ws = np.clip(7 + 3 * np.cos(2 * np.pi * (hr - 14) / 24) + 1.5 * _ar1(n, .95, .9, rng), 0.5, 40)
    wdir = (300 + _ar1(n, .985, 3.0, rng) + 50 * np.sin(2 * np.pi * np.arange(n) / (24 * 6))) % 360
    precip = np.where(_ar1(n, .97, .5, rng) > 3.4, 0.8, 0.0)
    pressure = 1014 + _ar1(n, .99, .3, rng)
    episode = _ar1(n, .994, .04, rng)          # multi-day regional pollution episodes

    frames = []
    for name, (lat, lon, dens) in LOCALITIES.items():
        pri = landuse_prior(name)
        base = 85 + 75 * dens + (30 if name in INDUSTRIAL else 0) + (15 if name in OUTSKIRTS else 0)
        diurnal = 0.22 * np.cos(2 * np.pi * (hr - 21) / 24) + 0.12 * np.cos(4 * np.pi * (hr - 8) / 24)
        signature = (1.2 * pri["garbage_burning"] * np.exp(-((hr - 21) ** 2) / 8)            # evening burning
                     + 0.4 * pri["industrial_smoke"] * (hr < 6)                               # night-time industry
                     + 0.5 * pri["vehicle_emission"] * np.exp(-((hr - 9) ** 2) / 6))          # morning rush
        logpm = (np.log(base) + 0.8 * episode + diurnal + signature - 0.035 * (ws - 7)
                 - 0.6 * (precip > 0) + 0.004 * (rh - 65) + _ar1(n, .9, .05, rng))
        pm25 = np.clip(np.exp(logpm), 8, 700)
        pm10 = pm25 * (1.4 + 3.0 * pri["construction_dust"]) * np.exp(rng.normal(0, .06, n))
        no2 = ((18 + 70 * pri["vehicle_emission"] + 60 * pri["industrial_smoke"]) * (1 + .35 * np.cos(2 * np.pi * (hr - 9) / 24))
               * np.exp(-0.03 * (ws - 7)) * np.exp(rng.normal(0, .1, n)))
        frames.append(pd.DataFrame({
            "time": times, "location": name, "lat": lat, "lon": lon,
            "pm25": pm25.round(1), "pm10": pm10.round(1), "no2": no2.round(1),
            "temp": temp.round(1), "rh": rh.round(0), "wind_speed": ws.round(1), "wind_dir": wdir.round(0),
            "precip": precip, "pressure": pressure.round(1)}))
    return pd.concat(frames, ignore_index=True)
