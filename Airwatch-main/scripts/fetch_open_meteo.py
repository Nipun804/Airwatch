"""Download REAL hourly air-quality + weather for the AirWatch localities from Open-Meteo (free, no API key).

    python -m scripts.fetch_open_meteo

Writes data/air_quality.csv and data/air_quality_meta.json. Afterwards re-train the forecast model:

    python -m ml.train_forecast

NOTE: Open-Meteo air quality = CAMS atmospheric-model estimates, NOT ground-monitor readings. Say so in your README.
"""
import json
import sys
import time

import pandas as pd
import requests

from app.config import DATA_DIR, LOCALITIES
from app.services.aq_data import DATA_FILE, META_FILE, now_ist

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
RENAME = {"pm2_5": "pm25", "nitrogen_dioxide": "no2", "temperature_2m": "temp", "relative_humidity_2m": "rh",
          "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir", "surface_pressure": "pressure"}
COLUMNS = ["time", "location", "lat", "lon", "pm25", "pm10", "no2", "temp", "rh", "wind_speed", "wind_dir", "precip", "pressure"]


def parse(aq_json, wx_json, name, lat, lon):
    """Merge the two Open-Meteo 'hourly' payloads into the AirWatch schema."""
    a, w = pd.DataFrame(aq_json["hourly"]), pd.DataFrame(wx_json["hourly"])
    df = a.merge(w, on="time", how="inner").rename(columns=RENAME).rename(columns={"precipitation": "precip"})
    df["time"] = pd.to_datetime(df["time"])
    df["location"], df["lat"], df["lon"] = name, lat, lon
    return df[COLUMNS]


def fetch_location(name, lat, lon, past_days=92, forecast_days=3):
    common = dict(latitude=lat, longitude=lon, timezone="Asia/Kolkata", past_days=past_days, forecast_days=forecast_days)
    aq = requests.get(AQ_URL, params={**common, "hourly": "pm2_5,pm10,nitrogen_dioxide"}, timeout=60)
    aq.raise_for_status()
    wx = requests.get(WX_URL, params={**common, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                                                          "wind_direction_10m,precipitation,surface_pressure"}, timeout=60)
    wx.raise_for_status()
    return parse(aq.json(), wx.json(), name, lat, lon)


def main():
    frames = []
    for name, (lat, lon, _) in LOCALITIES.items():
        try:
            frames.append(fetch_location(name, lat, lon))
            print(f"ok   {name}")
        except Exception as e:  # keep going; report at the end
            print(f"FAIL {name}: {e}")
        time.sleep(0.3)   # be polite to the free API
    if len(frames) < 3:
        sys.exit("Too few locations downloaded - check your internet connection and try again.")
    df = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    meta = {"source": "open-meteo", "t0": str(now_ist()), "n_locations": len(frames),
            "label": "Open-Meteo air quality (CAMS atmospheric-model estimates) + weather; hourly; not ground-monitor readings"}
    META_FILE.write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(df)} rows for {len(frames)} locations -> {DATA_FILE}")


if __name__ == "__main__":
    main()
