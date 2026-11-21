"""Pollution hotspot identification: where is air quality REPEATEDLY bad, and who/what is behind it?"""
import pandas as pd

from app.services.aq_data import pm25_category
from app.services.geo import haversine_m


def station_table(df, t0, incidents=None, forecasts=None, when_h=0, threshold=120.0, radius_m=3000, attribution=None):
    """One row per area. when_h=0 -> latest reading; when_h>0 -> forecast value at t0+when_h hours.
    hotspot_index (0-100) = 45% recurrence (share of last 7 days above threshold) + 25% average level
                           + 30% open citizen-report pressure nearby. Weights are an editable assumption."""
    t0 = pd.Timestamp(t0)
    last7 = df[(df.time <= t0) & (df.time > t0 - pd.Timedelta(days=7))]
    rows = []
    for name, g in last7.groupby("location"):
        g = g.sort_values("time")
        value = float(g.pm25.iloc[-1])
        if when_h and forecasts is not None:
            f = forecasts[(forecasts.location == name) & (forecasts.h == when_h)]
            if len(f):
                value = float(f.pm25.iloc[0])
        lat, lon = float(g.lat.iloc[0]), float(g.lon.iloc[0])
        near = [i for i in (incidents or []) if i.status not in ("resolved", "rejected")
                and float(haversine_m(lat, lon, i.lat, i.lon)) <= radius_m]
        pressure = sum(i.priority_score / 100 for i in near)
        exceed = float((g.pm25 > threshold).mean())
        index = 100 * (0.45 * exceed + 0.25 * min(1.0, float(g.pm25.mean()) / 250) + 0.30 * min(1.0, pressure / 3))
        driver = None
        if attribution and name in attribution:
            driver = max(attribution[name]["shares"], key=attribution[name]["shares"].get)
        cat, color = pm25_category(value)
        rows.append({"location": name, "lat": lat, "lon": lon, "pm25": round(value, 1), "category": cat, "color": color,
                     "mean_7d": round(float(g.pm25.mean()), 1), "exceed_7d": round(exceed, 3),
                     "open_incidents": len(near), "hotspot_index": round(index, 1), "likely_driver": driver})
    return pd.DataFrame(rows).sort_values("hotspot_index", ascending=False).reset_index(drop=True)
