"""Pollution SOURCE identification (indicative evidence fusion, NOT a validated source-apportionment study).

For each area we fuse three independent kinds of evidence about the 5 source types:
  1. CITIZEN evidence  - nearby complaints (weighted by AI priority, unique citizens, verification, resolution)
  2. SENSOR signatures - heuristics on PM10/PM2.5 ratio, NO2, rush-hour / evening / night patterns, wind sector
  3. LAND-USE prior    - industrial areas, outskirts (stubble), dense traffic areas (config.landuse_prior)
The shares are normalised scores, not tonnes of emissions. All heuristics are explicit and editable below.
"""
import numpy as np
import pandas as pd

from app.config import CATEGORIES, DATA_DIR, LOCALITIES, landuse_prior
from app.services.geo import haversine_m

WEIGHTS = {"citizen": 0.45, "sensor": 0.30, "prior": 0.25}
SOURCE_LABEL = {"garbage_burning": "Waste burning", "construction_dust": "Construction & dust",
                "industrial_smoke": "Industry", "vehicle_emission": "Traffic / vehicles", "stubble_burning": "Crop-residue burning"}


def _rel(x, ref):
    """0.5 = same as the network median, 1.0 = 50% higher (or more), 0.0 = 50% lower (or less)."""
    if not ref or not np.isfinite(x) or not np.isfinite(ref):
        return 0.5
    return float(np.clip(0.5 + 0.5 * (x / ref - 1) / 0.5, 0, 1))


def _hour_mean(g, hours):
    v = g.loc[g.time.dt.hour.isin(hours), "pm25"]
    return float(v.mean()) if len(v) else np.nan


def _loc_features(g):
    day = _hour_mean(g, range(12, 17))
    nw = g.wind_dir.between(270, 350)
    nw_ratio = (g.pm25[nw].mean() / g.pm25[~nw].mean()) if nw.sum() >= 6 and (~nw).sum() >= 6 else 1.0
    return {"ratio": float((g.pm10 / g.pm25).median()), "no2": float(g.no2.mean()),
            "rush": _hour_mean(g, range(7, 11)) / day, "evening": _hour_mean(g, range(18, 24)) / day,
            "night": _hour_mean(g, range(0, 6)) / day, "nw": float(nw_ratio)}


def sensor_scores(df, t0, days=7):
    """Returns {area: ({source: score 0..1}, [explanations])}."""
    d = df[(df.time <= t0) & (df.time > pd.Timestamp(t0) - pd.Timedelta(days=days))]
    feats = {name: _loc_features(g) for name, g in d.groupby("location")}
    ref = {k: float(np.nanmedian([f[k] for f in feats.values()])) for k in next(iter(feats.values()))}
    out = {}
    for name, f in feats.items():
        r = {k: _rel(f[k], ref[k]) for k in f}
        s = {"construction_dust": r["ratio"], "vehicle_emission": 0.6 * r["no2"] + 0.4 * r["rush"],
             "industrial_smoke": 0.5 * r["no2"] + 0.5 * r["night"], "garbage_burning": r["evening"], "stubble_burning": r["nw"]}
        notes = []
        if r["ratio"] > .65:
            notes.append(f"PM10/PM2.5 ratio {f['ratio']:.2f} vs network {ref['ratio']:.2f}: coarse-dust signature (construction/road dust)")
        if r["no2"] > .65:
            notes.append(f"NO2 {f['no2']:.0f} vs network {ref['no2']:.0f} µg/m³: combustion from traffic/industry")
        if r["rush"] > .65:
            notes.append("Morning rush-hour PM2.5 peak stronger than elsewhere: traffic signature")
        if r["evening"] > .65:
            notes.append("Evening PM2.5 spikes stronger than elsewhere: waste/biomass burning signature")
        if r["night"] > .65:
            notes.append("PM2.5 stays high through the night: continuous industrial emissions signature")
        if r["nw"] > .65:
            notes.append("PM2.5 higher when wind blows from the north-west: possible upwind crop-residue smoke")
        out[name] = (s, notes)
    return out


def _norm(d):
    s = sum(d.values())
    return {k: (v / s if s > 0 else 1 / len(d)) for k, v in d.items()}


def citizen_scores(lat, lon, incidents, radius_m=3000):
    """Weighted complaint evidence per source near a point. incidents: IncidentOut-like objects."""
    score, counts, top = {c: 0.0 for c in CATEGORIES}, {c: 0 for c in CATEGORIES}, {c: 0.0 for c in CATEGORIES}
    for i in incidents:
        if i.status == "rejected" or float(haversine_m(lat, lon, i.lat, i.lon)) > radius_m:
            continue
        w = (i.priority_score / 100) * (0.5 + 0.5 * min(1.0, np.log1p(i.unique_supporters) / np.log1p(10)))
        w *= 1.0 if i.verified else 0.5            # unverified reports count half
        w *= 0.5 if i.status == "resolved" else 1.0  # resolved incidents count half
        score[i.category] += w
        counts[i.category] += 1
        top[i.category] = max(top[i.category], i.priority_score)
    return score, counts, top


def attribute_sources(df, t0, incidents, radius_m=3000):
    """{area: {shares, components, evidence, n_incidents}}"""
    sensors = sensor_scores(df, t0)
    out = {}
    for name, (sens, notes) in sensors.items():
        g = df[df.location == name].iloc[0]
        lat, lon = float(g.lat), float(g.lon)
        cit, counts, top = citizen_scores(lat, lon, incidents, radius_m)
        comps = {"sensor": _norm({k: v ** 2 for k, v in sens.items()}), "prior": landuse_prior(name)}
        w = dict(WEIGHTS)
        if sum(cit.values()) > 0:
            comps["citizen"] = _norm(cit)
        else:
            w.pop("citizen")
        tot = sum(w.values())
        shares = _norm({c: sum(w[k] / tot * comps[k][c] for k in w) for c in CATEGORIES})
        evidence = [f"{counts[c]} nearby report(s) of {SOURCE_LABEL[c].lower()} (highest priority {top[c]:.0f})"
                    for c in CATEGORIES if counts[c]]
        if not evidence:
            evidence.append("No nearby citizen reports; estimate relies on sensor signatures and land use")
        out[name] = {"shares": shares, "components": comps, "evidence": evidence + notes,
                     "n_incidents": int(sum(counts.values()))}
    return out


# ---------------------------------------------------------------- emission-style density maps
def archive_points(incidents, include_synthetic=True):
    """(lat, lon, category, weight) rows: live incidents (+ optional SIMULATED complaint archive for density)."""
    rows = [(i.lat, i.lon, i.category, float(max(1, i.report_count))) for i in incidents if i.status != "rejected"]
    arch = DATA_DIR / "complaints.csv"
    if include_synthetic and arch.exists():
        a = pd.read_csv(arch, usecols=["lat", "lon", "category"])
        rows += [(r.lat, r.lon, r.category, 1.0) for r in a.itertuples()]
    return pd.DataFrame(rows, columns=["lat", "lon", "category", "weight"])


def emission_grids(points, bandwidth_km=2.5, nx=110, ny=85, margin=0.12):
    """Gaussian kernel density of complaints per source over the NCR bounding box (relative intensity)."""
    lats = [v[0] for v in LOCALITIES.values()]
    lons = [v[1] for v in LOCALITIES.values()]
    lat0, lat1, lon0, lon1 = min(lats) - margin, max(lats) + margin, min(lons) - margin, max(lons) + margin
    glat, glon = np.meshgrid(np.linspace(lat0, lat1, ny), np.linspace(lon0, lon1, nx), indexing="ij")
    kx = 111_320 * np.cos(np.radians((lat0 + lat1) / 2))
    sigma2 = (bandwidth_km * 1000) ** 2
    grids = {c: np.zeros((ny, nx)) for c in CATEGORIES}
    for r in points.itertuples():
        if r.category in grids:
            d2 = ((glon - r.lon) * kx) ** 2 + ((glat - r.lat) * 111_320) ** 2
            grids[r.category] += r.weight * np.exp(-d2 / (2 * sigma2))
    return {"extent": (lon0, lon1, lat0, lat1), "grids": grids}
