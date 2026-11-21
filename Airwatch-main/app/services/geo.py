"""Geo helpers: distances, nearest locality, nearest school/hospital."""
from functools import lru_cache

import numpy as np
import pandas as pd

from app.config import DATA_DIR, LOCALITIES

EARTH_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Works with scalars or numpy arrays (broadcasting)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_M * np.arcsin(np.sqrt(a))


def distance_matrix(lats, lons):
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    return haversine_m(lats[:, None], lons[:, None], lats[None, :], lons[None, :])


def offset_latlon(lat, lon, north_m, east_m):
    """Move a point by metres north/east (small distances only)."""
    return lat + north_m / 111_320.0, lon + east_m / (111_320.0 * np.cos(np.radians(lat)))


def nearest_locality(lat, lon):
    """Returns (name, density_index) of the closest known locality (crude reverse geocode)."""
    best, best_d = None, float("inf")
    for name, (la, lo, dens) in LOCALITIES.items():
        d = float(haversine_m(lat, lon, la, lo))
        if d < best_d:
            best, best_d, best_dens = name, d, dens
    return best, best_dens


def locality_density(name, default=0.5):
    return LOCALITIES.get(name, (None, None, default))[2]


@lru_cache(maxsize=1)
def load_places(path=None):
    """Schools/hospitals. data/places.csv columns: name,type,lat,lon"""
    path = path or DATA_DIR / "places.csv"
    if not path.exists():
        return pd.DataFrame(columns=["name", "type", "lat", "lon"])
    return pd.read_csv(path)


def nearest_sensitive_m(lat, lon, places=None):
    """Distance (m) to the nearest school/hospital; 5000 if no places are loaded."""
    places = load_places() if places is None else places
    if len(places) == 0:
        return 5000.0
    return float(np.min(haversine_m(lat, lon, places["lat"].values, places["lon"].values)))
