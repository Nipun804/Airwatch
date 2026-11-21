"""Duplicate detection + hotspot clustering.

Idea: reports about the same incident are (a) the same pollution category and (b) physically close.
We cluster per category with DBSCAN (min_samples=1, haversine) so every report belongs to a hotspot,
and a lone report is simply a hotspot of size 1.
"""
import json

import numpy as np
from sklearn.cluster import DBSCAN

from app.config import ARTIFACTS_DIR
from app.services.geo import EARTH_M, distance_matrix, haversine_m

DEFAULT_RADIUS_M = 150.0


def load_radius():
    p = ARTIFACTS_DIR / "dedupe_config.json"
    if p.exists():
        return float(json.loads(p.read_text())["radius_m"])
    return DEFAULT_RADIUS_M


def cluster_hotspots(lats, lons, categories, radius_m=None):
    """Batch clustering. Returns an int label per report; labels are unique across categories."""
    radius_m = radius_m or load_radius()
    lats, lons, categories = np.asarray(lats, float), np.asarray(lons, float), np.asarray(categories)
    labels = np.full(len(lats), -1, dtype=int)
    next_label = 0
    for cat in np.unique(categories):
        idx = np.where(categories == cat)[0]
        coords = np.radians(np.c_[lats[idx], lons[idx]])
        sub = DBSCAN(eps=radius_m / EARTH_M, min_samples=1, metric="haversine", algorithm="ball_tree").fit_predict(coords)
        labels[idx] = sub + next_label
        next_label += sub.max() + 1
    return labels


def find_matching_hotspot(lat, lon, category, hotspots, radius_m=None):
    """Online matching for a NEW complaint: nearest same-category hotspot within the radius, else None.
    `hotspots` is any iterable of objects with .id .lat .lon .category (e.g. Hotspot rows)."""
    radius_m = radius_m or load_radius()
    best, best_d = None, radius_m
    for h in hotspots:
        if h.category != category:
            continue
        d = float(haversine_m(lat, lon, h.lat, h.lon))
        if d <= best_d:
            best, best_d = h, d
    return best


def evaluate_duplicates(lats, lons, categories, cluster_ids, radius_m):
    """Pairwise duplicate detection: predict 'same incident' = same category AND distance <= radius."""
    categories, cluster_ids = np.asarray(categories), np.asarray(cluster_ids)
    D = distance_matrix(lats, lons)
    pred = (D <= radius_m) & (categories[:, None] == categories[None, :])
    true = cluster_ids[:, None] == cluster_ids[None, :]
    iu = np.triu_indices(len(categories), k=1)
    p, t = pred[iu], true[iu]
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
