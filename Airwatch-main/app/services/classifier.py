"""Inference wrapper around the trained category + severity models."""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.config import ARTIFACTS_DIR, CATEGORIES

TOKEN_PATTERN = r"[^\s.,!?।;:]+"   # keeps Devanagari matras intact (default \w+ splits them)

# Crude fallback so the API still works before you have trained anything.
_FALLBACK_KEYWORDS = {
    "garbage_burning": ["garbage", "kachra", "kooda", "कचरा", "कूड़ा", "waste", "trash"],
    "construction_dust": ["construction", "dust", "dhool", "धूल", "debris", "malba"],
    "industrial_smoke": ["factory", "chimney", "industrial", "fakatri", "फैक्ट्री", "chemical"],
    "vehicle_emission": ["truck", "bus", "vehicle", "diesel", "gaadi", "ट्रक", "गाड़ी"],
    "stubble_burning": ["stubble", "parali", "पराली", "crop", "khet", "खेत", "straw"],
}


def build_severity_frame(texts, nearest_sensitive_m, duration_hours):
    """Shared by training and inference so features always match."""
    sens = np.asarray(nearest_sensitive_m, dtype=float)
    dur = np.asarray(duration_hours, dtype=float)
    return pd.DataFrame({
        "text": list(texts),
        "sens_prox": 1.0 - np.clip(sens / 1000.0, 0, 1),
        "duration_norm": np.clip(dur / 48.0, 0, 1),
    })


@lru_cache(maxsize=None)
def _load(name):
    path = ARTIFACTS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:          # e.g. scikit-learn version mismatch -> caller can retrain
        return None


def models_available():
    return _load("category_model") is not None and _load("severity_model") is not None


def predict(text, nearest_sensitive_m=None, duration_hours=None):
    sens = 5000.0 if nearest_sensitive_m is None else nearest_sensitive_m
    dur = 6.0 if duration_hours is None else duration_hours
    cat_model, sev_model = _load("category_model"), _load("severity_model")

    if cat_model is None or sev_model is None:
        low = text.lower()
        scores = {c: sum(k in low for k in kws) for c, kws in _FALLBACK_KEYWORDS.items()}
        cat = max(scores, key=scores.get) if max(scores.values()) > 0 else CATEGORIES[0]
        return {"category": cat, "category_confidence": 0.0, "severity": "medium",
                "severity_confidence": 0.0, "needs_human_review": True, "model": "fallback"}

    cat_proba = cat_model.predict_proba([text])[0]
    cat_i = int(np.argmax(cat_proba))
    frame = build_severity_frame([text], [sens], [dur])
    sev_proba = sev_model.predict_proba(frame)[0]
    sev_i = int(np.argmax(sev_proba))
    cat_conf, sev_conf = float(cat_proba[cat_i]), float(sev_proba[sev_i])
    return {
        "category": str(cat_model.classes_[cat_i]), "category_confidence": round(cat_conf, 3),
        "severity": str(sev_model.classes_[sev_i]), "severity_confidence": round(sev_conf, 3),
        # Low-confidence predictions are flagged for a human; the AI only recommends.
        "needs_human_review": bool(cat_conf < 0.6 or sev_conf < 0.5),
        "model": "trained",
    }


def probabilities(text, nearest_sensitive_m=None, duration_hours=None):
    """Full class probabilities for the 'try the model' demo. None if models aren't trained."""
    cat_model, sev_model = _load("category_model"), _load("severity_model")
    if cat_model is None or sev_model is None:
        return None
    sens = 5000.0 if nearest_sensitive_m is None else nearest_sensitive_m
    dur = 6.0 if duration_hours is None else duration_hours
    cp = cat_model.predict_proba([text])[0]
    sp = sev_model.predict_proba(build_severity_frame([text], [sens], [dur]))[0]
    return {"category": dict(zip(map(str, cat_model.classes_), map(float, cp))),
            "severity": dict(zip(map(str, sev_model.classes_), map(float, sp)))}
