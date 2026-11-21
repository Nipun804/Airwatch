"""Transparent, explainable priority score (0-100). Every factor and its contribution is returned."""
import math

from app.config import CATEGORY_RISK, SEVERITY_SCORE

WEIGHTS = {
    "severity": 0.30,
    "sensitive_proximity": 0.20,   # school / hospital nearby
    "unique_supporters": 0.15,
    "category_risk": 0.15,
    "duration": 0.10,
    "population_density": 0.10,    # proxy for "people affected"
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def _proximity(d_m):
    """1.0 within 200 m of a school/hospital, falling linearly to 0 at 1000 m."""
    if d_m is None:
        return 0.0
    return max(0.0, min(1.0, (1000.0 - d_m) / 800.0))


def _supporters(n):
    """Log-scaled so 25 unique citizens ~ saturates the factor (stops brigading)."""
    return min(1.0, math.log1p(max(n, 1) - 1) / math.log1p(24))


def level_for(score):
    return "critical" if score >= 75 else "high" if score >= 55 else "medium" if score >= 35 else "low"


def compute_priority(severity, category, nearest_sensitive_m=None, unique_supporters=1,
                     duration_hours=None, density_index=0.5):
    values = {
        "severity": SEVERITY_SCORE[severity],
        "sensitive_proximity": _proximity(nearest_sensitive_m),
        "unique_supporters": _supporters(unique_supporters),
        "category_risk": CATEGORY_RISK[category],
        "duration": min(1.0, (duration_hours or 0.0) / 48.0),
        "population_density": max(0.0, min(1.0, density_index)),
    }
    breakdown = {
        k: {"value": round(v, 3), "weight": WEIGHTS[k], "points": round(100 * WEIGHTS[k] * v, 1)}
        for k, v in values.items()
    }
    score = round(sum(b["points"] for b in breakdown.values()), 1)
    return {"score": score, "level": level_for(score), "breakdown": breakdown}
