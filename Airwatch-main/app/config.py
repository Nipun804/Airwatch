"""Shared constants. Single source of truth for categories, severities, routing and localities."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
DB_URL = os.environ.get("AIRWATCH_DB_URL", f"sqlite:///{ROOT / 'airwatch.db'}")

CATEGORIES = [
    "garbage_burning",
    "construction_dust",
    "industrial_smoke",
    "vehicle_emission",
    "stubble_burning",
]
SEVERITIES = ["low", "medium", "high", "critical"]  # ordered

SEVERITY_SCORE = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.00}

# Assumed relative health/environmental risk of each source (0-1). Tune/justify in README.
CATEGORY_RISK = {
    "garbage_burning": 0.80,
    "construction_dust": 0.50,
    "industrial_smoke": 0.90,
    "vehicle_emission": 0.40,
    "stubble_burning": 0.70,
}

# SIMULATED departments for the demo (no real government integration).
DEPARTMENTS = [
    "Municipal Corporation - Sanitation",
    "Municipal Corporation - Building & Construction",
    "State Pollution Control Board",
    "Transport & Traffic Department",
    "Agriculture Department",
]
ROUTING = {
    "garbage_burning": "Municipal Corporation - Sanitation",
    "construction_dust": "Municipal Corporation - Building & Construction",
    "industrial_smoke": "State Pollution Control Board",
    "vehicle_emission": "Transport & Traffic Department",
    "stubble_burning": "Agriculture Department",
}

# name: (lat, lon, population-density index 0-1). Coordinates are approximate;
# the density index is an ILLUSTRATIVE proxy for "people affected" - say so in the README.
LOCALITIES = {
    "Sector 62 Noida": (28.6270, 77.3649, 0.70),
    "Sector 18 Noida": (28.5708, 77.3260, 0.80),
    "Indirapuram": (28.6460, 77.3690, 0.90),
    "Vaishali": (28.6450, 77.3400, 0.85),
    "Dwarka": (28.5921, 77.0460, 0.75),
    "Rohini": (28.7495, 77.0565, 0.80),
    "Lajpat Nagar": (28.5677, 77.2430, 0.95),
    "Okhla": (28.5355, 77.2750, 0.85),
    "Anand Vihar": (28.6469, 77.3152, 0.90),
    "Sector 29 Gurugram": (28.4690, 77.0670, 0.70),
    "Faridabad Sector 15": (28.4089, 77.3178, 0.65),
    "Raj Nagar Ghaziabad": (28.6692, 77.4538, 0.75),
    "Najafgarh": (28.6090, 76.9855, 0.60),
    "Mundka": (28.6824, 77.0310, 0.65),
    "Bawana": (28.7990, 77.0330, 0.50),
    "Narela": (28.8530, 77.0920, 0.45),
    "Pari Chowk Greater Noida": (28.4744, 77.5030, 0.40),
    "Mayur Vihar": (28.6090, 77.2970, 0.90),
    "Punjabi Bagh": (28.6683, 77.1313, 0.85),
    "Sarita Vihar": (28.5290, 77.2900, 0.70),
}
OUTSKIRTS = ["Narela", "Bawana", "Najafgarh", "Pari Chowk Greater Noida", "Mundka", "Faridabad Sector 15"]
INDUSTRIAL = ["Okhla", "Bawana", "Narela", "Mundka", "Faridabad Sector 15", "Anand Vihar"]


import zlib  # noqa: E402


def landuse_prior(name: str) -> dict:
    """Crude land-use prior of which pollution sources matter in an area. An ASSUMPTION (editable):
    industrial areas -> industry, outskirts -> stubble burning, dense areas -> vehicles."""
    dens = LOCALITIES[name][2] if name in LOCALITIES else 0.5
    h = (zlib.crc32(name.encode()) % 100) / 100.0   # deterministic per-area variation
    w = {
        "garbage_burning": 0.10 + 0.10 * h,
        "construction_dust": 0.10 + 0.15 * (1 - h) * dens,
        "industrial_smoke": 0.04,
        "vehicle_emission": 0.10 + 0.40 * dens,
        "stubble_burning": 0.02,
    }
    if name in INDUSTRIAL:
        w["industrial_smoke"] += 0.45
    if name in OUTSKIRTS:
        w["stubble_burning"] += 0.30
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}
