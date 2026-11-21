import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import crud
from app.database import Base
from app.models import Complaint, Hotspot, User
from app.services import classifier, lifecycle
from app.services.dedupe import cluster_hotspots, find_matching_hotspot
from app.services.geo import haversine_m, offset_latlon
from app.services.priority import compute_priority


@pytest.fixture()
def db():
    engine = create_engine("sqlite://")
    from app import models  # noqa: F401
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    crud.seed_departments(s)
    yield s
    s.close()


def _complaint(db, reporter, hotspot_id=None, status="submitted"):
    c = Complaint(reporter_id=reporter, description="garbage burning", lat=28.6, lon=77.3, category="garbage_burning",
                  severity="high", status=status, hotspot_id=hotspot_id)
    db.add(c); db.commit()
    return c


def test_haversine_one_degree_latitude():
    assert abs(float(haversine_m(0, 0, 1, 0)) - 111_195) < 300


def test_offset_roundtrip():
    lat, lon = offset_latlon(28.6, 77.3, 100, 0)
    assert abs(float(haversine_m(28.6, 77.3, lat, lon)) - 100) < 1


def test_priority_ordering_and_breakdown():
    hi = compute_priority("critical", "industrial_smoke", 120, 20, 40, 0.9)
    lo = compute_priority("low", "vehicle_emission", 3000, 1, 1, 0.3)
    assert hi["score"] > lo["score"]
    assert hi["level"] == "critical" and lo["level"] == "low"
    assert abs(sum(b["points"] for b in hi["breakdown"].values()) - hi["score"]) < 0.2


def test_more_unique_supporters_raises_priority_but_saturates():
    s = lambda n: compute_priority("medium", "garbage_burning", 500, n, 5, 0.5)["score"]
    assert s(1) < s(5) < s(25) and s(25) == s(500)


def test_lifecycle_valid_path_and_history(db):
    c = _complaint(db, reporter=1)
    for new in ["under_verification", "assigned", "acknowledged", "in_progress", "resolved"]:
        lifecycle.apply_transition(db, c, new, actor="test")
    db.commit()
    assert c.status == "resolved" and c.resolved_at is not None
    assert [h.to_status for h in c.history][-1] == "resolved" and len(c.history) == 5
    lifecycle.apply_transition(db, c, "reopened", actor="citizen"); db.commit()
    assert c.resolved_at is None


def test_lifecycle_rejects_skips(db):
    c = _complaint(db, reporter=1)
    with pytest.raises(lifecycle.InvalidTransition):
        lifecycle.apply_transition(db, c, "resolved", actor="x")
    with pytest.raises(lifecycle.InvalidTransition):
        lifecycle.apply_transition(db, c, "nonsense", actor="x")


def test_unique_supporters_counts_each_citizen_once(db):
    for i in (1, 2, 3):
        db.add(User(name=f"u{i}", email=f"u{i}@x.com")); db.commit()
    h = Hotspot(category="garbage_burning", lat=28.6, lon=77.3); db.add(h); db.commit()
    c1, _ = _complaint(db, 1, h.id), _complaint(db, 1, h.id)   # same citizen reports twice
    assert crud.add_support(db, c1.id, 2) is True
    assert crud.add_support(db, c1.id, 2) is False              # duplicate support rejected
    assert crud.add_support(db, c1.id, 1) is True               # reporter "supporting" own incident
    assert crud.unique_supporters(db, h.id) == 2                # citizens 1 and 2


def test_clustering_groups_nearby_same_category_only():
    lat, lon = 28.6, 77.3
    lat2, lon2 = offset_latlon(lat, lon, 40, 0)       # 40 m away
    lat3, lon3 = offset_latlon(lat, lon, 2000, 0)     # 2 km away
    labels = cluster_hotspots([lat, lat2, lat3, lat], [lon, lon2, lon3, lon],
                              ["garbage_burning", "garbage_burning", "garbage_burning", "construction_dust"], radius_m=150)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]
    assert labels[3] != labels[0]                      # same spot, different category


def test_find_matching_hotspot():
    h = Hotspot(id=7, category="garbage_burning", lat=28.6, lon=77.3)
    la, lo = offset_latlon(28.6, 77.3, 60, 0)
    assert find_matching_hotspot(la, lo, "garbage_burning", [h], 150) is h
    assert find_matching_hotspot(la, lo, "industrial_smoke", [h], 150) is None


def test_classifier_predicts_when_trained():
    if not classifier.models_available():
        pytest.skip("run `python -m ml.train` first")
    r = classifier.predict("Factory chimney releasing black smoke near Okhla. Huge amount of smoke, children coughing.", 120, 24)
    assert r["category"] == "industrial_smoke" and r["severity"] in ("high", "critical")
    assert classifier.predict("खेत में पराली जलाई जा रही है")["category"] == "stubble_burning"
