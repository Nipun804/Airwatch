import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import crud
from app.database import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    from app import models  # noqa: F401
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as s:
        crud.seed_departments(s)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    yield TestClient(app)          # no `with` -> lifespan (real DB) is not run
    app.dependency_overrides.clear()


def _user(client, n):
    return client.post("/users", json={"name": f"U{n}", "email": f"u{n}@x.com"}).json()["id"]


def _report(client, uid, text, lat, lon, **kw):
    r = client.post("/complaints", json={"reporter_id": uid, "description": text, "lat": lat, "lon": lon, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def test_full_flow(client):
    u1, u2, u3, u4 = (_user(client, i) for i in range(1, 5))
    assert client.get("/").json()["status"] == "ok"

    a = _report(client, u1, "Garbage is being burned near Sector 62 Noida. Thick smoke spreading.", 28.6270, 77.3649, duration_hours=6)
    b = _report(client, u2, "kachra jalaya ja raha hai Sector 62 ke paas", 28.6272, 77.3650)
    assert a["status"] == "under_verification" and a["category"] == "garbage_burning"
    assert a["hotspot_id"] == b["hotspot_id"]                         # de-duplicated into one incident
    assert a["department"] == "Municipal Corporation - Sanitation"    # routed
    assert b["unique_supporters"] == 2

    assert client.get("/feed").json() == []                            # unverified reports stay out of the public feed
    assert len(client.get("/feed", params={"verified_only": False}).json()) == 1

    # human verification moves the whole incident
    r = client.patch(f"/complaints/{a['id']}/status", json={"new_status": "assigned", "actor": "mod"})
    assert r.status_code == 200 and r.json()["updated_reports"] == 2
    feed = client.get("/feed").json()
    assert len(feed) == 1 and feed[0]["report_count"] == 2 and feed[0]["unique_supporters"] == 2

    before = feed[0]["priority_score"]
    s = client.post(f"/complaints/{a['id']}/support", json={"user_id": u3})
    assert s.status_code == 200 and s.json()["unique_supporters"] == 3 and s.json()["priority_score"] >= before
    assert client.post(f"/complaints/{a['id']}/support", json={"user_id": u3}).status_code == 409   # no double-counting
    assert client.post(f"/complaints/{a['id']}/support", json={"user_id": u1}).status_code == 409   # reporter can't self-support

    bad = client.patch(f"/complaints/{a['id']}/status", json={"new_status": "resolved", "actor": "officer"})
    assert bad.status_code == 409                                       # can't skip steps
    dept_id = next(d["id"] for d in client.get("/departments").json() if d["name"] == a["department"])
    q = client.get(f"/departments/{dept_id}/queue")
    assert q.status_code == 200 and len(q.json()) == 1
    for st in ("acknowledged", "in_progress", "resolved"):
        assert client.patch(f"/complaints/{a['id']}/status", json={"new_status": st, "actor": "officer"}).status_code == 200

    stats = client.get("/stats").json()
    assert stats["resolution_rate"] == 1.0 and stats["resolved_incidents"] == 1 and stats["total_reports"] == 2

    assert client.post(f"/complaints/{a['id']}/reopen", json={"user_id": u4}).status_code == 403   # outsider
    ro = client.post(f"/complaints/{a['id']}/reopen", json={"user_id": u3, "note": "still burning"})
    assert ro.status_code == 200 and ro.json()["status"] == "reopened" and ro.json()["updated_reports"] == 2

    detail = client.get(f"/complaints/{a['id']}").json()
    assert [h["to_status"] for h in detail["history"]][:2] == ["under_verification", "assigned"]
    assert "severity" in detail["priority_breakdown"]


def test_resolved_incident_is_not_rejoined_and_rejection_is_isolated(client):
    u1, u2 = _user(client, 1), _user(client, 2)
    a = _report(client, u1, "Garbage is being burned near Sector 62 Noida.", 28.6270, 77.3649)
    for st in ("assigned", "acknowledged", "in_progress", "resolved"):
        client.patch(f"/complaints/{a['id']}/status", json={"new_status": st, "actor": "x"} if st != "assigned"
                     else {"new_status": "assigned", "actor": "x"})
    c = _report(client, u2, "Garbage is being burned near Sector 62 Noida again.", 28.6270, 77.3649)
    assert c["hotspot_id"] != a["hotspot_id"]                           # new incident after the old one closed

    d = _report(client, u1, "Trucks emitting black smoke near Anand Vihar.", 28.6469, 77.3152)
    r = client.patch(f"/complaints/{d['id']}/status", json={"new_status": "rejected", "actor": "mod", "note": "fake"})
    assert r.status_code == 200
    assert d["hotspot_id"] not in {i["hotspot_id"] for i in client.get("/hotspots").json()}


def test_unknown_user_and_validation(client):
    r = client.post("/complaints", json={"reporter_id": 999, "description": "smoke everywhere", "lat": 28.6, "lon": 77.3})
    assert r.status_code == 404
    r = client.post("/complaints", json={"reporter_id": 1, "description": "x", "lat": 200, "lon": 77.3})
    assert r.status_code == 422
