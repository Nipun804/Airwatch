"""The core flow behind POST /complaints, plus incident-level priority refresh."""
from sqlalchemy import select

from app import crud
from app.config import CATEGORIES, ROUTING, SEVERITIES
from app.models import Complaint, Hotspot, User
from app.schemas import ComplaintCreate
from app.services import classifier, lifecycle
from app.services.dedupe import find_matching_hotspot
from app.services.geo import nearest_locality, nearest_sensitive_m
from app.services.priority import compute_priority

CLOSED = ["resolved", "rejected"]


def _open_hotspots(db):
    """Hotspots that still have at least one unresolved report (a resolved incident is not re-joined)."""
    return db.scalars(select(Hotspot).where(
        Hotspot.id.in_(select(Complaint.hotspot_id).where(Complaint.status.notin_(CLOSED))))).all()


def refresh_incident(db, hotspot):
    """Recompute centroid + ONE priority for the whole incident (worst severity, closest school,
    longest duration, unique citizens) and stamp it on every non-rejected report."""
    cs = db.scalars(select(Complaint).where(Complaint.hotspot_id == hotspot.id, Complaint.status != "rejected")).all()
    if not cs:
        return None
    hotspot.lat = sum(c.lat for c in cs) / len(cs)
    hotspot.lon = sum(c.lon for c in cs) / len(cs)
    severity = max((c.severity for c in cs), key=SEVERITIES.index)
    sens = [c.nearest_sensitive_m for c in cs if c.nearest_sensitive_m is not None]
    durs = [c.duration_hours for c in cs if c.duration_hours is not None]
    _, density = nearest_locality(hotspot.lat, hotspot.lon)
    p = compute_priority(severity, hotspot.category, min(sens) if sens else None,
                         crud.unique_supporters(db, hotspot.id), max(durs) if durs else None, density)
    for c in cs:
        c.priority_score, c.priority_breakdown = p["score"], p["breakdown"]
    return p


def submit_complaint(db, data: ComplaintCreate) -> Complaint:
    if db.get(User, data.reporter_id) is None:
        raise LookupError(f"User {data.reporter_id} not found")

    sens = nearest_sensitive_m(data.lat, data.lon)
    ai = classifier.predict(data.description, sens, data.duration_hours)
    category = ai["category"]
    # When the model is unsure, trust the category the citizen picked.
    if data.user_category in CATEGORIES and ai["category_confidence"] < 0.6:
        category = data.user_category
    locality, _ = nearest_locality(data.lat, data.lon)

    hotspot = find_matching_hotspot(data.lat, data.lon, category, _open_hotspots(db))
    if hotspot is None:
        hotspot = Hotspot(category=category, lat=data.lat, lon=data.lon)
        db.add(hotspot)
        db.flush()

    c = Complaint(
        reporter_id=data.reporter_id, description=data.description, media_url=data.media_url,
        lat=data.lat, lon=data.lon, locality=locality, nearest_sensitive_m=sens,
        duration_hours=data.duration_hours, user_category=data.user_category,
        category=category, category_confidence=ai["category_confidence"],
        severity=ai["severity"], severity_confidence=ai["severity_confidence"],
        needs_human_review=ai["needs_human_review"], hotspot_id=hotspot.id,
        department_id=crud.department_id_for(db, ROUTING[category]),   # simulated routing
    )
    db.add(c)
    db.flush()
    lifecycle.apply_transition(db, c, "under_verification", actor="AirWatch AI",
                               note="AI analysis complete; awaiting human verification")
    refresh_incident(db, hotspot)
    db.commit()
    return c
