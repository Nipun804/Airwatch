"""Business actions shared by the FastAPI routes and the Streamlit app."""
from sqlalchemy import select

from app import crud
from app.models import Complaint, Hotspot, User
from app.services import lifecycle, pipeline


class ActionError(Exception):
    """Carries an HTTP-style status code so the API can map it and the UI can show it."""
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def get_complaint(db, complaint_id) -> Complaint:
    c = db.get(Complaint, complaint_id)
    if c is None:
        raise ActionError(404, "Complaint not found")
    return c


def get_or_create_user(db, name, email) -> User:
    """Idempotent by email. NOTE: no real authentication in the hackathon MVP."""
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(name=name, email=email)
        db.add(user)
        db.commit()
    return user


def support(db, complaint_id, user_id) -> dict:
    """'I see this too.' Each citizen counts once per incident, so duplicates can't inflate pressure."""
    c = get_complaint(db, complaint_id)
    if c.status == "rejected":
        raise ActionError(400, "This report was rejected")
    if db.get(User, user_id) is None:
        raise ActionError(404, "User not found")
    if user_id in crud.incident_citizens(db, c.hotspot_id):
        raise ActionError(409, "You have already reported or supported this incident")
    crud.add_support(db, c.id, user_id)
    p = pipeline.refresh_incident(db, db.get(Hotspot, c.hotspot_id))
    db.commit()
    return {"unique_supporters": crud.unique_supporters(db, c.hotspot_id),
            "priority_score": p["score"], "priority_level": p["level"]}


def change_status(db, complaint_id, new_status, actor, note=None):
    """Moves the whole incident. Returns (complaint, number_of_reports_updated)."""
    c = get_complaint(db, complaint_id)
    try:
        n = lifecycle.apply_to_incident(db, c, new_status, actor, note)
    except lifecycle.InvalidTransition as e:
        raise ActionError(409, str(e))
    if c.hotspot_id:
        pipeline.refresh_incident(db, db.get(Hotspot, c.hotspot_id))
    db.commit()
    return c, n


def reopen(db, complaint_id, user_id, note=None):
    """A citizen involved in the incident flags that 'resolved' wasn't really fixed."""
    c = get_complaint(db, complaint_id)
    if user_id not in crud.incident_citizens(db, c.hotspot_id):
        raise ActionError(403, "Only citizens who reported or supported this incident can reopen it")
    return change_status(db, complaint_id, "reopened", f"citizen:{user_id}", note or "Citizen says issue persists")
