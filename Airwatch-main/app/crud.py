"""Small reusable DB helpers (kept out of the routes so routes, pipeline and tests share one copy)."""
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import DEPARTMENTS
from app.models import Complaint, Department, Support


def seed_departments(db):
    existing = set(db.scalars(select(Department.name)).all())
    for name in DEPARTMENTS:
        if name not in existing:
            db.add(Department(name=name))
    db.commit()


def department_id_for(db, name):
    return db.scalar(select(Department.id).where(Department.name == name))


def add_support(db, complaint_id, user_id) -> bool:
    """One support per user per complaint. Returns False if this user already supported it."""
    db.add(Support(complaint_id=complaint_id, user_id=user_id))
    try:
        db.commit()
        return True
    except IntegrityError:
        db.rollback()
        return False


def incident_citizens(db, hotspot_id) -> set:
    """Distinct citizens behind an incident = reporters + supporters (rejected/fake reports excluded)."""
    ok = Complaint.status != "rejected"
    reporters = db.scalars(select(Complaint.reporter_id).where(Complaint.hotspot_id == hotspot_id, ok)).all()
    supporters = db.scalars(
        select(Support.user_id).join(Complaint, Support.complaint_id == Complaint.id)
        .where(Complaint.hotspot_id == hotspot_id, ok)
    ).all()
    return set(reporters) | set(supporters)


def unique_supporters(db, hotspot_id) -> int:
    return len(incident_citizens(db, hotspot_id))
