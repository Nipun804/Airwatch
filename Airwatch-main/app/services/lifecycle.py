"""Complaint status state machine with an audit trail."""
from datetime import datetime, timezone

from app.models import Complaint, StatusHistory

STATUSES = ["submitted", "under_verification", "assigned", "acknowledged",
            "in_progress", "resolved", "reopened", "rejected"]

TRANSITIONS = {
    "submitted": {"under_verification"},
    "under_verification": {"assigned", "rejected"},   # rejected = fake / invalid report
    "assigned": {"acknowledged"},
    "acknowledged": {"in_progress"},
    "in_progress": {"resolved"},
    "resolved": {"reopened"},                          # citizens can flag an inadequate fix
    "reopened": {"assigned", "in_progress"},
    "rejected": set(),
}


class InvalidTransition(ValueError):
    pass


def allowed_next(current):
    return sorted(TRANSITIONS.get(current, set()))


def validate_transition(current, new):
    if new not in STATUSES:
        raise InvalidTransition(f"Unknown status '{new}'")
    if new not in TRANSITIONS.get(current, set()):
        raise InvalidTransition(f"Cannot move from '{current}' to '{new}'. Allowed: {allowed_next(current)}")


def apply_transition(db, complaint: Complaint, new_status, actor, note=None):
    """Validates, updates the complaint and logs history. The caller commits."""
    validate_transition(complaint.status, new_status)
    db.add(StatusHistory(complaint_id=complaint.id, from_status=complaint.status,
                         to_status=new_status, actor=actor, note=note))
    complaint.status = new_status
    complaint.updated_at = datetime.now(timezone.utc)
    complaint.resolved_at = complaint.updated_at if new_status == "resolved" else None
    return complaint


def apply_to_incident(db, complaint: Complaint, new_status, actor, note=None) -> int:
    """Status changes apply to the whole incident (hotspot): 50 reports, one resolution.
    The target complaint must have a valid transition (else InvalidTransition). Siblings for which the
    move isn't valid are skipped. 'rejected' (fake report) only ever applies to the one complaint."""
    from sqlalchemy import select
    validate_transition(complaint.status, new_status)
    targets = [complaint]
    if new_status != "rejected" and complaint.hotspot_id:
        targets = db.scalars(select(Complaint).where(Complaint.hotspot_id == complaint.hotspot_id,
                                                      Complaint.status != "rejected")).all()
    changed = 0
    for c in targets:
        if new_status in TRANSITIONS.get(c.status, set()):
            apply_transition(db, c, new_status, actor, note)
            changed += 1
    return changed
