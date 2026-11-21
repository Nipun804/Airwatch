from collections import defaultdict

from sqlalchemy import select

from app import crud
from app.models import Complaint, Department, Hotspot
from app.schemas import ComplaintDetail, ComplaintOut, HistoryOut, IncidentOut
from app.services.lifecycle import allowed_next
from app.services.priority import level_for

UNVERIFIED = ("submitted", "under_verification")


def _dept_name(db, dept_id):
    d = db.get(Department, dept_id) if dept_id else None
    return d.name if d else None


def complaint_out(db, c: Complaint) -> ComplaintOut:
    return ComplaintOut(
        id=c.id, public_id=c.public_id, description=c.description, lat=c.lat, lon=c.lon, locality=c.locality,
        category=c.category, category_confidence=c.category_confidence, severity=c.severity,
        severity_confidence=c.severity_confidence, priority_score=c.priority_score,
        priority_level=level_for(c.priority_score), status=c.status, verified=c.status not in UNVERIFIED + ("rejected",),
        needs_human_review=c.needs_human_review, hotspot_id=c.hotspot_id, department=_dept_name(db, c.department_id),
        unique_supporters=crud.unique_supporters(db, c.hotspot_id) if c.hotspot_id else 1,
        nearest_sensitive_m=c.nearest_sensitive_m, duration_hours=c.duration_hours, media_url=c.media_url,
        created_at=c.created_at, updated_at=c.updated_at, allowed_next_statuses=allowed_next(c.status))


def complaint_detail(db, c: Complaint) -> ComplaintDetail:
    base = complaint_out(db, c).model_dump()
    return ComplaintDetail(**base, priority_breakdown=c.priority_breakdown,
                           history=[HistoryOut.model_validate(h) for h in c.history])


def build_incidents(db, verified_only=False, category=None, status=None, sort="priority"):
    complaints = db.scalars(select(Complaint).where(
        Complaint.hotspot_id.is_not(None), Complaint.status != "rejected").order_by(Complaint.created_at, Complaint.id)).all()
    groups = defaultdict(list)
    for c in complaints:
        groups[c.hotspot_id].append(c)

    out = []
    for hid, cs in groups.items():
        rep, h = cs[0], db.get(Hotspot, hid)
        verified = rep.status not in UNVERIFIED
        if (verified_only and not verified) or (category and h.category != category) or (status and rep.status != status):
            continue
        sens = [c.nearest_sensitive_m for c in cs if c.nearest_sensitive_m is not None]
        out.append(IncidentOut(
            hotspot_id=hid, complaint_id=rep.id, public_id=rep.public_id, description=rep.description,
            category=h.category, severity=max((c.severity for c in cs), key=["low", "medium", "high", "critical"].index),
            priority_score=rep.priority_score, priority_level=level_for(rep.priority_score), status=rep.status,
            verified=verified, locality=rep.locality, lat=h.lat, lon=h.lon, report_count=len(cs),
            unique_supporters=crud.unique_supporters(db, hid), department=_dept_name(db, rep.department_id),
            nearest_sensitive_m=min(sens) if sens else None, first_reported_at=rep.created_at,
            updated_at=max(c.updated_at for c in cs), resolved_at=rep.resolved_at))
    if sort == "recent":
        out.sort(key=lambda i: i.first_reported_at.replace(tzinfo=None), reverse=True)
    else:
        out.sort(key=lambda i: i.priority_score, reverse=True)
    return out


OPEN_FOR_DEPT = {"assigned", "acknowledged", "in_progress", "reopened"}


def department_queue(db, department_name):
    return [i for i in build_incidents(db, verified_only=True)
            if i.department == department_name and i.status in OPEN_FOR_DEPT]


def compute_stats(db):
    """Community Action Scoreboard (incident level; rejected reports excluded)."""
    from collections import Counter
    inc = build_incidents(db)
    verified = [i for i in inc if i.verified]
    resolved = [i for i in verified if i.status == "resolved"]
    hours = [(i.resolved_at.replace(tzinfo=None) - i.first_reported_at.replace(tzinfo=None)).total_seconds() / 3600
             for i in resolved if i.resolved_at]
    by_dept = defaultdict(lambda: {"open": 0, "resolved": 0})
    for i in verified:
        by_dept[i.department or "unassigned"]["resolved" if i.status == "resolved" else "open"] += 1
    return {
        "total_reports": sum(i.report_count for i in inc),
        "total_incidents": len(inc),
        "awaiting_verification": len(inc) - len(verified),
        "verified_incidents": len(verified),
        "resolved_incidents": len(resolved),
        "resolution_rate": round(len(resolved) / len(verified), 3) if verified else 0.0,
        "avg_resolution_hours": round(sum(hours) / len(hours), 2) if hours else None,
        "by_status": dict(Counter(i.status for i in inc)),
        "by_category": dict(Counter(i.category for i in inc)),
        "unresolved_by_locality": Counter(i.locality for i in verified if i.status != "resolved").most_common(10),
        "by_department": dict(by_dept),
    }
