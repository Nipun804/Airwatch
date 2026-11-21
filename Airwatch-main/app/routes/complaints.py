from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Complaint
from app.schemas import (ComplaintCreate, ComplaintDetail, ComplaintOut, ReopenRequest, StatusResult,
                         StatusUpdate, SupportCreate, SupportResult)
from app.services import actions, lifecycle, pipeline
from app.services.serializers import complaint_detail, complaint_out

router = APIRouter(prefix="/complaints", tags=["complaints"])


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except actions.ActionError as e:
        raise HTTPException(e.status_code, str(e))


@router.post("", response_model=ComplaintDetail, status_code=201)
def submit(data: ComplaintCreate, db: Session = Depends(get_db)):
    """Citizen submits a report: AI classifies it, de-duplicates it into an incident, scores priority,
    routes it to a (simulated) department and queues it for human verification."""
    try:
        c = pipeline.submit_complaint(db, data)
    except LookupError as e:
        raise HTTPException(404, str(e))
    return complaint_detail(db, c)


@router.get("", response_model=list[ComplaintOut])
def list_complaints(status: str | None = None, category: str | None = None, hotspot_id: int | None = None,
                    reporter_id: int | None = None, sort: str = Query("recent", pattern="^(recent|priority)$"),
                    limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Raw report list (my complaints / admin / debugging). The public feed is GET /feed."""
    q = select(Complaint)
    for col, val in ((Complaint.status, status), (Complaint.category, category),
                     (Complaint.hotspot_id, hotspot_id), (Complaint.reporter_id, reporter_id)):
        if val is not None:
            q = q.where(col == val)
    q = q.order_by(Complaint.priority_score.desc() if sort == "priority" else Complaint.created_at.desc()).limit(limit)
    return [complaint_out(db, c) for c in db.scalars(q).all()]


@router.get("/{complaint_id}", response_model=ComplaintDetail)
def get_complaint(complaint_id: int, db: Session = Depends(get_db)):
    return complaint_detail(db, _guard(actions.get_complaint, db, complaint_id))


@router.post("/{complaint_id}/support", response_model=SupportResult)
def support(complaint_id: int, data: SupportCreate, db: Session = Depends(get_db)):
    return SupportResult(**_guard(actions.support, db, complaint_id, data.user_id))


@router.patch("/{complaint_id}/status", response_model=StatusResult)
def update_status(complaint_id: int, data: StatusUpdate, db: Session = Depends(get_db)):
    """Verification / department updates; moves the whole incident. MVP: no auth, `actor` is a label."""
    c, n = _guard(actions.change_status, db, complaint_id, data.new_status, data.actor, data.note)
    return StatusResult(public_id=c.public_id, status=c.status, updated_reports=n,
                        allowed_next_statuses=lifecycle.allowed_next(c.status))


@router.post("/{complaint_id}/reopen", response_model=StatusResult)
def reopen(complaint_id: int, data: ReopenRequest, db: Session = Depends(get_db)):
    c, n = _guard(actions.reopen, db, complaint_id, data.user_id, data.note)
    return StatusResult(public_id=c.public_id, status=c.status, updated_reports=n,
                        allowed_next_statuses=lifecycle.allowed_next(c.status))
