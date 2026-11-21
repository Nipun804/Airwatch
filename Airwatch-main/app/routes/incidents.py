"""Public feed, hotspot map data, department queue and accountability stats."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Department
from app.schemas import IncidentOut
from app.services.serializers import build_incidents, compute_stats, department_queue

router = APIRouter(tags=["feed & accountability"])
SORT = Query("priority", pattern="^(priority|recent)$")


@router.get("/feed", response_model=list[IncidentOut])
def feed(category: str | None = None, status: str | None = None, sort: str = SORT,
         verified_only: bool = True, limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Public community feed: one card per incident (not per report), verified incidents by default."""
    return build_incidents(db, verified_only, category, status, sort)[:limit]


@router.get("/hotspots", response_model=list[IncidentOut])
def hotspots(category: str | None = None, verified_only: bool = False, db: Session = Depends(get_db)):
    """Map data: every incident with coordinates, report count and unique supporters."""
    return build_incidents(db, verified_only, category)


@router.get("/departments/{department_id}/queue", response_model=list[IncidentOut])
def department_queue_route(department_id: int, db: Session = Depends(get_db)):
    """Simulated department dashboard: open incidents routed to this department, highest priority first."""
    dept = db.get(Department, department_id)
    if dept is None:
        raise HTTPException(404, "Department not found")
    return department_queue(db, dept.name)


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    """Community Action Scoreboard (incident level; rejected reports excluded)."""
    return compute_stats(db)
