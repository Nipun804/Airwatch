from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=200)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    role: str


class DepartmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str


class ComplaintCreate(BaseModel):
    reporter_id: int
    description: str = Field(min_length=5, max_length=2000)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    user_category: str | None = None        # optional category picked by the citizen
    duration_hours: float | None = Field(default=None, ge=0)
    media_url: str | None = None


class HistoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    from_status: str | None
    to_status: str
    actor: str
    note: str | None
    created_at: datetime


class ComplaintOut(BaseModel):
    id: int
    public_id: str
    description: str
    lat: float
    lon: float
    locality: str | None
    category: str
    category_confidence: float | None
    severity: str
    severity_confidence: float | None
    priority_score: float
    priority_level: str
    status: str
    verified: bool
    needs_human_review: bool
    hotspot_id: int | None
    department: str | None
    unique_supporters: int
    nearest_sensitive_m: float | None
    duration_hours: float | None
    media_url: str | None
    created_at: datetime
    updated_at: datetime
    allowed_next_statuses: list[str]


class ComplaintDetail(ComplaintOut):
    priority_breakdown: dict | None
    history: list[HistoryOut]
    ai_disclaimer: str = "AI analysis is a recommendation; human verification is required."


class IncidentOut(BaseModel):
    """One incident = one hotspot = many citizen reports. This is what the public feed shows."""
    hotspot_id: int
    complaint_id: int            # representative (earliest) report
    public_id: str
    description: str
    category: str
    severity: str
    priority_score: float
    priority_level: str
    status: str
    verified: bool
    locality: str | None
    lat: float
    lon: float
    report_count: int
    unique_supporters: int
    department: str | None
    nearest_sensitive_m: float | None
    first_reported_at: datetime
    updated_at: datetime
    resolved_at: datetime | None


class StatusUpdate(BaseModel):
    new_status: str
    actor: str = Field(min_length=1)
    note: str | None = None


class StatusResult(BaseModel):
    public_id: str
    status: str
    updated_reports: int
    allowed_next_statuses: list[str]


class SupportCreate(BaseModel):
    user_id: int


class SupportResult(BaseModel):
    unique_supporters: int
    priority_score: float
    priority_level: str


class ReopenRequest(BaseModel):
    user_id: int
    note: str | None = None
