"""Demo data: runs the REAL pipeline on a set of realistic reports, then moves incidents through the
lifecycle and back-dates timestamps so the feed, map and scoreboard look alive on first load."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app import crud
from app.config import LOCALITIES
from app.models import Complaint, StatusHistory, User
from app.schemas import ComplaintCreate
from app.services import lifecycle, pipeline
from app.services.geo import load_places, offset_latlon

PATH = ["assigned", "acknowledged", "in_progress", "resolved"]


def _near_school(locality, metres=90):
    places = load_places()
    row = places[places.name.str.startswith(locality) & (places.type == "school")].iloc[0]
    return offset_latlon(row.lat, row.lon, metres, 20)


def _incidents():
    sx = _near_school("Dwarka")
    L = lambda n: LOCALITIES[n][:2]
    return [  # (reports[(user, text, (lat, lon), hours)], final_status, age_hours, supporters)
        ([(1, "Garbage is being burned near Sector 62 Noida. Thick smoke spreading, children coughing.", L("Sector 62 Noida"), 6),
          (2, "kachra jalaya ja raha hai Sector 62 ke paas, bahut dhuan hai", (28.6272, 77.3650), 5),
          (3, "कचरा जलाया जा रहा है, सांस लेना मुश्किल हो रहा है", (28.6269, 77.3647), 4)], "resolved", 50, []),
        ([(2, "Heavy dust from building construction in Indirapuram. Please look into it.", L("Indirapuram"), 20)],
         "assigned", 20, [1, 3]),
        ([(4, "Factory chimney releasing black smoke in Okhla. Huge amount of smoke, people coughing.", L("Okhla"), 30),
          (5, "factory ki chimney se kaala dhuan nikal raha hai Okhla mein, saans lena mushkil", (28.5357, 77.2752), 28)],
         "in_progress", 30, [1]),
        ([(5, "Trucks emitting black smoke near Anand Vihar. Minor issue but please check.", L("Anand Vihar"), 2)],
         "acknowledged", 8, []),
        ([(1, "खेत में पराली जलाई जा रही है Narela के पास", L("Narela"), 12)], "under_verification", 1, []),
        ([(3, "Open garbage burning beside the school. Thick smoke spreading, children coughing and eyes burning.", sx, 8),
          (5, "kachra jalaya ja raha hai school ke paas, bachche khaans rahe hain", (sx[0] + 0.0002, sx[1]), 7)],
         "assigned", 4, [1, 2, 4]),
        ([(4, "Construction site throwing dust everywhere in Lajpat Nagar. The smell is noticeable.", L("Lajpat Nagar"), 15)],
         "assigned", 15, []),
        ([(2, "Factory is emitting toxic fumes in Bawana. Burning something at night.", L("Bawana"), 26),
          (3, "फैक्ट्री की चिमनी से काला धुआं निकल रहा है Bawana", (28.7991, 77.0331), 24)], "assigned", 26, []),
        ([(1, "Old diesel vehicles polluting in Rohini. Not too much smoke yet.", L("Rohini"), 3)],
         "under_verification", 0.5, []),
        ([(5, "Small trash fire at Mayur Vihar, plastic being burned. Minor issue but please check.", L("Mayur Vihar"), 5)], "resolved", 70, []),
    ]


def _backdate(db, complaints, age_hours):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=age_hours)
    for c in complaints:
        hist = db.scalars(select(StatusHistory).where(StatusHistory.complaint_id == c.id).order_by(StatusHistory.id)).all()
        c.created_at = start
        step = (age_hours * 0.85) / max(len(hist), 1)
        for k, h in enumerate(hist):
            h.created_at = start + timedelta(hours=step * k)
        last = hist[-1].created_at if hist else start
        c.updated_at = last
        c.resolved_at = last if c.status == "resolved" else None


def seed_demo(db, n_users=5):
    if db.scalar(select(Complaint.id).limit(1)) is not None:
        return False
    crud.seed_departments(db)
    for i in range(1, n_users + 1):
        if not db.scalar(select(User).where(User.email == f"citizen{i}@demo.com")):
            db.add(User(name=f"Citizen {i}", email=f"citizen{i}@demo.com"))
    db.commit()
    for reports, final, age, supporters in _incidents():
        made = [pipeline.submit_complaint(db, ComplaintCreate(reporter_id=u, description=t, lat=ll[0], lon=ll[1],
                                                              duration_hours=h)) for u, t, ll, h in reports]
        first = db.get(Complaint, made[0].id)
        for sid in supporters:
            if sid not in crud.incident_citizens(db, first.hotspot_id):
                crud.add_support(db, first.id, sid)
        pipeline.refresh_incident(db, first.hotspot)
        for status in PATH[:PATH.index(final) + 1] if final in PATH else []:
            lifecycle.apply_to_incident(db, first, status, "demo-officer", f"{status.replace('_', ' ').title()} (demo data)")
        db.commit()
        _backdate(db, [db.get(Complaint, m.id) for m in made], age)
        db.commit()
    return True
