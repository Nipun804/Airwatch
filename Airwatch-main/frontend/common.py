"""Shared Streamlit helpers: page setup, sign-in, cards, maps. The UI calls the same service layer as the API."""
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from sqlalchemy import select

from app import crud
from app.config import ARTIFACTS_DIR, DATA_DIR, ROOT
from app.database import SessionLocal, init_db
from app.models import User
from app.services import actions, aq_data, classifier, forecast
from app.services.demo import seed_demo
from app.services.lifecycle import allowed_next
from app.services.serializers import complaint_detail

UPLOAD_DIR = ROOT / "uploads"

CAT_LABEL = {"garbage_burning": "Garbage burning", "construction_dust": "Construction dust",
             "industrial_smoke": "Industrial smoke", "vehicle_emission": "Vehicle emissions",
             "stubble_burning": "Stubble burning"}
CAT_ICON = {"garbage_burning": "🔥", "construction_dust": "🏗️", "industrial_smoke": "🏭",
            "vehicle_emission": "🚛", "stubble_burning": "🌾"}
STATUS_LABEL = {"submitted": "Submitted", "under_verification": "Under verification", "assigned": "Assigned",
                "acknowledged": "Acknowledged", "in_progress": "Action in progress", "resolved": "Resolved",
                "reopened": "Reopened", "rejected": "Rejected"}
STATUS_COLOR = {"submitted": "#6b7280", "under_verification": "#6b7280", "assigned": "#2563eb",
                "acknowledged": "#7c3aed", "in_progress": "#d97706", "resolved": "#16a34a",
                "reopened": "#dc2626", "rejected": "#6b7280"}
LEVEL_COLOR = {"critical": "#d62728", "high": "#f97316", "medium": "#d4a000", "low": "#2ca02c"}
ACTION_LABEL = {"assigned": "✅ Verify & assign", "acknowledged": "👍 Acknowledge", "in_progress": "🛠️ Start action",
                "resolved": "✔️ Mark resolved", "rejected": "🚫 Reject report", "reopened": "↩️ Reopen"}

_CSS = """<style>
.aw-badge{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-size:.78rem;font-weight:600;margin-right:6px}
.aw-hero{font-size:1.05rem;opacity:.85}
</style>"""


# ---------------------------------------------------------------- bootstrap
def ensure_models():
    """Train on the fly if artifacts are missing or incompatible with this environment (e.g. Streamlit Cloud)."""
    if not classifier.models_available():
        if not (DATA_DIR / "complaints.csv").exists():
            subprocess.run([sys.executable, "-m", "ml.generate_data"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-m", "ml.train"], cwd=ROOT, check=True)
        classifier._load.cache_clear()
    if not forecast.models_available():
        subprocess.run([sys.executable, "-m", "ml.train_forecast"], cwd=ROOT, check=True)
        forecast._load.cache_clear()


@st.cache_resource(show_spinner="Starting AirWatch (first run trains the models if needed)…")
def _bootstrap():
    init_db()
    ensure_models()
    with SessionLocal() as db:
        crud.seed_departments(db)
        seed_demo(db)          # no-op if the DB already has data
    return True


@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def setup(title, icon="🌫️", require_login=False):
    st.set_page_config(page_title=f"{title} · AirWatch", page_icon=icon, layout="wide")
    _bootstrap()
    st.markdown(_CSS, unsafe_allow_html=True)
    _sidebar()
    if require_login and current_user() is None:
        st.title(f"{icon} {title}")
        st.info("👈 Please sign in from the sidebar first (or use *Quick demo login*).")
        st.stop()


# ---------------------------------------------------------------- identity (no real auth in the MVP)
def current_user():
    return st.session_state.get("user")


def _login(user: User):
    st.session_state["user"] = {"id": user.id, "name": user.name, "email": user.email}


def _sidebar():
    with st.sidebar:
        st.markdown("## 🌫️ AirWatch")
        st.caption("Make pollution visible, complaints actionable, resolution measurable.")
        user = current_user()
        if user:
            st.success(f"Signed in as **{user['name']}**")
            if st.button("Sign out", key="signout"):
                st.session_state.pop("user", None)
                st.session_state.pop("last_submission", None)
                st.rerun()
            return
        with st.form("login_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            if st.form_submit_button("Continue"):
                if name.strip() and "@" in email:
                    with db_session() as db:
                        _login(actions.get_or_create_user(db, name.strip(), email.strip().lower()))
                    st.rerun()
                else:
                    st.error("Enter a name and a valid email.")
        with st.expander("Quick demo login"):
            with db_session() as db:
                demo = db.scalars(select(User).where(User.email.like("%@demo.com")).order_by(User.id)).all()
                options = {u.name: u for u in demo}
            if options:
                pick = st.selectbox("Demo citizen", list(options), key="demo_pick")
                if st.button("Log in", key="demo_login"):
                    _login(options[pick])
                    st.rerun()
        st.caption("Hackathon demo: sign-in is simulated (no passwords).")


# ---------------------------------------------------------------- small UI bits
def badge(text, color):
    return f'<span class="aw-badge" style="background:{color}">{text}</span>'


def priority_badge(level, score):
    return badge(f"{level.upper()} · {score:.0f}/100", LEVEL_COLOR[level])


def status_badge(status):
    return badge(STATUS_LABEL.get(status, status), STATUS_COLOR.get(status, "#6b7280"))


def time_ago(dt):
    dt = dt.replace(tzinfo=None)
    secs = max(0, (datetime.now(timezone.utc).replace(tzinfo=None) - dt).total_seconds())
    if secs < 90:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


def save_upload(uploaded):
    UPLOAD_DIR.mkdir(exist_ok=True)
    name = f"{uuid4().hex}{Path(uploaded.name).suffix.lower()}"
    (UPLOAD_DIR / name).write_bytes(uploaded.getbuffer())
    return f"uploads/{name}"


def show_media(media_url, width=280):
    if media_url and (ROOT / media_url).exists():
        st.image(str(ROOT / media_url), width=width, caption="Citizen evidence")


def breakdown_chart(breakdown):
    import pandas as pd
    if not breakdown:
        return
    labels = {"severity": "Severity", "sensitive_proximity": "Near school/hospital", "unique_supporters": "Citizens affected",
              "category_risk": "Pollution type risk", "duration": "Duration", "population_density": "Population density"}
    df = pd.DataFrame({"factor": [labels.get(k, k) for k in breakdown], "points": [v["points"] for v in breakdown.values()]})
    st.bar_chart(df, x="factor", y="points", height=240)
    st.caption("Points contributed to the 0–100 priority score. Weighted formula - every factor is visible.")


def timeline(history):
    for h in history:
        note = f" — {h.note}" if h.note else ""
        st.markdown(f"- **{STATUS_LABEL.get(h.to_status, h.to_status)}** · {h.actor} · {time_ago(h.created_at)}{note}")


# ---------------------------------------------------------------- incident card (feed + dashboards)
def incident_card(db, inc, user, prefix, actions_enabled=True):
    with st.container(border=True):
        head = st.columns([5, 3])
        head[0].markdown(f"#### {CAT_ICON.get(inc.category, '')} {CAT_LABEL.get(inc.category, inc.category)} · {inc.locality}")
        head[1].markdown(priority_badge(inc.priority_level, inc.priority_score) + status_badge(inc.status),
                         unsafe_allow_html=True)
        st.write(inc.description)
        dept = f" · routed to **{inc.department}**" if inc.department else ""
        near = f" · 🏫 school/hospital **{inc.nearest_sensitive_m:.0f} m** away" if inc.nearest_sensitive_m and inc.nearest_sensitive_m < 1000 else ""
        st.caption(f"{inc.public_id} · first reported {time_ago(inc.first_reported_at)} · **{inc.report_count} report(s)** from "
                   f"**{inc.unique_supporters} unique citizen(s)**{dept}{near}")
        if actions_enabled:
            cols = st.columns([2, 2, 6])
            citizens = crud.incident_citizens(db, inc.hotspot_id)
            already = bool(user and user["id"] in citizens)
            if cols[0].button("✅ Counted" if already else "🙋 I see this too", key=f"{prefix}_sup_{inc.hotspot_id}",
                              disabled=already or user is None, help=None if user else "Sign in to support"):
                try:
                    r = actions.support(db, inc.complaint_id, user["id"])
                    st.toast(f"Thanks! {r['unique_supporters']} citizens now backing this incident.")
                    st.rerun()
                except actions.ActionError as e:
                    st.error(str(e))
            if inc.status == "resolved" and already and cols[1].button("↩️ Not fixed", key=f"{prefix}_re_{inc.hotspot_id}"):
                try:
                    actions.reopen(db, inc.complaint_id, user["id"], "Citizen says the issue persists")
                    st.toast("Incident reopened for the department.")
                    st.rerun()
                except actions.ActionError as e:
                    st.error(str(e))
        with st.expander("Details, AI priority breakdown & timeline"):
            from app.models import Complaint
            rep = db.get(Complaint, inc.complaint_id)
            d = complaint_detail(db, rep)
            left, right = st.columns(2)
            with left:
                st.markdown("**Status timeline**")
                timeline(d.history)
                show_media(d.media_url)
            with right:
                st.markdown(f"**AI analysis** · category confidence {d.category_confidence or 0:.0%}, "
                            f"severity *{d.severity}* ({d.severity_confidence or 0:.0%})")
                breakdown_chart(d.priority_breakdown)


# ---------------------------------------------------------------- map
def incident_map(incidents, height=520, key="map"):
    import folium
    from streamlit_folium import st_folium
    if incidents:
        lat = sum(i.lat for i in incidents) / len(incidents)
        lon = sum(i.lon for i in incidents) / len(incidents)
    else:
        lat, lon = 28.62, 77.22
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="OpenStreetMap")
    for i in incidents:
        folium.CircleMarker(
            [i.lat, i.lon], radius=7 + min(14, 3 * (i.report_count ** 0.5)), color=LEVEL_COLOR[i.priority_level],
            fill=True, fill_opacity=0.65, weight=2,
            popup=folium.Popup(f"<b>{CAT_LABEL.get(i.category, i.category)}</b><br>{i.locality}<br>"
                               f"Priority {i.priority_score:.0f}/100 ({i.priority_level})<br>"
                               f"{STATUS_LABEL.get(i.status, i.status)}<br>{i.report_count} reports · "
                               f"{i.unique_supporters} citizens", max_width=250),
            tooltip=f"{CAT_LABEL.get(i.category, i.category)} · {i.locality}").add_to(m)
    st_folium(m, height=height, use_container_width=True, returned_objects=[], key=key)
    st.caption("Circle size = number of reports · colour = priority "
               "(🔴 critical, 🟠 high, 🟡 medium, 🟢 low)")


# ---------------------------------------------------------------- air quality (forecast / hotspots / sources)
SOURCE_COLOR = {"Waste burning": "#e4572e", "Construction & dust": "#b08968", "Industry": "#6d597a",
                "Traffic / vehicles": "#2a9d8f", "Crop-residue burning": "#e9c46a"}


@st.cache_data(ttl=3600, show_spinner="Loading air-quality data…")
def get_aq():
    return aq_data.load_air_quality()


@st.cache_data(ttl=3600, show_spinner="Running the 48-hour forecast…")
def get_forecasts():
    df, meta = get_aq()
    return forecast.forecast_all(df, meta["t0"])


def aq_banner(meta):
    if aq_data.is_real(meta):
        st.info(f"🛰️ Data: {meta['label']} · as of {meta['t0']:%d %b %Y, %H:%M} IST")
    else:
        st.warning("**Simulated air-quality data.** The structure is realistic but the values are not measurements. "
                   "To use real hourly data run `python -m scripts.fetch_open_meteo` then `python -m ml.train_forecast`.")


def band_legend():
    chips = " ".join(badge(f"{lab} ≤ {ub:g}" if ub != float("inf") else f"{lab} > 250", col)
                     for ub, lab, col in aq_data.PM25_BANDS)
    st.markdown(chips, unsafe_allow_html=True)
    st.caption("PM2.5 µg/m³ · CPCB National AQI concentration bands (defined for 24-h averages; applied to hourly values as an approximation).")


def aq_map(table, incidents=None, show_bubbles=True, show_heat=True, show_incidents=True, height=560, key="aq_map"):
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    m = folium.Map(location=[table.lat.mean(), table.lon.mean()], zoom_start=10, tiles="OpenStreetMap")
    if show_heat:
        HeatMap([[r.lat, r.lon, min(1.0, r.pm25 / 250)] for r in table.itertuples()], radius=48, blur=38, min_opacity=0.25,
                gradient={0.2: "#2ca02c", 0.4: "#e6b800", 0.6: "#ff7f0e", 0.8: "#d62728", 1.0: "#7f1d1d"}).add_to(m)
    if show_bubbles:
        for r in table.itertuples():
            html = (f'<div style="background:{r.color};color:#fff;border:2px solid #fff;border-radius:50%;width:40px;'
                    f'height:40px;line-height:36px;text-align:center;font-weight:700;font-size:13px;'
                    f'box-shadow:0 0 5px rgba(0,0,0,.55)">{r.pm25:.0f}</div>')
            folium.Marker([r.lat, r.lon], icon=folium.DivIcon(html=html, icon_size=(40, 40), icon_anchor=(20, 20)),
                          tooltip=f"{r.location}: {r.pm25:.0f} µg/m³ ({r.category})",
                          popup=folium.Popup(f"<b>{r.location}</b><br>PM2.5: {r.pm25:.0f} µg/m³ ({r.category})<br>"
                                             f"Elevated {r.exceed_7d:.0%} of the last 7 days<br>Open citizen reports: {r.open_incidents}<br>"
                                             f"Hotspot index: {r.hotspot_index:.0f}/100", max_width=260)).add_to(m)
    if show_incidents:
        for i in incidents or []:
            if i.status in ("rejected",):
                continue
            folium.CircleMarker([i.lat, i.lon], radius=6, color="#111", weight=2, fill=True, fill_color=LEVEL_COLOR[i.priority_level],
                                fill_opacity=0.95, tooltip=f"📍 {CAT_LABEL.get(i.category, i.category)} · {i.locality} · "
                                                           f"{STATUS_LABEL.get(i.status, i.status)} ({i.report_count} reports)").add_to(m)
    st_folium(m, height=height, use_container_width=True, returned_objects=[], key=key)
