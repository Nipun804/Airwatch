import streamlit as st

from frontend.common import (CAT_ICON, CAT_LABEL, badge, current_user, db_session, get_aq, get_forecasts,
                             incident_card, priority_badge, setup)
from app.services import aq_data
from app.services.serializers import build_incidents, compute_stats

setup("Home", "🌫️")

st.title("🌫️ AirWatch")
st.markdown('<p class="aw-hero"><b>Community-driven air-quality action platform.</b> Citizens report pollution, AI analyses '
            'and prioritises it, and departments are held accountable for the response.</p>', unsafe_allow_html=True)

with db_session() as db:
    s = compute_stats(db)
    top = build_incidents(db, verified_only=True, sort="priority")[:4]

    m = st.columns(5)
    m[0].metric("Citizen reports", s["total_reports"])
    m[1].metric("Unique incidents", s["total_incidents"])
    m[2].metric("Awaiting verification", s["awaiting_verification"])
    m[3].metric("Resolved", s["resolved_incidents"])
    m[4].metric("Resolution rate", f"{s['resolution_rate']:.0%}")

    st.divider()
    st.subheader("🌫️ Air-quality outlook (next 48 h)")
    try:
        _df, _meta = get_aq()
        _fc = get_forecasts()
        _peak = _fc.loc[_fc.groupby("location").pm25.idxmax()].sort_values("pm25", ascending=False).head(3)
        for col, r in zip(st.columns(3), _peak.itertuples()):
            cat, color = aq_data.pm25_category(r.pm25)
            col.metric(f"Peak forecast · {r.location}", f"{r.pm25:.0f} µg/m³", f"{r.time:%a %I %p}", delta_color="off")
            col.markdown(badge(cat, color), unsafe_allow_html=True)
        st.caption(("Simulated data - " if not aq_data.is_real(_meta) else "") + "PM2.5 forecast; see the Air Quality Forecast page for details.")
    except Exception:       # never let the outlook strip break the home page
        st.caption("Air-quality outlook unavailable right now.")

    st.divider()
    st.subheader("How it works")
    a, b, c = st.columns(3)
    a.markdown("### 1️⃣ Citizens report\nPhoto, location and a short description in **English, Hinglish or Hindi**. "
               "Many reports of the same fire become **one incident**.")
    b.markdown("### 2️⃣ AI analyses\nClassifies the pollution type and severity, checks distance to **schools & hospitals**, "
               "and computes an **explainable 0–100 priority**.")
    c.markdown("### 3️⃣ Authorities act\nHuman verification, routing to the right department, status updates, "
               "and **citizens can reopen** a fix that didn't work.")

    st.divider()
    st.subheader("🚨 Highest-priority open incidents")
    top_open = [i for i in top if i.status != "resolved"] or top
    for inc in top_open[:3]:
        incident_card(db, inc, current_user(), "home", actions_enabled=False)
    st.caption("Use the sidebar to report pollution, browse the live feed and map, act as a moderator/department, "
               "view the accountability scoreboard, or inspect the ML model performance.")
