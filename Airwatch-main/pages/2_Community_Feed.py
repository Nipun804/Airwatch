import streamlit as st

from app.config import CATEGORIES
from app.services.serializers import build_incidents
from frontend.common import CAT_ICON, CAT_LABEL, STATUS_LABEL, current_user, db_session, incident_card, incident_map, setup

setup("Community Feed", "📡")
st.title("📡 Community pollution feed")
st.caption("Verified incidents only. Many reports of the same problem are grouped into one incident, and we show "
           "**unique citizens** separately from **report count** so duplicates can't inflate pressure.")

f = st.columns([2, 2, 2])
cat = f[0].selectbox("Pollution type", ["All"] + CATEGORIES, format_func=lambda x: "All" if x == "All" else f"{CAT_ICON[x]} {CAT_LABEL[x]}")
status = f[1].selectbox("Status", ["All", "assigned", "acknowledged", "in_progress", "resolved", "reopened"],
                        format_func=lambda x: "All" if x == "All" else STATUS_LABEL[x])
sort = f[2].radio("Sort by", ["priority", "recent"], horizontal=True, format_func=str.title)

with db_session() as db:
    incidents = build_incidents(db, verified_only=True, category=None if cat == "All" else cat,
                                status=None if status == "All" else status, sort=sort)
    tab_feed, tab_map = st.tabs([f"📰 Feed ({len(incidents)})", "🗺️ Hotspot map"])
    with tab_feed:
        if not incidents:
            st.info("No verified incidents match these filters yet.")
        for inc in incidents[:25]:
            incident_card(db, inc, current_user(), "feed")
    with tab_map:
        incident_map(incidents, key="feed_map")
