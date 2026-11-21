import pandas as pd
import streamlit as st

from app.services.serializers import compute_stats
from frontend.common import CAT_LABEL, STATUS_LABEL, db_session, setup

setup("Scoreboard", "📊")
st.title("📊 Community action scoreboard")
st.caption("Accountability at department and locality level. No individual officers are named or targeted.")

with db_session() as db:
    s = compute_stats(db)

m = st.columns(5)
m[0].metric("Citizen reports", s["total_reports"])
m[1].metric("Verified incidents", s["verified_incidents"])
m[2].metric("Resolved", s["resolved_incidents"])
m[3].metric("Resolution rate", f"{s['resolution_rate']:.0%}")
m[4].metric("Avg. time to resolve", f"{s['avg_resolution_hours']:.0f} h" if s["avg_resolution_hours"] is not None else "n/a")

a, b = st.columns(2)
with a:
    st.subheader("Incidents by status")
    st.bar_chart(pd.DataFrame({"incidents": {STATUS_LABEL.get(k, k): v for k, v in s["by_status"].items()}}))
    st.subheader("Unresolved incidents by locality")
    if s["unresolved_by_locality"]:
        st.bar_chart(pd.DataFrame(s["unresolved_by_locality"], columns=["locality", "unresolved"]).set_index("locality"))
    else:
        st.success("No unresolved verified incidents.")
with b:
    st.subheader("Incidents by pollution type")
    st.bar_chart(pd.DataFrame({"incidents": {CAT_LABEL.get(k, k): v for k, v in s["by_category"].items()}}))
    st.subheader("Department performance")
    if s["by_department"]:
        st.bar_chart(pd.DataFrame(s["by_department"]).T[["open", "resolved"]])
