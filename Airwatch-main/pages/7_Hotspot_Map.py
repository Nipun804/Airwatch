import pandas as pd
import streamlit as st

from app.services import aq_data, aq_hotspots, sources
from app.services.serializers import build_incidents
from frontend.common import (CAT_LABEL, STATUS_LABEL, aq_banner, aq_map, band_legend, badge, db_session, get_aq,
                             get_forecasts, setup)

setup("Hotspot Map", "🗺️")
st.title("🗺️ Pollution hotspot map")
st.caption("Where air quality is **repeatedly** elevated, combined with what citizens are reporting on the ground. "
           "Use it to decide which locations need inspection first.")

df, meta = get_aq()
fc_all, t0 = get_forecasts(), meta["t0"]
aq_banner(meta)

c = st.columns([3, 3, 4])
when = c[0].radio("Show PM2.5", ["Now", "In 24 h (forecast)", "In 48 h (forecast)"], horizontal=False)
thr = c[1].select_slider("'Elevated' means above", options=[60, 90, 120, 250], value=120,
                         format_func=lambda v: f"{v} µg/m³ ({aq_data.pm25_category(v)[0]})")
with c[2]:
    st.markdown("**Layers**")
    l1, l2, l3 = st.columns(3)
    bubbles, heat, inc_on = l1.checkbox("PM2.5 bubbles", True), l2.checkbox("Heat layer", True), l3.checkbox("Citizen reports", True)

h = {"Now": 0, "In 24 h (forecast)": 24, "In 48 h (forecast)": 48}[when]
with db_session() as db:
    incidents = build_incidents(db, verified_only=False)
attribution = sources.attribute_sources(df, t0, incidents)
table = aq_hotspots.station_table(df, t0, incidents, fc_all, h, float(thr), attribution=attribution)

aq_map(table, incidents, bubbles, heat, inc_on, key=f"hotspot_{h}")
band_legend()
st.caption("Bubble = PM2.5 µg/m³ at the monitoring location · dots = citizen incidents, coloured by AI priority.")

st.subheader("🔎 Inspect first")
top = table.head(3)
cols = st.columns(3)
for col, r in zip(cols, top.itertuples()):
    with col, st.container(border=True):
        st.markdown(f"### {r.location}")
        st.markdown(badge(f"{r.category} · {r.pm25:.0f}", r.color) + badge(f"index {r.hotspot_index:.0f}/100", "#334155"),
                    unsafe_allow_html=True)
        driver = sources.SOURCE_LABEL.get(r.likely_driver, "unknown")
        st.write(f"Above {thr} µg/m³ **{r.exceed_7d:.0%}** of the last 7 days · **{r.open_incidents}** open citizen report(s) "
                 f"nearby · likely driver: **{driver}**")

st.subheader("All monitored areas")
show = table.assign(driver=table.likely_driver.map(sources.SOURCE_LABEL), exceed=table.exceed_7d * 100)[
    ["location", "pm25", "category", "mean_7d", "exceed", "open_incidents", "hotspot_index", "driver"]]
st.dataframe(show, hide_index=True, width="stretch", column_config={
    "location": "Area", "pm25": st.column_config.NumberColumn(f"PM2.5 ({when.lower()})", format="%.0f"),
    "category": "Category", "mean_7d": st.column_config.NumberColumn("7-day average", format="%.0f"),
    "exceed": st.column_config.ProgressColumn("Time elevated (7 d)", min_value=0, max_value=100, format="%.0f%%"),
    "open_incidents": "Open reports", "driver": "Likely driver",
    "hotspot_index": st.column_config.ProgressColumn("Hotspot index", min_value=0, max_value=100, format="%.0f")})
st.caption("Hotspot index = 45% recurrence (share of the last 7 days above your threshold) + 25% average level + 30% open citizen-report "
           "pressure within 3 km. Weights are an editable assumption, not a validated risk model.")
