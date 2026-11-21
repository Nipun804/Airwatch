import streamlit as st

from app.config import CATEGORIES, LOCALITIES
from app.schemas import ComplaintCreate
from app.services import pipeline
from app.services.serializers import complaint_detail
from frontend.common import (CAT_ICON, CAT_LABEL, STATUS_LABEL, breakdown_chart, current_user, db_session,
                             priority_badge, save_upload, setup, show_media)
from app import crud
from sqlalchemy import select, func
from app.models import Complaint

setup("Report Pollution", "📝", require_login=True)
user = current_user()
st.title("📝 Report pollution")


def _result_card(complaint_id):
    with db_session() as db:
        from app.models import Complaint as C
        c = db.get(C, complaint_id)
        d = complaint_detail(db, c)
        n_reports = db.scalar(select(func.count(Complaint.id)).where(Complaint.hotspot_id == c.hotspot_id,
                                                                     Complaint.status != "rejected"))
        n_citizens = crud.unique_supporters(db, c.hotspot_id)
    with st.container(border=True):
        st.markdown("### 🤖 AirWatch AI Assistant")
        st.success(f"Report **{d.public_id}** received. Here's what the AI found:")
        m = st.columns(4)
        m[0].metric("Pollution type", f"{CAT_ICON.get(d.category, '')} {CAT_LABEL.get(d.category, d.category)}",
                    f"{(d.category_confidence or 0):.0%} confidence", delta_color="off")
        m[1].metric("Severity", d.severity.title(), f"{(d.severity_confidence or 0):.0%} confidence", delta_color="off")
        m[2].metric("Priority score", f"{d.priority_score:.0f} / 100")
        m[3].metric("Nearest school/hospital", f"{d.nearest_sensitive_m:.0f} m" if d.nearest_sensitive_m is not None else "n/a")
        st.markdown(priority_badge(d.priority_level, d.priority_score), unsafe_allow_html=True)
        if n_reports > 1:
            st.info(f"🔗 This matches an existing incident nearby. It now has **{n_reports} reports** from "
                    f"**{n_citizens} citizens**, which raises its priority.")
        else:
            st.info("🆕 This is a new incident. Ask neighbours to support it so it gets noticed.")
        st.write(f"📮 Routed to **{d.department}** (simulated). Status: **{STATUS_LABEL[d.status]}**.")
        if d.needs_human_review:
            st.warning("The AI is not fully confident, so a moderator will review this carefully.")
        with st.expander("Why this priority?"):
            breakdown_chart(d.priority_breakdown)
        show_media(d.media_url)
        st.caption("⚠️ Illustrative AI-generated score, not a validated risk measurement. "
                   "Human verification is required before this appears in the public feed.")


if "last_submission" in st.session_state:
    _result_card(st.session_state["last_submission"])
    if st.button("➕ Report another incident"):
        st.session_state.pop("last_submission")
        st.rerun()
    st.stop()

# ---- 1. where? (locality shortcut + click on the map to fine-tune)
names = list(LOCALITIES)


def _on_locality():
    la, lo, _ = LOCALITIES[st.session_state["rp_loc"]]
    st.session_state["rp_lat"], st.session_state["rp_lon"] = la, lo


if "rp_lat" not in st.session_state:
    st.session_state["rp_loc"] = names[0]
    st.session_state["rp_lat"], st.session_state["rp_lon"], _ = LOCALITIES[names[0]]

st.subheader("1. Where is it happening?")
st.selectbox("Area (jump to a locality)", names, key="rp_loc", on_change=_on_locality)

import folium
from streamlit_folium import st_folium

m = folium.Map(location=[st.session_state["rp_lat"], st.session_state["rp_lon"]], zoom_start=15)
folium.Marker([st.session_state["rp_lat"], st.session_state["rp_lon"]], tooltip="Reported location").add_to(m)
clicked = st_folium(m, height=300, use_container_width=True, returned_objects=["last_clicked"], key="pick_map")
if clicked and clicked.get("last_clicked"):
    new = (round(clicked["last_clicked"]["lat"], 6), round(clicked["last_clicked"]["lng"], 6))
    if new != (st.session_state["rp_lat"], st.session_state["rp_lon"]):
        st.session_state["rp_lat"], st.session_state["rp_lon"] = new
        st.rerun()
st.caption("👆 Click the map to drop the pin exactly where the pollution is.")
with st.expander("Or enter coordinates manually"):
    c1, c2 = st.columns(2)
    c1.number_input("Latitude", key="rp_lat", format="%.6f")
    c2.number_input("Longitude", key="rp_lon", format="%.6f")

# ---- 2. what?
st.subheader("2. What are you seeing?")
with st.form("report_form"):
    text = st.text_area("Describe the problem (English, Hinglish or Hindi)", height=110,
                        placeholder="e.g. Garbage is being burned near the park, thick smoke and children coughing. / "
                                    "kachra jalaya ja raha hai, bahut dhuan hai / कचरा जलाया जा रहा है")
    c1, c2 = st.columns(2)
    cat_choice = c1.selectbox("Pollution type (optional - the AI will decide)", ["Let AI decide"] + CATEGORIES,
                              format_func=lambda x: x if x == "Let AI decide" else f"{CAT_ICON[x]} {CAT_LABEL[x]}")
    hours = c2.number_input("How long has it been going on? (hours, optional)", min_value=0.0, value=0.0, step=1.0)
    photo = st.file_uploader("Photo evidence (optional)", type=["jpg", "jpeg", "png"])
    st.caption("Privacy: your exact identity is never shown in the public feed. Avoid photographing faces or number plates.")
    submitted = st.form_submit_button("🚀 Submit report", type="primary")

if submitted:
    if len(text.strip()) < 5:
        st.error("Please describe the problem (at least a few words).")
    else:
        with st.spinner("AI is analysing your report…"):
            with db_session() as db:
                c = pipeline.submit_complaint(db, ComplaintCreate(
                    reporter_id=user["id"], description=text.strip(), lat=st.session_state["rp_lat"],
                    lon=st.session_state["rp_lon"], user_category=None if cat_choice == "Let AI decide" else cat_choice,
                    duration_hours=hours or None, media_url=save_upload(photo) if photo else None))
                st.session_state["last_submission"] = c.id
        st.rerun()
