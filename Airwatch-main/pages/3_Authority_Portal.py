import streamlit as st
from sqlalchemy import select

from app.models import Complaint, Department
from app.services import actions
from app.services.lifecycle import allowed_next
from app.services.serializers import build_incidents, department_queue
from frontend.common import (ACTION_LABEL, CAT_ICON, CAT_LABEL, STATUS_LABEL, db_session, incident_card, priority_badge,
                             setup, show_media, status_badge, time_ago)

setup("Authority Portal", "🏛️")
st.title("🏛️ Authority portal")
st.warning("**Simulated portal.** The hackathon demo has no real government integration or authentication. "
           "Real routing would need verified department contacts or an official partnership.")

role = st.radio("Acting as", ["Moderator (verify reports)", "Department officer"], horizontal=True)

with db_session() as db:
    if role.startswith("Moderator"):
        actor = st.text_input("Your name / ID", value="Moderator-1")
        pending = build_incidents(db, verified_only=False, status="under_verification", sort="recent")
        st.subheader(f"Reports awaiting verification ({len(pending)})")
        st.caption("The AI only recommends. A human confirms the incident is real before it is published and routed.")
        if not pending:
            st.success("Nothing to verify right now. 🎉")
        for inc in pending:
            with st.container(border=True):
                h = st.columns([5, 3])
                h[0].markdown(f"#### {CAT_ICON[inc.category]} {CAT_LABEL[inc.category]} · {inc.locality}")
                h[1].markdown(priority_badge(inc.priority_level, inc.priority_score), unsafe_allow_html=True)
                reports = db.scalars(select(Complaint).where(Complaint.hotspot_id == inc.hotspot_id,
                                                             Complaint.status != "rejected")).all()
                for r in reports:
                    flag = " ⚠️ *low AI confidence*" if r.needs_human_review else ""
                    st.markdown(f"> **{r.public_id}** · {time_ago(r.created_at)}{flag}\n>\n> {r.description}")
                    show_media(r.media_url, 220)
                st.caption(f"AI suggests: **{CAT_LABEL[inc.category]}**, severity **{inc.severity}** · "
                           f"routes to **{inc.department}**")
                note = st.text_input("Note (optional)", key=f"mnote_{inc.hotspot_id}")
                b = st.columns([2, 2, 6])
                if b[0].button(ACTION_LABEL["assigned"], key=f"ver_{inc.hotspot_id}", type="primary"):
                    try:
                        actions.change_status(db, inc.complaint_id, "assigned", actor, note or "Verified by moderator")
                        st.rerun()
                    except actions.ActionError as e:
                        st.error(str(e))
                if b[1].button(ACTION_LABEL["rejected"], key=f"rej_{inc.hotspot_id}"):
                    try:
                        actions.change_status(db, inc.complaint_id, "rejected", actor, note or "Rejected: invalid or fake report")
                        st.rerun()
                    except actions.ActionError as e:
                        st.error(str(e))
    else:
        depts = db.scalars(select(Department).order_by(Department.id)).all()
        dept = st.selectbox("Department", depts, format_func=lambda d: d.name)
        actor = st.text_input("Officer name / ID", value=f"{dept.name} officer")
        queue = department_queue(db, dept.name)
        st.subheader(f"Open incidents for {dept.name} ({len(queue)})")
        if not queue:
            st.success("No open incidents. 🎉")
        for inc in queue:
            incident_card(db, inc, None, "dept", actions_enabled=False)
            note = st.text_input("Update note (shown in the public timeline)", key=f"dnote_{inc.hotspot_id}")
            nxt = [s for s in allowed_next(inc.status) if s != "rejected"]
            cols = st.columns(max(len(nxt), 1) + 1)
            for col, s in zip(cols, nxt):
                if col.button(ACTION_LABEL.get(s, s), key=f"d_{s}_{inc.hotspot_id}"):
                    if s == "resolved" and not note.strip():
                        st.error("Please add a resolution note (what action was taken?).")
                    else:
                        try:
                            actions.change_status(db, inc.complaint_id, s, actor, note or None)
                            st.rerun()
                        except actions.ActionError as e:
                            st.error(str(e))
