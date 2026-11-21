"""Headless smoke tests: every Streamlit page must render without exceptions."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["streamlit_app.py", "pages/1_Report_Pollution.py", "pages/2_Community_Feed.py",
         "pages/3_Authority_Portal.py", "pages/4_Scoreboard.py", "pages/5_Model_Performance.py",
         "pages/6_Air_Quality_Forecast.py", "pages/7_Hotspot_Map.py", "pages/8_Pollution_Sources.py"]


def _run(path, **session):
    at = AppTest.from_file(str(ROOT / path), default_timeout=90)
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _run(page, user={"id": 1, "name": "Citizen 1", "email": "citizen1@demo.com"})
    assert not at.exception, [e.value for e in at.exception]


def test_report_page_requires_login():
    at = _run("pages/1_Report_Pollution.py")
    assert not at.exception and any("sign in" in i.value.lower() for i in at.info)


def test_submit_report_end_to_end():
    at = _run("pages/1_Report_Pollution.py", user={"id": 2, "name": "Citizen 2", "email": "citizen2@demo.com"})
    at.text_area[0].set_value("Factory chimney releasing black smoke near Okhla. Huge amount of smoke, people coughing.")
    at.button[[b.label for b in at.button].index("🚀 Submit report")].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("AirWatch AI Assistant" in m.value for m in at.markdown)
    assert "last_submission" in at.session_state
