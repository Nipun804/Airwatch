import altair as alt
import pandas as pd
import streamlit as st

from app.services import aq_data, forecast
from frontend.common import aq_banner, badge, get_aq, get_forecasts, setup

setup("Air Quality Forecast", "📈")
st.title("📈 48-hour PM2.5 forecast")
st.caption("Machine-learning forecast of fine-particle pollution with an 80% uncertainty band. "
           "Not an official health advisory.")

df, meta = get_aq()
fc_all, t0, M = get_forecasts(), meta["t0"], forecast.load_metrics()
aq_banner(meta)

c = st.columns([3, 2, 3])
loc = c[0].selectbox("Area", sorted(df.location.unique()), index=sorted(df.location.unique()).index("Anand Vihar")
                     if "Anand Vihar" in set(df.location) else 0)
hours = 24 if c[1].radio("Horizon", ["24H", "48H"], index=1, horizontal=True) == "24H" else 48
thr = c[2].slider("Alert me above (µg/m³)", 30, 250, 120, step=10,
                  help="CPCB bands: 60 satisfactory · 90 moderate · 120 poor · 250 very poor")

fcl = fc_all[(fc_all.location == loc) & (fc_all.h <= hours)]
hist = forecast.history(df, loc, t0, 24)
hist7 = forecast.history(df, loc, t0, 168)

main, side = st.columns([2.3, 1])
with main:
    with st.container(border=True):
        st.markdown(f"**↗ {hours}-hour forecast · {loc}**")
        band = alt.Chart(fcl).mark_area(opacity=0.18, color="#d62728").encode(
            x=alt.X("time:T", title=None), y=alt.Y("lo:Q", title="PM2.5 (µg/m³)"), y2="hi:Q")
        line = alt.Chart(fcl).mark_line(color="#d62728", point=True).encode(
            x="time:T", y="pm25:Q", tooltip=[alt.Tooltip("time:T", title="Time", format="%a %d %b %H:%M"),
                                              alt.Tooltip("pm25:Q", title="Forecast", format=".0f"),
                                              alt.Tooltip("lo:Q", title="10th pct", format=".0f"),
                                              alt.Tooltip("hi:Q", title="90th pct", format=".0f")])
        obs = alt.Chart(hist).mark_line(color="#2563eb").encode(x="time:T", y="pm25:Q")
        rule = alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[5, 4], color="gray").encode(y="y:Q")
        now = alt.Chart(pd.DataFrame({"t": [t0]})).mark_rule(color="#888").encode(x="t:T")
        st.altair_chart(band + line + obs + rule + now)
        st.caption("🔵 last 24 h (observed/estimated) · 🔴 forecast with 80% band · dashed line = your alert level · grey line = now")

    cross = forecast.first_crossing(fcl, thr)
    if cross:
        st.warning(f"⚠️ **Alert:** PM2.5 in **{loc}** is forecast to exceed **{thr} µg/m³** from "
                   f"**{cross['first_time']:%a %d %b, %I %p}**. Peak ≈ **{cross['peak_value']:.0f}** at {cross['peak_time']:%a %I %p}.")
    else:
        st.success(f"✅ No forecast hour above {thr} µg/m³ in the next {hours} hours for {loc}.")

    d1, d2 = fcl[fcl.h <= 24], fc_all[(fc_all.location == loc) & (fc_all.h > 24)]
    m = st.columns(3)
    for col, label, val in ((m[0], "Next 24 h average", d1.pm25.mean()),
                            (m[1], "24–48 h average", d2.pm25.mean()),
                            (m[2], f"Peak (next {hours} h)", fcl.pm25.max())):
        cat, color = aq_data.pm25_category(val)
        col.metric(label, f"{val:.0f} µg/m³")
        col.markdown(badge(cat, color), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**📅 Historical trend (7 days)**")
        area = alt.Chart(hist7).mark_area(opacity=0.45, color="#2563eb", line=True).encode(
            x=alt.X("time:T", title=None), y=alt.Y("pm25:Q", title="PM2.5 (µg/m³)"))
        st.altair_chart(area)

with side:
    with st.container(border=True):
        st.markdown("**Weather conditions (now)**")
        w = hist.iloc[-1]
        st.metric("🌡️ Temperature", f"{w.temp:.0f} °C")
        st.metric("💧 Humidity", f"{w.rh:.0f} %")
        st.metric("🌬️ Wind", f"{w.wind_speed:.0f} km/h {aq_data.cardinal(w.wind_dir)}")
    with st.container(border=True):
        st.markdown("**Model performance** (held-out test period)")
        if M:
            o, p = M["overall"], M["persistence"]
            lvl = M["mean_pm25_test"]
            for name, key in (("RMSE", "rmse"), ("MAE", "mae")):
                st.markdown(f"{name} · **{o[key]:.1f}** µg/m³ (no-change baseline {p[key]:.1f})")
                st.progress(min(1.0, o[key] / lvl), text=f"{o[key] / lvl:.0%} of the average level")
            st.markdown(f"Bias · **{o['bias']:+.1f}** µg/m³")
            st.markdown(f"**{1 - o['rmse'] / p['rmse']:.0%} lower error** than assuming no change · "
                        f"80% band covers **{M['band_coverage_80']:.0%}** of observations")
            if M["data_source"] == "simulated":
                st.caption("⚠️ Evaluated on SIMULATED data: shows the pipeline works, not real-world accuracy.")
        else:
            st.info("Run `python -m ml.train_forecast` to produce metrics.")
    with st.container(border=True):
        st.markdown("**Data sources**")
        real = aq_data.is_real(meta)
        rows = [("Meteorology (wind, humidity, rain…)", "Active" if real else "Simulated", "#16a34a" if real else "#d97706"),
                ("PM2.5 / PM10 / NO₂", "Active (CAMS model)" if real else "Simulated", "#16a34a" if real else "#d97706"),
                ("Ground monitors (CPCB)", "Not connected", "#6b7280"),
                ("Satellite fire detection", "Not connected", "#6b7280"),
                ("Citizen reports (AirWatch)", "Active", "#16a34a")]
        for name, status, col in rows:
            a, b = st.columns([3, 2])
            a.write(name)
            b.markdown(badge(status, col), unsafe_allow_html=True)

if M:
    with st.expander("📊 Accuracy by forecast horizon & what drives the forecast", expanded=False):
        t1, t2 = st.columns(2)
        hz = pd.DataFrame(M["by_horizon"]).T.rename_axis("hours ahead").rename(
            columns={"model_rmse": "AirWatch model", "persistence_rmse": "No-change baseline", "seasonal_naive_rmse": "Same-hour-yesterday"}).round(1)
        t1.markdown("**RMSE (µg/m³) by horizon** (lower is better)")
        t1.dataframe(hz, width="stretch")
        t1.caption("Right after 'now' the no-change baseline is hard to beat; the model's advantage grows with the horizon.")
        imp = pd.DataFrame({"share of predictive power": M["importance"]})
        t2.markdown("**What drives the forecast** (grouped permutation importance)")
        t2.bar_chart(imp, horizontal=True)
