import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.config import CATEGORIES, LOCALITIES
from app.services import sources
from app.services.serializers import build_incidents
from frontend.common import SOURCE_COLOR, aq_banner, db_session, get_aq, setup

setup("Pollution Sources", "🏭")
st.title("🏭 Pollution source identification")
st.warning("**Indicative, not a measurement.** Shares come from fusing citizen reports, sensor signatures and land-use "
           "knowledge. They are *normalised scores*, not tonnes of emissions or a validated source-apportionment study.")

df, meta = get_aq()
t0 = meta["t0"]
aq_banner(meta)
with db_session() as db:
    incidents = build_incidents(db, verified_only=False)
attr = sources.attribute_sources(df, t0, incidents)
L = sources.SOURCE_LABEL

tab1, tab2, tab3 = st.tabs(["🧭 Source mix by area", "🗺️ Emission-style density maps", "ℹ️ How it works"])

with tab1:
    loc = st.selectbox("Area", sorted(attr), index=sorted(attr).index("Okhla") if "Okhla" in attr else 0)
    a = attr[loc]
    left, right = st.columns([3, 2])
    with left:
        mix = pd.DataFrame({"Source": [L[c] for c in CATEGORIES], "Share": [a["shares"][c] for c in CATEGORIES]}).sort_values("Share")
        st.markdown(f"**Likely contributing sources · {loc}**")
        st.altair_chart(alt.Chart(mix).mark_bar().encode(
            x=alt.X("Share:Q", axis=alt.Axis(format="%"), title=None), y=alt.Y("Source:N", sort="-x", title=None),
            color=alt.Color("Source:N", scale=alt.Scale(domain=list(SOURCE_COLOR), range=list(SOURCE_COLOR.values())), legend=None),
            tooltip=["Source", alt.Tooltip("Share:Q", format=".0%")]).properties(height=230))
        comp = pd.DataFrame({L[c]: {k.title(): a["components"][k][c] for k in a["components"]} for c in CATEGORIES}).T
        st.markdown("**Evidence behind the estimate** (each column is a normalised share)")
        st.dataframe(comp.style.format("{:.0%}"), width="stretch")
    with right:
        st.markdown("**Why?**")
        for e in a["evidence"]:
            st.markdown(f"- {e}")
        st.caption(f"{a['n_incidents']} citizen incident(s) within 3 km considered.")

    st.divider()
    st.markdown("**All areas at a glance**")
    long = pd.DataFrame([{"Area": n, "Source": L[c], "Share": v["shares"][c]} for n, v in attr.items() for c in CATEGORIES])
    st.altair_chart(alt.Chart(long).mark_bar().encode(
        y=alt.Y("Area:N", sort=alt.SortField("Share", order="descending"), title=None), x=alt.X("Share:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("Source:N", scale=alt.Scale(domain=list(SOURCE_COLOR), range=list(SOURCE_COLOR.values()))),
        tooltip=["Area", "Source", alt.Tooltip("Share:Q", format=".0%")]).properties(height=470))

with tab2:
    c1, c2 = st.columns(2)
    synth = c1.checkbox("Include simulated historical complaint archive (adds density for the demo)", True)
    bw = c2.slider("Smoothing (km)", 1.0, 6.0, 2.5, 0.5)
    pts = sources.archive_points(incidents, synth)
    G = sources.emission_grids(pts, bandwidth_km=bw)
    lon0, lon1, lat0, lat1 = G["extent"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), constrained_layout=True)
    lons, lats = [v[1] for v in LOCALITIES.values()], [v[0] for v in LOCALITIES.values()]
    total = sum(G["grids"].values())
    im = None
    for ax, key in zip(axes.ravel(), CATEGORIES + ["total"]):
        grid = total if key == "total" else G["grids"][key]
        im = ax.imshow(grid / (grid.max() or 1), origin="lower", extent=(lon0, lon1, lat0, lat1), cmap="OrRd", vmin=0, vmax=1,
                       aspect=1 / np.cos(np.radians(28.6)))
        ax.scatter(lons, lats, s=6, c="#334155", alpha=.6)
        ax.set_title("All sources" if key == "total" else L[key], fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="Relative complaint density (each panel scaled to its own maximum)")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Maps show the relative density of **citizen complaints** per source across Delhi NCR (dots = monitoring areas). "
               "They are *not* measured emissions. With a real emission inventory or satellite fire data the same layout could show actual emissions.")

with tab3:
    st.markdown(f"""
**For each area, three kinds of evidence are fused for the five source types:**

1. **Citizen evidence ({sources.WEIGHTS['citizen']:.0%})**: verified complaints within 3 km, weighted by AI priority and the number of
   *unique* citizens; unverified and resolved reports count half.
2. **Sensor signatures ({sources.WEIGHTS['sensor']:.0%})**: heuristics on the last 7 days of data, relative to the network median:
   PM10/PM2.5 ratio (coarse dust → construction), NO₂ and morning rush-hour peaks (traffic), NO₂ and night-time persistence (industry),
   evening spikes (waste burning), and PM2.5 when winds blow from the north-west (crop-residue smoke).
3. **Land-use prior ({sources.WEIGHTS['prior']:.0%})**: industrial areas, outskirts near farmland and dense traffic corridors.
   If there are no nearby reports, the citizen component is dropped and the weights are re-normalised.

**Limitations (be upfront with judges):** the signature rules are domain heuristics, not fitted to labelled source data; the shares are
relative scores; with only a few reports the citizen component is noisy; the demo data is simulated unless you ran the Open-Meteo fetch.
A production system would add a real emission inventory, satellite fire detections (e.g. NASA FIRMS), traffic counts and chemical
speciation data, and validate against source-apportionment studies.
""")
