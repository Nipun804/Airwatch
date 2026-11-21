import json

import pandas as pd
import streamlit as st

from app.config import ARTIFACTS_DIR
from app.services import classifier
from frontend.common import CAT_LABEL, setup

setup("Model Performance", "🧠")
st.title("🧠 AI model performance")
st.warning("**Honest disclosure:** training and test data are **synthetic** (template-based complaints in English, Hinglish "
           "and Hindi with injected noise, typos and vague reports; the test set contains only incidents unseen in training). "
           "Real-world accuracy would be lower.")

mp = ARTIFACTS_DIR / "metrics.json"
if not mp.exists():
    st.error("No metrics found. Run `python -m ml.train`.")
    st.stop()
M = json.loads(mp.read_text())
cat, sev, dd = M["category"], M["severity"], M["dedupe_hotspots"]

c = st.columns(4)
c[0].metric("Category macro-F1", f"{cat['final_macro_f1']:.3f}", f"+{cat['final_macro_f1'] - cat['baseline_word_tfidf_macro_f1']:.3f} vs baseline")
c[1].metric("Severity macro-F1", f"{sev['text_plus_context_macro_f1']:.3f}",
            f"+{sev['text_plus_context_macro_f1'] - sev['text_only_macro_f1']:.3f} with context")
c[2].metric("Duplicate-detection F1", f"{dd['pairwise_duplicate_f1']:.3f}")
c[3].metric("Hotspot clustering ARI", f"{dd['hotspot_adjusted_rand_index']:.3f}")
st.caption(f"Test set: {M['data']['n_test']} reports from {M['data']['n_test_incidents']} unseen incidents "
           f"(train: {M['data']['n_train']}). Severity is off by {sev['ordinal_mae_levels']:.2f} levels on average.")

t1, t2, t3, t4 = st.tabs(["Category classifier", "Severity", "Duplicates & hotspots", "🔬 Try it live"])
with t1:
    st.markdown(f"Baseline (word TF-IDF + logistic regression): **{cat['baseline_word_tfidf_macro_f1']:.3f}** → "
                f"final (word + character n-gram TF-IDF): **{cat['final_macro_f1']:.3f}**. "
                "Character n-grams help with Hinglish spelling variants and typos.")
    l, r = st.columns(2)
    png = ARTIFACTS_DIR / "confusion_category.png"
    if png.exists():
        l.image(str(png), caption="Confusion matrix (test set)")
    rows = {CAT_LABEL.get(k, k): {"precision": v["precision"], "recall": v["recall"], "f1": v["f1-score"], "support": v["support"]}
            for k, v in cat["per_class"].items() if k in CAT_LABEL}
    r.dataframe(pd.DataFrame(rows).T.round(3), width="stretch")
    r.markdown("**Macro-F1 by language**")
    r.dataframe(pd.DataFrame({"macro-F1": cat["by_language_macro_f1"]}).round(3), width="stretch")
with t2:
    st.markdown(f"Text only: **{sev['text_only_macro_f1']:.3f}** → text + context (distance to school/hospital, duration): "
                f"**{sev['text_plus_context_macro_f1']:.3f}**. Accuracy {sev['accuracy']:.3f}. "
                "Severity is an inherently fuzzier label than category, so it scores lower.")
with t3:
    st.markdown(f"Two reports are the **same incident** if they share a category and lie within a radius. "
                f"Radius chosen on training data: **{dd['radius_m_chosen_on_train']} m**.")
    st.dataframe(pd.DataFrame({"precision": [dd["pairwise_duplicate_precision"]], "recall": [dd["pairwise_duplicate_recall"]],
                               "F1": [dd["pairwise_duplicate_f1"]], "ARI": [dd["hotspot_adjusted_rand_index"]]}).round(3),
                 hide_index=True)
    st.markdown(f"{dd['true_incidents']} true incidents → {dd['predicted_hotspots']} predicted hotspots on the test set.")
    st.line_chart(pd.DataFrame({"train F1": dd["train_f1_by_radius"]}).rename_axis("radius (m)"))
with t4:
    txt = st.text_area("Type a complaint in English, Hinglish or Hindi",
                       "kachra jalaya ja raha hai school ke paas, bahut dhuan hai, bachche khaans rahe hain")
    d1, d2 = st.columns(2)
    dist = d1.slider("Distance to nearest school/hospital (m)", 0, 2000, 150)
    hrs = d2.slider("Duration (hours)", 0, 72, 6)
    if txt.strip():
        pr = classifier.probabilities(txt, dist, hrs)
        if pr is None:
            st.error("Models not loaded.")
        else:
            x, y = st.columns(2)
            x.markdown("**Pollution type**")
            x.bar_chart(pd.DataFrame({"probability": {CAT_LABEL.get(k, k): v for k, v in pr["category"].items()}}))
            y.markdown("**Severity**")
            y.bar_chart(pd.DataFrame({"probability": pr["severity"]}).loc[["low", "medium", "high", "critical"]])
