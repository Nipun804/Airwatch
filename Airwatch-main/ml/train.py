"""Train + evaluate everything. Run:  python -m ml.generate_data && python -m ml.train

Splits by incident (cluster_id) so no incident appears in both train and test.
Writes models + metrics.json + confusion matrix into ml/artifacts/.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from app.config import ARTIFACTS_DIR, CATEGORIES, DATA_DIR, SEVERITIES
from app.services.classifier import TOKEN_PATTERN, build_severity_frame
from app.services.dedupe import cluster_hotspots, evaluate_duplicates

SEED = 42


def text_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


def clf():
    return LogisticRegression(max_iter=3000, C=10, class_weight="balanced")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "complaints.csv")
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(df, groups=df.cluster_id))
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    metrics = {"data": {"note": "SYNTHETIC data", "n_train": len(train), "n_test": len(test),
                        "n_test_incidents": int(test.cluster_id.nunique())}}

    # ---------- 1. category classifier ----------
    baseline = Pipeline([("tfidf", TfidfVectorizer(token_pattern=TOKEN_PATTERN)), ("clf", LogisticRegression(max_iter=3000))])
    baseline.fit(train.text, train.category)
    base_pred = baseline.predict(test.text)

    cat_model = Pipeline([("tfidf", text_features()), ("clf", clf())])
    cat_model.fit(train.text, train.category)
    cat_pred = cat_model.predict(test.text)
    metrics["category"] = {
        "baseline_word_tfidf_macro_f1": round(f1_score(test.category, base_pred, average="macro"), 4),
        "final_macro_f1": round(f1_score(test.category, cat_pred, average="macro"), 4),
        "final_accuracy": round(accuracy_score(test.category, cat_pred), 4),
        "per_class": classification_report(test.category, cat_pred, output_dict=True, zero_division=0),
        "by_language_macro_f1": {
            lang: round(f1_score(test.category[m], cat_pred[m.values], average="macro"), 4)
            for lang, m in ((l, test.language == l) for l in ["en", "hinglish", "hi"])
        },
    }
    cm = confusion_matrix(test.category, cat_pred, labels=CATEGORIES)
    pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES).to_csv(ARTIFACTS_DIR / "confusion_category.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45, ha="right")
        ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
        for i in range(len(CATEGORIES)):
            for j in range(len(CATEGORIES)):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Category confusion matrix (synthetic test set)")
        fig.tight_layout(); fig.savefig(ARTIFACTS_DIR / "confusion_category.png", dpi=150)
    except ImportError:
        pass

    # ---------- 2. severity classifier ----------
    def sev_pipeline(use_numeric=True):
        cols = [("text", text_features(), "text")]
        if use_numeric:
            cols.append(("num", "passthrough", ["sens_prox", "duration_norm"]))
        return Pipeline([("pre", ColumnTransformer(cols)), ("clf", clf())])

    Xtr = build_severity_frame(train.text, train.nearest_sensitive_m, train.duration_hours)
    Xte = build_severity_frame(test.text, test.nearest_sensitive_m, test.duration_hours)
    sev_text_only = sev_pipeline(False).fit(Xtr, train.severity)
    sev_model = sev_pipeline(True).fit(Xtr, train.severity)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    sev_pred = sev_model.predict(Xte)
    ord_mae = float(np.mean(np.abs(test.severity.map(order) - pd.Series(sev_pred).map(order))))
    metrics["severity"] = {
        "text_only_macro_f1": round(f1_score(test.severity, sev_text_only.predict(Xte), average="macro"), 4),
        "text_plus_context_macro_f1": round(f1_score(test.severity, sev_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(test.severity, sev_pred), 4),
        "ordinal_mae_levels": round(ord_mae, 4),
    }

    # ---------- 3. duplicate detection + hotspot clustering ----------
    # pick the radius on TRAIN incidents (true categories), report on TEST (predicted categories)
    sweep = {r: evaluate_duplicates(train.lat, train.lon, train.category, train.cluster_id, r)["f1"]
             for r in (50, 75, 100, 150, 200, 300, 500)}
    best_r = max(sweep, key=sweep.get)
    (ARTIFACTS_DIR / "dedupe_config.json").write_text(json.dumps({"radius_m": best_r}))
    dup = evaluate_duplicates(test.lat, test.lon, cat_pred, test.cluster_id, best_r)
    labels = cluster_hotspots(test.lat, test.lon, cat_pred, radius_m=best_r)
    metrics["dedupe_hotspots"] = {
        "radius_m_chosen_on_train": best_r,
        "train_f1_by_radius": {str(k): round(v, 4) for k, v in sweep.items()},
        "pairwise_duplicate_precision": round(dup["precision"], 4),
        "pairwise_duplicate_recall": round(dup["recall"], 4),
        "pairwise_duplicate_f1": round(dup["f1"], 4),
        "hotspot_adjusted_rand_index": round(adjusted_rand_score(test.cluster_id, labels), 4),
        "true_incidents": int(test.cluster_id.nunique()), "predicted_hotspots": int(len(set(labels))),
    }

    joblib.dump(cat_model, ARTIFACTS_DIR / "category_model.joblib")
    joblib.dump(sev_model, ARTIFACTS_DIR / "severity_model.joblib")
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_class"} for k, v in metrics.items()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
