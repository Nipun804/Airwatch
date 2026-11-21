"""Train + evaluate the PM2.5 forecaster.  Run:  python -m ml.train_forecast
Uses data/air_quality.csv if you fetched real data, otherwise the labelled simulation."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from app.config import ARTIFACTS_DIR
from app.services import forecast as fc
from app.services.aq_data import load_air_quality

SEED = 42


def _err(y, p):
    d = np.asarray(p) - np.asarray(y)
    return {"rmse": float(np.sqrt(np.mean(d ** 2))), "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d))}


def _make(**kw):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.07, max_leaf_nodes=24, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=SEED, **kw)


def _resid(d):
    """Target: log-change versus persistence ('no change from now')."""
    return np.log1p(d["y"]) - np.log1p(d["persist"])


def _group_importance(model, X, y_res, rng):
    """Grouped permutation importance: increase in (log-scale) RMSE when a whole feature group is shuffled."""
    base = np.sqrt(np.mean((model.predict(X) - y_res) ** 2))
    out = {}
    for name, cols in fc.GROUPS.items():
        deltas = []
        for _ in range(3):
            Xp = X.copy()
            Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(Xp))]
            deltas.append(np.sqrt(np.mean((model.predict(Xp) - y_res) ** 2)) - base)
        out[name] = max(float(np.mean(deltas)), 0.0)
    total = sum(out.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df, meta = load_air_quality()
    t0 = meta["t0"]
    train, cal, test = fc.build_dataset(df, t0)
    print(f"data source: {meta['source']} | train rows {len(train):,} | calibration {len(cal):,} | test rows {len(test):,}")

    X, y = train[fc.FEATURES], _resid(train)
    models = {"median": _make(loss="squared_error").fit(X, y),
              "q10": _make(loss="quantile", quantile=0.1).fit(X, y),
              "q90": _make(loss="quantile", quantile=0.9).fit(X, y), "features": fc.FEATURES, "k": 1.0}

    # choose the band-widening factor k on the calibration slice (smallest k giving >= 80% coverage)
    yc = cal["y"].to_numpy()
    for k in np.arange(0.5, 8.0, 0.05):
        _, lo_c, hi_c = fc.band(models, cal, k=k)
        if np.mean((yc >= lo_c) & (yc <= hi_c)) >= 0.80:
            models["k"] = float(round(k, 2))
            break
    print("band widening factor k =", models["k"])

    med, lo, hi = fc._predict(models, test)
    yt = test["y"].to_numpy()
    by_h = {}
    for h in (1, 6, 12, 24, 36, 48):
        m = (test["_h"] == h).to_numpy()
        by_h[str(h)] = {"model_rmse": _err(yt[m], med[m])["rmse"], "persistence_rmse": _err(yt[m], test["persist"][m])["rmse"],
                        "seasonal_naive_rmse": _err(yt[m], test["snaive"][m])["rmse"]}
    sample = test.sample(min(4000, len(test)), random_state=SEED)
    metrics = {
        "data_source": meta["source"], "data_label": meta.get("label", ""), "t0": str(t0),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train_rows": int(len(train)), "n_test_rows": int(len(test)), "n_locations": int(df.location.nunique()),
        "mean_pm25_test": float(yt.mean()),
        "overall": _err(yt, med), "persistence": _err(yt, test["persist"]), "seasonal_naive": _err(yt, test["snaive"]),
        "band_coverage_80": float(np.mean((yt >= lo) & (yt <= hi))), "band_factor_k": models["k"], "by_horizon": by_h,
        "importance": _group_importance(models["median"], sample[fc.FEATURES], _resid(sample), np.random.default_rng(SEED)),
    }
    joblib.dump(models, ARTIFACTS_DIR / "forecast_models.joblib")
    (ARTIFACTS_DIR / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))
    fc._load.cache_clear()
    print("overall", {k: round(v, 2) for k, v in metrics["overall"].items()},
          "| persistence rmse", round(metrics["persistence"]["rmse"], 1), "| coverage80", round(metrics["band_coverage_80"], 3))


if __name__ == "__main__":
    main()
