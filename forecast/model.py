import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .features import FEATURES, REGIONS, LAGS, build, calendar, load


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def new_model():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)


def backtest(df: pd.DataFrame, folds: int = 3, horizon: int = 8) -> dict:
    """Expanding-window backtest: train on the past, test on the next `horizon` weeks."""
    feats = build(df)
    weeks = sorted(feats["week"].unique())
    res = {"model": [], "naive": [], "seasonal_naive": []}
    for k in range(folds, 0, -1):
        cut = weeks[-k * horizon]
        end = weeks[-(k - 1) * horizon] if k > 1 else None
        train = feats[feats["week"] < cut]
        test = feats[(feats["week"] >= cut) & ((feats["week"] < end) if end is not None else True)]
        m = new_model().fit(train[FEATURES], train["revenue"])
        res["model"].append(mape(test["revenue"], m.predict(test[FEATURES])))
        res["naive"].append(mape(test["revenue"], test["lag_1"]))
        res["seasonal_naive"].append(mape(test["revenue"], test["lag_52"]))
    return {k: round(float(np.mean(v)), 2) for k, v in res.items()}


def train(csv_path, out_dir) -> dict:
    df = load(csv_path)
    metrics = backtest(df)
    feats = build(df)
    model = new_model().fit(feats[FEATURES], feats["revenue"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    version = pd.Timestamp(df["week"].max()).strftime("%Y%m%d")
    meta = {"version": version, "trained_through": str(df["week"].max().date()), "rows": int(len(feats)),
            "backtest_mape": metrics, "features": FEATURES}
    joblib.dump(model, out / f"model_{version}.joblib")
    (out / f"model_{version}.json").write_text(json.dumps(meta, indent=2))
    (out / "latest.json").write_text(json.dumps({"version": version}))
    return meta


def load_latest(model_dir):
    d = Path(model_dir)
    version = json.loads((d / "latest.json").read_text())["version"]
    return joblib.load(d / f"model_{version}.joblib"), json.loads((d / f"model_{version}.json").read_text())


def forecast(model, history: pd.DataFrame, region: str, weeks: int) -> list:
    """Recursive multi-step forecast for one region."""
    if region not in REGIONS:
        raise ValueError(f"unknown region: {region}")
    h = history[history["region"] == region].sort_values("week")
    if len(h) < max(LAGS) + 1:
        raise ValueError("not enough history")
    series = h.set_index("week")["revenue"].copy()
    out = []
    for _ in range(weeks):
        nxt = series.index[-1] + pd.Timedelta(weeks=1)
        cal = calendar(pd.Series([nxt])).iloc[0]
        row = {f"lag_{l}": series.iloc[-l] for l in LAGS}
        row["roll4_mean"] = series.iloc[-4:].mean()
        row.update({"week_sin": cal["week_sin"], "week_cos": cal["week_cos"], "is_q4_peak": cal["is_q4_peak"],
                    "region_code": REGIONS.index(region)})
        pred = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        series.loc[nxt] = pred
        out.append({"week": str(nxt.date()), "forecast": round(pred, 2)})
    return out
