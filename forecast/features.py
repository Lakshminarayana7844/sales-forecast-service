"""Feature engineering for weekly revenue per region. Only past values are used (no leakage)."""
import numpy as np
import pandas as pd

LAGS = (1, 2, 4, 52)
FEATURES = [f"lag_{l}" for l in LAGS] + ["roll4_mean", "week_sin", "week_cos", "is_q4_peak", "region_code"]
REGIONS = ["AMER", "APAC", "EMEA"]


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    if df.duplicated(["week", "region"]).any():
        raise ValueError("duplicate (week, region) rows")
    if df["revenue"].isna().any() or (df["revenue"] < 0).any():
        raise ValueError("revenue must be present and non-negative")
    return df.sort_values(["region", "week"]).reset_index(drop=True)


def calendar(weeks: pd.Series) -> pd.DataFrame:
    wk = weeks.dt.isocalendar().week.astype(int)
    return pd.DataFrame({
        "week_sin": np.sin(2 * np.pi * wk / 52),
        "week_cos": np.cos(2 * np.pi * wk / 52),
        "is_q4_peak": (wk >= 47).astype(int),
    })


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for region, g in df.groupby("region"):
        g = g.sort_values("week").copy()
        for l in LAGS:
            g[f"lag_{l}"] = g["revenue"].shift(l)
        g["roll4_mean"] = g["revenue"].shift(1).rolling(4).mean()
        g = pd.concat([g.reset_index(drop=True), calendar(g["week"]).reset_index(drop=True)], axis=1)
        g["region_code"] = REGIONS.index(region)
        out.append(g)
    res = pd.concat(out).dropna(subset=FEATURES).reset_index(drop=True)
    return res
