import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecast import model as M
from forecast.features import FEATURES, build, load

DATA = "data/weekly_sales.csv"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    meta = M.train(DATA, d)
    return d, meta


def test_load_validates(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("week,region,revenue\n2024-01-01,EMEA,10\n2024-01-01,EMEA,11\n")
    with pytest.raises(ValueError):
        load(p)
    p.write_text("week,region,revenue\n2024-01-01,EMEA,-1\n")
    with pytest.raises(ValueError):
        load(p)


def test_features_have_no_leakage():
    df = load(DATA)
    f = build(df)
    r = f[f["region"] == "EMEA"].iloc[10]
    hist = df[(df["region"] == "EMEA") & (df["week"] < r["week"])].sort_values("week")["revenue"]
    assert r["lag_1"] == hist.iloc[-1]
    assert r["lag_52"] == hist.iloc[-52]
    assert abs(r["roll4_mean"] - hist.iloc[-4:].mean()) < 1e-9
    assert not f[FEATURES].isna().any().any()


def test_model_beats_baselines(trained):
    m = trained[1]["backtest_mape"]
    assert m["model"] < m["naive"]
    assert m["model"] < 10


def test_artifacts_versioned(trained):
    d, meta = trained
    assert (d / f"model_{meta['version']}.joblib").exists()
    assert (d / f"model_{meta['version']}.json").exists()
    assert M.load_latest(d)[1]["version"] == meta["version"]


def test_forecast_shape_and_dates(trained):
    model, _ = M.load_latest(trained[0])
    df = load(DATA)
    out = M.forecast(model, df, "EMEA", 6)
    assert len(out) == 6
    assert pd.Timestamp(out[0]["week"]) == df["week"].max() + pd.Timedelta(weeks=1)
    assert all(o["forecast"] > 0 for o in out)


def test_forecast_reasonable_level(trained):
    model, _ = M.load_latest(trained[0])
    df = load(DATA)
    last = df[df.region == "EMEA"].tail(8)["revenue"].mean()
    out = M.forecast(model, df, "EMEA", 4)
    assert 0.5 * last < out[0]["forecast"] < 1.8 * last


def test_unknown_region(trained):
    model, _ = M.load_latest(trained[0])
    with pytest.raises(ValueError):
        M.forecast(model, load(DATA), "MARS", 2)


def test_api(trained, monkeypatch):
    import forecast.api as api
    monkeypatch.setattr(api, "MODEL_DIR", trained[0])
    api._state.clear()
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/model").json()["version"] == trained[1]["version"]
    r = c.get("/forecast", params={"region": "APAC", "weeks": 3}).json()
    assert len(r["forecast"]) == 3
    assert c.get("/forecast", params={"region": "MARS"}).status_code == 400
    assert c.get("/forecast", params={"region": "EMEA", "weeks": 100}).status_code == 422
