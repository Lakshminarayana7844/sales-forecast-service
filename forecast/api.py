import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from . import model as M
from .features import REGIONS, load

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
DATA = Path(os.environ.get("DATA_PATH", ROOT / "data" / "weekly_sales.csv"))

app = FastAPI(title="Sales Forecast Service", version="1.0.0")
_state = {}


def _get():
    if not _state:
        if not (MODEL_DIR / "latest.json").exists():
            M.train(DATA, MODEL_DIR)
        _state["model"], _state["meta"] = M.load_latest(MODEL_DIR)
        _state["history"] = load(DATA)
    return _state


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_info():
    return _get()["meta"]


@app.get("/forecast")
def get_forecast(region: str = Query(...), weeks: int = Query(4, ge=1, le=26)):
    s = _get()
    try:
        rows = M.forecast(s["model"], s["history"], region, weeks)
    except ValueError as e:
        raise HTTPException(400 if region not in REGIONS else 422, str(e))
    return {"region": region, "model_version": s["meta"]["version"], "forecast": rows}
