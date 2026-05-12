"""
api/server.py
=============
FastAPI REST API — exposes forecast endpoints for the dashboard UI.
Deployable to Vercel (via vercel.json) or run locally with uvicorn.
"""

import os
import sys
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sales Forecast API",
    description="ARIMA + LSTM retail sales forecasting with Neon DB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (plots, dashboard)
static_path = Path("./static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    model: str = "both"          # "arima" | "lstm" | "both"
    horizon: int = 30            # Days to forecast into the future
    save_db: bool = False

class ForecastResponse(BaseModel):
    run_id: str
    model: str
    arima_metrics: Optional[dict]
    lstm_metrics: Optional[dict]
    forecast_dates: List[str]
    arima_forecast: Optional[List[float]]
    lstm_forecast: Optional[List[float]]
    status: str


# ─── In-Memory Cache ─────────────────────────────────────────────────────────

forecast_cache = {}


# ─── Helper: Run forecast ────────────────────────────────────────────────────

def _run_forecast(model: str = "both", horizon: int = 30) -> dict:
    from data.data_loader import load_csv_data, add_features, split_data, scale_data
    from models.arima_model import ARIMAForecaster, evaluate_arima

    # LSTM optional — TensorFlow not available on Vercel (size limit)
    try:
        from models.lstm_model import LSTMForecaster, evaluate_lstm
        LSTM_AVAILABLE = True
    except ImportError:
        LSTM_AVAILABLE = False
        logger.warning("TensorFlow not available — LSTM disabled. Run locally for LSTM.")

    df = add_features(load_csv_data())
    train, test = split_data(df)
    train_sales = train["sales"].values
    test_sales = test["sales"].values

    result = {
        "arima_metrics": None, "lstm_metrics": None,
        "arima_forecast": None, "lstm_forecast": None,
        "forecast_dates": [],
        "lstm_available": LSTM_AVAILABLE,
    }

    # Future dates
    last_date = pd.Timestamp(test["date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                  periods=horizon, freq="D")
    result["forecast_dates"] = [str(d.date()) for d in future_dates]

    if model in ("arima", "both"):
        arima = ARIMAForecaster()
        arima.fit(train_sales)
        future_pred, _ = arima.predict(horizon)
        result["arima_forecast"] = future_pred.tolist()
        test_pred = arima.predict_rolling(test_sales)
        result["arima_metrics"] = evaluate_arima(test_sales, test_pred)

    if model in ("lstm", "both") and LSTM_AVAILABLE:
        train_scaled, test_scaled, scaler = scale_data(train_sales, test_sales)
        lstm = LSTMForecaster()
        lstm.fit(train_scaled, scaler)
        last_seq = np.concatenate([train_scaled.flatten(), test_scaled.flatten()])
        future_pred = lstm.predict_future(last_seq, horizon)
        result["lstm_forecast"] = future_pred.tolist()
        test_pred = lstm.predict(train_scaled, test_scaled)
        result["lstm_metrics"] = evaluate_lstm(test_sales, test_pred)
    elif model in ("lstm", "both") and not LSTM_AVAILABLE:
        result["lstm_forecast"] = None
        result["lstm_metrics"] = {"note": "LSTM not available on Vercel. Run locally."}

    return result


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Sales Forecast API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/forecast", "/data/summary", "/results/latest", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/forecast", response_model=ForecastResponse)
def run_forecast(req: ForecastRequest, background_tasks: BackgroundTasks):
    """Run ARIMA/LSTM forecast and return predictions."""
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Forecast request: model={req.model}, horizon={req.horizon}")

    try:
        result = _run_forecast(req.model, req.horizon)
        response = ForecastResponse(
            run_id=run_id,
            model=req.model,
            arima_metrics=result["arima_metrics"],
            lstm_metrics=result["lstm_metrics"],
            forecast_dates=result["forecast_dates"],
            arima_forecast=result["arima_forecast"],
            lstm_forecast=result["lstm_forecast"],
            status="success",
        )
        forecast_cache[run_id] = response.dict()

        if req.save_db:
            background_tasks.add_task(_save_results_bg, run_id, result)

        return response
    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/summary")
def data_summary():
    """Return summary stats of the loaded sales dataset."""
    from data.data_loader import load_csv_data, add_features
    df = add_features(load_csv_data())
    return {
        "rows": len(df),
        "date_range": {"start": str(df["date"].min().date()), "end": str(df["date"].max().date())},
        "sales_stats": {
            "mean": round(df["sales"].mean(), 2),
            "std": round(df["sales"].std(), 2),
            "min": round(df["sales"].min(), 2),
            "max": round(df["sales"].max(), 2),
            "median": round(df["sales"].median(), 2),
        },
        "has_missing": df["sales"].isnull().any(),
    }


@app.get("/results/latest")
def latest_results():
    """Return the most recent model run metrics from cache or DB."""
    if forecast_cache:
        latest = list(forecast_cache.values())[-1]
        return latest
    try:
        from utils.database import db
        if db.connect():
            return {"runs": db.get_latest_runs(limit=5)}
    except Exception:
        pass
    return {"message": "No results yet. Run /forecast first."}


@app.get("/plots/{name}")
def get_plot(name: str):
    """Serve a generated plot by filename."""
    path = f"./static/plots/{name}"
    if Path(path).exists():
        return FileResponse(path)
    raise HTTPException(status_code=404, detail=f"Plot '{name}' not found.")


def _save_results_bg(run_id: str, result: dict):
    try:
        from utils.database import db
        if db.connect():
            db.create_tables()
    except Exception as e:
        logger.error(f"BG DB save failed: {e}")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("APP_PORT", 8000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    uvicorn.run("api.server:app", host="0.0.0.0", port=port, reload=debug)
