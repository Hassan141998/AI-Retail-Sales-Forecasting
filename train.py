"""
train.py
========
Main training pipeline — runs ARIMA + LSTM, evaluates, visualizes, and saves results.
Run: python train.py [--model arima|lstm|both] [--save-db]
"""

import os
import sys
import uuid
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# Configure logger
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
log_path = Path("./logs")
log_path.mkdir(exist_ok=True)
logger.add("./logs/train.log", rotation="10 MB", retention="30 days")


def run_pipeline(model: str = "both", save_db: bool = False):
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"🚀 Starting Sales Forecast Pipeline | Run ID: {run_id}")

    # ── 1. Load & Prepare Data ────────────────────────────────────────────────
    from data.data_loader import (
        load_csv_data, add_features, split_data, scale_data, create_sequences
    )

    logger.info("📥 Loading data...")
    df = load_csv_data()
    df = add_features(df)
    train, test = split_data(df)

    train_sales = train["sales"].values
    test_sales = test["sales"].values
    test_dates = test["date"].values

    logger.info(f"📊 Dataset: {len(df)} rows | Train: {len(train)} | Test: {len(test)}")

    # ── 2. Visualize Raw Data ─────────────────────────────────────────────────
    from utils.visualizer import (
        plot_sales_overview, plot_arima_forecast, plot_lstm_forecast,
        plot_model_comparison, plot_training_history, create_interactive_dashboard
    )

    logger.info("📈 Generating data overview...")
    plot_sales_overview(df)

    arima_preds = None
    arima_metrics = None
    lstm_preds = None
    lstm_metrics = None
    arima_conf = None

    # ── 3. ARIMA ──────────────────────────────────────────────────────────────
    if model in ("arima", "both"):
        logger.info("\n── ARIMA Model ───────────────────────────────────")
        from models.arima_model import ARIMAForecaster, evaluate_arima, check_stationarity

        check_stationarity(pd.Series(train_sales))

        arima = ARIMAForecaster()
        arima.fit(train_sales)

        # Rolling one-step-ahead for accurate metrics
        arima_preds = arima.predict_rolling(test_sales)
        arima_metrics = evaluate_arima(test_sales, arima_preds)

        # Multi-step confidence interval just for the plot bands
        _, arima_conf = arima.predict(len(test_sales))

        plot_arima_forecast(test_dates, test_sales, arima_preds, arima_conf, arima_metrics)
        arima.save()
        logger.success(f"✅ ARIMA done | {arima_metrics}")

    # ── 4. LSTM ───────────────────────────────────────────────────────────────
    if model in ("lstm", "both"):
        logger.info("\n── LSTM Model ────────────────────────────────────")
        from models.lstm_model import LSTMForecaster, evaluate_lstm

        train_scaled, test_scaled, scaler = scale_data(train_sales, test_sales)

        lstm = LSTMForecaster()
        lstm.fit(train_scaled, scaler)
        lstm_preds = lstm.predict(train_scaled, test_scaled)
        lstm_metrics = evaluate_lstm(test_sales, lstm_preds)

        # Future forecast (+30 days)
        forecast_horizon = int(os.getenv("FORECAST_HORIZON", 30))
        last_seq = np.concatenate([train_scaled.flatten(), test_scaled.flatten()])
        future_preds = lstm.predict_future(last_seq, forecast_horizon)
        future_dates = pd.date_range(
            start=pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=1),
            periods=forecast_horizon, freq="D"
        )

        plot_lstm_forecast(train, test, lstm_preds, future_dates, future_preds, lstm_metrics)
        plot_training_history(lstm.history)
        lstm.save()
        logger.success(f"✅ LSTM done | {lstm_metrics}")

    # ── 5. Comparison ─────────────────────────────────────────────────────────
    if model == "both" and arima_metrics and lstm_metrics:
        logger.info("\n── Model Comparison ──────────────────────────────")
        plot_model_comparison(arima_metrics, lstm_metrics, test_dates,
                               test_sales, arima_preds, lstm_preds)
        create_interactive_dashboard(df, test_dates, test_sales,
                                      arima_preds, lstm_preds,
                                      arima_metrics, lstm_metrics)
        _print_comparison_table(arima_metrics, lstm_metrics)

    # ── 6. Save to Neon DB ────────────────────────────────────────────────────
    if save_db:
        _save_to_db(run_id, test_dates, test_sales,
                    arima_preds, arima_metrics,
                    lstm_preds, lstm_metrics,
                    len(train), len(test))

    logger.success(f"\n🎉 Pipeline complete! Check ./static/plots/ for outputs.")
    return {
        "run_id": run_id,
        "arima_metrics": arima_metrics,
        "lstm_metrics": lstm_metrics,
    }


def _print_comparison_table(arima: dict, lstm: dict):
    print("\n" + "=" * 50)
    print(f"{'Metric':<10} {'ARIMA':>12} {'LSTM':>12} {'Winner':>10}")
    print("=" * 50)
    for metric in ["MAE", "RMSE", "MAPE"]:
        a, l = arima[metric], lstm[metric]
        winner = "ARIMA ✓" if a < l else "LSTM ✓"
        print(f"{metric:<10} {a:>12.2f} {l:>12.2f} {winner:>10}")
    metric = "R2"
    a, l = arima[metric], lstm[metric]
    winner = "ARIMA ✓" if a > l else "LSTM ✓"
    print(f"{metric:<10} {a:>12.4f} {l:>12.4f} {winner:>10}")
    print("=" * 50)


def _save_to_db(run_id, test_dates, test_sales,
                arima_preds, arima_metrics,
                lstm_preds, lstm_metrics,
                train_rows, test_rows):
    from utils.database import db
    if db.connect():
        db.create_tables()
        if arima_preds is not None:
            db.save_forecast_results(run_id, "arima", test_dates, arima_preds, test_sales, arima_metrics)
            db.save_model_run(run_id, "arima", arima_metrics, train_rows, test_rows)
        if lstm_preds is not None:
            db.save_forecast_results(run_id, "lstm", test_dates, lstm_preds, test_sales, lstm_metrics)
            db.save_model_run(run_id, "lstm", lstm_metrics, train_rows, test_rows)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Forecast Training Pipeline")
    parser.add_argument("--model", choices=["arima", "lstm", "both"], default="both",
                        help="Which model(s) to train")
    parser.add_argument("--save-db", action="store_true",
                        help="Save results to Neon PostgreSQL DB")
    args = parser.parse_args()

    results = run_pipeline(model=args.model, save_db=args.save_db)
    print(f"\nRun ID: {results['run_id']}")
