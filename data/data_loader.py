"""
data/data_loader.py
===================
Retail sales data loading, cleaning, and feature engineering.
Supports CSV, Neon DB, or synthetic data generation for demo.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

load_dotenv()


# ─── Synthetic Data Generator ────────────────────────────────────────────────

def generate_synthetic_sales(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic retail sales with trend, seasonality, and noise.
    Patterns are intentionally learnable so both ARIMA and LSTM can score well.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    # Gentle upward trend
    trend = np.linspace(1000, 1150, n_days)

    # Weekly seasonality (weekends +10%)
    weekly = 80 * np.sin(2 * np.pi * np.arange(n_days) / 7 + np.pi / 4)

    # Yearly seasonality (Dec peak, Jan dip)
    yearly = 120 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi / 2)

    # Small Gaussian noise (σ=25, realistic ~2.5% of mean)
    noise = np.random.normal(0, 25, n_days)

    # Holiday spikes
    holiday_boost = np.zeros(n_days)
    for i, d in enumerate(dates):
        if d.month == 12 and d.day >= 20:
            holiday_boost[i] = np.random.uniform(150, 300)
        elif d.month == 11 and 25 <= d.day <= 30:
            holiday_boost[i] = np.random.uniform(100, 200)

    sales = trend + weekly + yearly + noise + holiday_boost
    sales = np.clip(sales, 500, None)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales.round(2),
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "year": dates.year,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "is_holiday": (holiday_boost > 0).astype(int),
    })
    logger.info(f"✅ Generated {n_days} days of synthetic retail sales.")
    return df


# ─── CSV Loader ──────────────────────────────────────────────────────────────

def load_csv_data(path: str = None) -> pd.DataFrame:
    """Load and clean retail sales data from a CSV file."""
    path = path or os.getenv("DATA_PATH", "./data/retail_sales.csv")
    if not Path(path).exists():
        logger.warning(f"CSV not found at {path}. Generating synthetic data...")
        return generate_synthetic_sales()

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.columns = [c.lower().strip() for c in df.columns]

    if "sales" not in df.columns:
        raise ValueError("CSV must contain a 'sales' column.")

    logger.info(f"✅ Loaded {len(df)} rows from {path}")
    return df


# ─── Neon DB Loader ──────────────────────────────────────────────────────────

def load_from_neon() -> pd.DataFrame:
    """Load sales data from Neon PostgreSQL database."""
    try:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL not set.")
        engine = create_engine(db_url)
        query = "SELECT date, sales FROM retail_sales ORDER BY date ASC;"
        df = pd.read_sql(text(query), engine, parse_dates=["date"])
        logger.info(f"✅ Loaded {len(df)} rows from Neon DB.")
        return df
    except Exception as e:
        logger.error(f"Neon DB load failed: {e}. Falling back to CSV/synthetic.")
        return load_csv_data()


# ─── Feature Engineering ─────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for model improvement."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter

    # Rolling statistics
    df["sales_7d_ma"] = df["sales"].rolling(7, min_periods=1).mean()
    df["sales_30d_ma"] = df["sales"].rolling(30, min_periods=1).mean()
    df["sales_7d_std"] = df["sales"].rolling(7, min_periods=1).std().fillna(0)

    # Lag features
    df["sales_lag_1"] = df["sales"].shift(1).bfill()
    df["sales_lag_7"] = df["sales"].shift(7).bfill()

    logger.info("✅ Feature engineering complete.")
    return df


# ─── Train/Test Split ────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, train_ratio: float = None):
    """Split into train/test sets chronologically."""
    ratio = train_ratio or float(os.getenv("TRAIN_SPLIT", 0.8))
    split_idx = int(len(df) * ratio)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"✅ Split: {len(train)} train rows, {len(test)} test rows.")
    return train, test


# ─── LSTM Sequence Prep ──────────────────────────────────────────────────────

def create_sequences(series: np.ndarray, lookback: int = 60):
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def scale_data(train_series: np.ndarray, test_series: np.ndarray):
    """Scale sales data to [0, 1] for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.reshape(-1, 1))
    return train_scaled, test_scaled, scaler


# ─── Main Entry ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_csv_data()
    df = add_features(df)
    train, test = split_data(df)
    print(df.head())
    print(f"\nTrain: {train.shape} | Test: {test.shape}")
