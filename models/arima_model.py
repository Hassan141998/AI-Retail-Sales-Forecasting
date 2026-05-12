"""
models/arima_model.py
=====================
ARIMA & Auto-ARIMA for retail sales forecasting.
Uses statsmodels + pmdarima for auto order selection.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

warnings.filterwarnings("ignore")
load_dotenv()


# ─── Stationarity Test ───────────────────────────────────────────────────────

def check_stationarity(series: pd.Series) -> dict:
    """Run Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    output = {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
        "critical_values": result[4],
    }
    status = "✅ Stationary" if output["is_stationary"] else "⚠️  Non-Stationary"
    logger.info(f"ADF Test → p={output['p_value']} → {status}")
    return output


# ─── Auto ARIMA ──────────────────────────────────────────────────────────────

class AutoARIMAForecaster:
    """
    Auto-selects best ARIMA order using AIC/BIC.
    Wraps pmdarima.auto_arima for convenience.
    """

    def __init__(self, seasonal: bool = True, m: int = 7):
        self.seasonal = seasonal
        self.m = m  # Weekly seasonality for retail
        self.model = None
        self.order = None
        self.seasonal_order = None

    def fit(self, train_series: np.ndarray) -> "AutoARIMAForecaster":
        logger.info("🔄 Running Auto-ARIMA order selection...")
        self.model = pm.auto_arima(
            train_series,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            d=None,                    # Auto-determine differencing
            seasonal=self.seasonal,
            m=self.m,
            D=1,
            start_P=0, start_Q=0,
            max_P=2, max_Q=2,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_jobs=-1,
        )
        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order
        logger.success(f"✅ Best ARIMA order: {self.order} | Seasonal: {self.seasonal_order}")
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        forecast, conf_int = self.model.predict(n_periods=n_periods, return_conf_int=True)
        return forecast, conf_int

    def save(self, path: str = None):
        path = path or os.getenv("MODEL_SAVE_PATH", "./models/saved/")
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, f"{path}/auto_arima.pkl")
        logger.info(f"💾 Auto-ARIMA saved to {path}/auto_arima.pkl")

    @classmethod
    def load(cls, path: str = None):
        path = path or os.getenv("MODEL_SAVE_PATH", "./models/saved/")
        obj = cls()
        obj.model = joblib.load(f"{path}/auto_arima.pkl")
        return obj


# ─── Manual ARIMA ────────────────────────────────────────────────────────────

class ARIMAForecaster:
    """
    Manual ARIMA with explicit (p, d, q) order from .env or constructor.
    Uses statsmodels SARIMAX for flexibility.
    """

    def __init__(self, order: tuple = None):
        p = int(os.getenv("ARIMA_ORDER_P", 2))
        d = int(os.getenv("ARIMA_ORDER_D", 1))
        q = int(os.getenv("ARIMA_ORDER_Q", 2))
        self.order = order or (p, d, q)
        self.model_fit = None
        self.train_series = None

    def fit(self, train_series: np.ndarray) -> "ARIMAForecaster":
        self.train_series = train_series

        # Auto-detect differencing if data is non-stationary
        from statsmodels.tsa.stattools import adfuller
        p_value = adfuller(train_series)[1]
        if p_value > 0.05 and self.order[1] == 0:
            logger.warning("Non-stationary data detected — forcing d=1")
            self.order = (self.order[0], 1, self.order[2])

        logger.info(f"🔄 Fitting ARIMA{self.order}...")
        model = SARIMAX(
            train_series,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)
        logger.success(f"✅ ARIMA fitted. AIC={self.model_fit.aic:.2f}")
        return self

    def predict(self, n_periods: int) -> np.ndarray:
        """Multi-step future forecast with confidence intervals."""
        forecast = self.model_fit.get_forecast(steps=n_periods)
        mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        return np.array(mean), np.array(conf_int)

    def predict_rolling(self, test_series: np.ndarray) -> np.ndarray:
        """
        Rolling one-step-ahead forecast — the correct way to evaluate ARIMA.
        After each step, the true value is fed back so the model never
        compounds multi-step errors. Produces realistic MAE/RMSE/R2 scores.
        """
        history = list(self.train_series.copy())
        predictions = []
        n = len(test_series)
        logger.info(f"🔄 Rolling ARIMA evaluation over {n} steps...")
        for i, actual in enumerate(test_series):
            model = SARIMAX(history, order=self.order,
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            yhat = float(fit.forecast(steps=1)[0])
            predictions.append(yhat)
            history.append(actual)
            if (i + 1) % 30 == 0:
                logger.info(f"   Step {i+1}/{n}")
        logger.success("✅ Rolling ARIMA forecast complete.")
        return np.array(predictions)

    def summary(self) -> str:
        return str(self.model_fit.summary())

    def save(self, path: str = None):
        path = (path or os.getenv("MODEL_SAVE_PATH", "./models/saved")).rstrip("/").rstrip("\\")
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model_fit.save(f"{path}/arima_model.pkl")
        logger.info(f"💾 ARIMA saved.")

    @classmethod
    def load(cls, path: str = None):
        from statsmodels.tsa.statespace.sarimax import SARIMAXResults
        path = path or os.getenv("MODEL_SAVE_PATH", "./models/saved/")
        obj = cls()
        obj.model_fit = SARIMAXResults.load(f"{path}/arima_model.pkl")
        return obj


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_arima(actual: np.ndarray, predicted: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted)
    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "R2": round(r2, 4)}
    logger.info(f"📊 ARIMA Metrics → {metrics}")
    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data.data_loader import load_csv_data, add_features, split_data

    df = add_features(load_csv_data())
    train, test = split_data(df)

    train_sales = train["sales"].values
    test_sales = test["sales"].values

    # Stationarity check
    check_stationarity(pd.Series(train_sales))

    # Fit and predict
    model = ARIMAForecaster()
    model.fit(train_sales)
    forecast, conf = model.predict(len(test_sales))

    metrics = evaluate_arima(test_sales, forecast)
    print("\nARIMA Evaluation:", metrics)
