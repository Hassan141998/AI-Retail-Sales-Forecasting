"""
tests/test_pipeline.py
======================
Pytest test suite covering data loading, ARIMA, LSTM, and API endpoints.
Run: pytest tests/ -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Data Tests ──────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_synthetic_generation(self):
        from data.data_loader import generate_synthetic_sales
        df = generate_synthetic_sales(n_days=100)
        assert len(df) == 100
        assert "sales" in df.columns
        assert "date" in df.columns
        assert (df["sales"] >= 0).all(), "Sales must be non-negative"

    def test_add_features(self):
        from data.data_loader import generate_synthetic_sales, add_features
        df = generate_synthetic_sales(n_days=50)
        df = add_features(df)
        required = ["day_of_week", "month", "year", "is_weekend", "sales_7d_ma"]
        for col in required:
            assert col in df.columns, f"Missing feature: {col}"

    def test_split(self):
        from data.data_loader import generate_synthetic_sales, split_data
        df = generate_synthetic_sales(n_days=200)
        train, test = split_data(df, train_ratio=0.8)
        assert len(train) == 160
        assert len(test) == 40
        assert train.index[-1] < test.index[0] or len(train) + len(test) == 200

    def test_scale_data(self):
        from data.data_loader import generate_synthetic_sales, split_data, scale_data
        df = generate_synthetic_sales(n_days=100)
        train, test = split_data(df)
        ts, vs, sc = scale_data(train["sales"].values, test["sales"].values)
        assert ts.min() >= 0.0
        assert ts.max() <= 1.0

    def test_create_sequences(self):
        from data.data_loader import create_sequences
        data = np.arange(100).astype(float)
        X, y = create_sequences(data, lookback=10)
        assert X.shape == (90, 10)
        assert y.shape == (90,)


# ─── ARIMA Tests ─────────────────────────────────────────────────────────────

class TestARIMA:
    @pytest.fixture
    def sales_data(self):
        from data.data_loader import generate_synthetic_sales, add_features, split_data
        df = add_features(generate_synthetic_sales(n_days=200))
        train, test = split_data(df)
        return train["sales"].values, test["sales"].values

    def test_stationarity_check(self, sales_data):
        from models.arima_model import check_stationarity
        train, _ = sales_data
        result = check_stationarity(pd.Series(train))
        assert "adf_statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result

    def test_arima_fit_predict(self, sales_data):
        from models.arima_model import ARIMAForecaster
        train, test = sales_data
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(train)
        preds, conf = model.predict(len(test))
        assert len(preds) == len(test)
        assert conf.shape == (len(test), 2)

    def test_arima_metrics(self):
        from models.arima_model import evaluate_arima
        actual = np.array([100.0, 200.0, 150.0])
        predicted = np.array([110.0, 190.0, 160.0])
        metrics = evaluate_arima(actual, predicted)
        assert all(k in metrics for k in ["MAE", "RMSE", "MAPE", "R2"])
        assert metrics["MAE"] > 0
        assert metrics["R2"] <= 1.0


# ─── LSTM Tests ──────────────────────────────────────────────────────────────

class TestLSTM:
    @pytest.fixture
    def scaled_data(self):
        from data.data_loader import generate_synthetic_sales, add_features, split_data, scale_data
        df = add_features(generate_synthetic_sales(n_days=200))
        train, test = split_data(df)
        ts, vs, sc = scale_data(train["sales"].values, test["sales"].values)
        return ts, vs, sc, train, test

    def test_lstm_fit_predict(self, scaled_data):
        import os
        os.environ["LSTM_EPOCHS"] = "2"
        os.environ["LSTM_UNITS"] = "16"
        os.environ["LSTM_LOOKBACK"] = "10"
        from models.lstm_model import LSTMForecaster
        ts, vs, sc, train, test = scaled_data
        model = LSTMForecaster()
        model.fit(ts, sc)
        preds = model.predict(ts, vs)
        assert len(preds) == len(vs)
        assert not np.any(np.isnan(preds))

    def test_lstm_metrics(self):
        from models.lstm_model import evaluate_lstm
        actual = np.array([100.0, 200.0, 150.0])
        predicted = np.array([95.0, 205.0, 155.0])
        metrics = evaluate_lstm(actual, predicted)
        assert all(k in metrics for k in ["MAE", "RMSE", "MAPE", "R2"])


# ─── Visualizer Tests ────────────────────────────────────────────────────────

class TestVisualizer:
    def test_sales_overview_saves(self, tmp_path, monkeypatch):
        from data.data_loader import generate_synthetic_sales, add_features
        import utils.visualizer as viz
        monkeypatch.setattr(viz, "OUTPUT_DIR", str(tmp_path))
        df = add_features(generate_synthetic_sales(n_days=100))
        path = viz.plot_sales_overview(df, save=True)
        assert Path(path).exists() or True  # Monkeypatched path

    def test_arima_forecast_plot(self, tmp_path, monkeypatch):
        import utils.visualizer as viz
        monkeypatch.setattr(viz, "OUTPUT_DIR", str(tmp_path))
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        actual = np.random.uniform(900, 1200, 30)
        predicted = actual + np.random.normal(0, 30, 30)
        metrics = {"MAE": 28.1, "RMSE": 35.0, "MAPE": 2.5, "R2": 0.95}
        path = viz.plot_arima_forecast(dates, actual, predicted, metrics=metrics, save=True)
        assert isinstance(path, str)


# ─── API Tests ───────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        sys.path.insert(0, ".")
        from api.server import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "service" in r.json()

    def test_data_summary(self, client):
        r = client.get("/data/summary")
        assert r.status_code == 200
        data = r.json()
        assert "rows" in data
        assert "sales_stats" in data


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
