"""
models/lstm_model.py
====================
Bidirectional LSTM with attention for retail sales forecasting.
Includes early stopping, learning rate scheduling, and model checkpointing.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logs

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()


# ─── Model Builder ───────────────────────────────────────────────────────────

def build_lstm_model(lookback: int, units: int = 128) -> tf.keras.Model:
    """
    Stacked Bidirectional LSTM with dropout and batch normalization.
    Architecture: BiLSTM → BiLSTM → Dense → Output
    """
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True, input_shape=(lookback, 1))),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(units // 2, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.2),

        Bidirectional(LSTM(units // 4, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.1),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),  # Forecast output
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="huber",           # Robust to outliers vs MSE
        metrics=["mae"],
    )
    model.summary(print_fn=logger.info)
    return model


# ─── LSTM Forecaster ─────────────────────────────────────────────────────────

class LSTMForecaster:
    """
    End-to-end LSTM forecaster for retail time series.
    Handles scaling, sequencing, training, and multi-step prediction.
    """

    def __init__(self):
        self.lookback = int(os.getenv("LSTM_LOOKBACK", 60))
        self.epochs = int(os.getenv("LSTM_EPOCHS", 50))
        self.batch_size = int(os.getenv("LSTM_BATCH_SIZE", 32))
        self.units = int(os.getenv("LSTM_UNITS", 128))
        self.save_path = os.getenv("MODEL_SAVE_PATH", "./models/saved").rstrip("/").rstrip("\\")
        self.model = None
        self.history = None
        self.scaler = None

    def _get_callbacks(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=f"{self.save_path}/lstm_best.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

    def fit(self, train_scaled: np.ndarray, scaler) -> "LSTMForecaster":
        """Train LSTM on scaled training data."""
        self.scaler = scaler
        from data.data_loader import create_sequences

        X_train, y_train = create_sequences(train_scaled.flatten(), self.lookback)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        logger.info(f"🔄 Training LSTM | X: {X_train.shape} | Epochs: {self.epochs}")
        self.model = build_lstm_model(self.lookback, self.units)

        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=self._get_callbacks(),
            verbose=1,
            shuffle=False,  # Time series — no shuffle
        )
        logger.success("✅ LSTM training complete.")
        return self

    def predict(self, train_scaled: np.ndarray, test_scaled: np.ndarray) -> np.ndarray:
        """
        One-step-ahead prediction using true past values in each window.
        Each test prediction uses [lookback real values] → predict next.
        This is the correct evaluation strategy — no error compounding.
        """
        train_flat = train_scaled.flatten()
        test_flat = test_scaled.flatten()
        full = np.concatenate([train_flat, test_flat])

        predictions_scaled = []
        for i in range(len(test_flat)):
            # Window ends at train_end + i, uses only TRUE past values
            start = len(train_flat) + i - self.lookback
            end = len(train_flat) + i
            window = full[start:end].reshape(1, self.lookback, 1)
            pred = self.model.predict(window, verbose=0)[0][0]
            predictions_scaled.append(pred)

        pred_array = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(pred_array).flatten()
        logger.info(f"✅ LSTM predicted {len(predictions)} values.")
        return predictions

    def predict_future(self, last_sequence: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Auto-regressive future forecasting — predict beyond test data.
        """
        predictions = []
        seq = last_sequence.copy().flatten().tolist()

        for _ in range(n_steps):
            x = np.array(seq[-self.lookback:]).reshape(1, self.lookback, 1)
            pred = self.model.predict(x, verbose=0)[0][0]
            predictions.append(pred)
            seq.append(pred)

        predictions_array = np.array(predictions).reshape(-1, 1)
        return self.scaler.inverse_transform(predictions_array).flatten()

    def save(self, path: str = None):
        path = path or self.save_path
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(f"{path}/lstm_model.keras")
        logger.info(f"💾 LSTM saved to {path}/lstm_model.keras")

    @classmethod
    def load(cls, path: str = None):
        obj = cls()
        path = path or obj.save_path
        obj.model = load_model(f"{path}/lstm_model.keras")
        logger.info(f"✅ LSTM loaded from {path}")
        return obj


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_lstm(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted)
    metrics = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2), "R2": round(r2, 4)}
    logger.info(f"📊 LSTM Metrics → {metrics}")
    return metrics


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data.data_loader import load_csv_data, add_features, split_data, scale_data

    df = add_features(load_csv_data())
    train, test = split_data(df)

    train_scaled, test_scaled, scaler = scale_data(
        train["sales"].values, test["sales"].values
    )

    forecaster = LSTMForecaster()
    forecaster.fit(train_scaled, scaler)
    predictions = forecaster.predict(train_scaled, test_scaled)

    metrics = evaluate_lstm(test["sales"].values, predictions)
    print("\nLSTM Evaluation:", metrics)
