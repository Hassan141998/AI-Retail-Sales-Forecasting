"""
utils/visualizer.py
===================
All matplotlib / plotly visualizations for the sales forecast project.
Generates publication-quality static charts and interactive HTML plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from loguru import logger

# ─── Style Setup ─────────────────────────────────────────────────────────────

DARK_THEME = {
    "bg": "#0a0e1a",
    "panel": "#111827",
    "grid": "#1e2d40",
    "text": "#e2e8f0",
    "accent": "#38bdf8",
    "green": "#34d399",
    "red": "#f87171",
    "purple": "#a78bfa",
    "orange": "#fb923c",
}

OUTPUT_DIR = "./static/plots"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def _apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(DARK_THEME["bg"])
    # Handle list, numpy array, or single Axes object
    if hasattr(ax_list, '__iter__') and not hasattr(ax_list, 'set_facecolor'):
        axes_flat = list(np.array(ax_list).flatten())
    else:
        axes_flat = [ax_list]
    for ax in axes_flat:
        ax.set_facecolor(DARK_THEME["panel"])
        ax.tick_params(colors=DARK_THEME["text"])
        ax.xaxis.label.set_color(DARK_THEME["text"])
        ax.yaxis.label.set_color(DARK_THEME["text"])
        ax.title.set_color(DARK_THEME["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME["grid"])
        ax.grid(True, color=DARK_THEME["grid"], alpha=0.5, linestyle="--", linewidth=0.5)


# ─── 1. Raw Sales Overview ───────────────────────────────────────────────────

def plot_sales_overview(df: pd.DataFrame, save: bool = True) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle("Retail Sales — Data Overview", color=DARK_THEME["text"],
                 fontsize=18, fontweight="bold", y=0.98)
    _apply_dark_style(fig, axes)

    # Raw sales
    axes[0].plot(df["date"], df["sales"], color=DARK_THEME["accent"],
                 linewidth=0.8, alpha=0.9)
    axes[0].fill_between(df["date"], df["sales"], alpha=0.15,
                         color=DARK_THEME["accent"])
    axes[0].set_title("Daily Sales Volume", pad=8)
    axes[0].set_ylabel("Sales ($)")

    # 7-day moving average
    if "sales_7d_ma" in df.columns:
        axes[1].plot(df["date"], df["sales_7d_ma"], color=DARK_THEME["green"],
                     linewidth=1.5, label="7-Day MA")
        axes[1].plot(df["date"], df["sales_30d_ma"], color=DARK_THEME["orange"],
                     linewidth=1.5, label="30-Day MA")
        axes[1].legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
                       labelcolor=DARK_THEME["text"])
        axes[1].set_title("Moving Averages")
        axes[1].set_ylabel("Sales ($)")

    # Monthly distribution
    monthly = df.groupby("month")["sales"].mean()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = [DARK_THEME["green"] if v > monthly.mean() else DARK_THEME["red"]
              for v in monthly.values]
    axes[2].bar(range(1, 13), monthly.values, color=colors, edgecolor=DARK_THEME["bg"])
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(months, color=DARK_THEME["text"])
    axes[2].set_title("Average Monthly Sales")
    axes[2].set_ylabel("Avg Sales ($)")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/sales_overview.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_THEME["bg"])
        plt.close()
        logger.info(f"📊 Saved: {path}")
    return path


# ─── 2. ARIMA Forecast Plot ──────────────────────────────────────────────────

def plot_arima_forecast(test_dates, actual, predicted, conf_int=None,
                        metrics: dict = None, save: bool = True) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    _apply_dark_style(fig, axes)
    fig.suptitle("ARIMA Forecast vs Actual", color=DARK_THEME["text"],
                 fontsize=16, fontweight="bold")

    # Main forecast
    axes[0].plot(test_dates, actual, color=DARK_THEME["text"], linewidth=1.5,
                 label="Actual", alpha=0.9)
    axes[0].plot(test_dates, predicted, color=DARK_THEME["accent"], linewidth=2,
                 label="ARIMA Forecast", linestyle="--")
    if conf_int is not None:
        axes[0].fill_between(test_dates, conf_int[:, 0], conf_int[:, 1],
                             alpha=0.15, color=DARK_THEME["accent"],
                             label="95% CI")
    axes[0].legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
                   labelcolor=DARK_THEME["text"])
    axes[0].set_ylabel("Sales ($)")
    axes[0].set_title("Forecast vs Actuals")

    if metrics:
        info = f"MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | MAPE: {metrics['MAPE']}% | R²: {metrics['R2']}"
        axes[0].set_xlabel(info, color=DARK_THEME["accent"], fontsize=10)

    # Residuals
    residuals = actual - predicted
    axes[1].bar(test_dates, residuals,
                color=[DARK_THEME["green"] if r >= 0 else DARK_THEME["red"] for r in residuals],
                alpha=0.7, width=1)
    axes[1].axhline(0, color=DARK_THEME["text"], linewidth=1, linestyle="--")
    axes[1].set_title("Residuals")
    axes[1].set_ylabel("Error ($)")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/arima_forecast.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_THEME["bg"])
        plt.close()
        logger.info(f"📊 Saved: {path}")
    return path


# ─── 3. LSTM Forecast Plot ───────────────────────────────────────────────────

def plot_lstm_forecast(train_df, test_df, predictions, future_dates=None,
                       future_preds=None, metrics: dict = None, save: bool = True) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    _apply_dark_style(fig, axes)
    fig.suptitle("LSTM Bidirectional — Forecast", color=DARK_THEME["text"],
                 fontsize=16, fontweight="bold")

    # Full timeline
    axes[0].plot(train_df["date"], train_df["sales"], color=DARK_THEME["text"],
                 linewidth=0.8, alpha=0.6, label="Training Data")
    axes[0].plot(test_df["date"], test_df["sales"], color=DARK_THEME["green"],
                 linewidth=1.5, label="Actual (Test)")
    axes[0].plot(test_df["date"], predictions, color=DARK_THEME["accent"],
                 linewidth=2, linestyle="--", label="LSTM Forecast")

    if future_dates is not None and future_preds is not None:
        axes[0].plot(future_dates, future_preds, color=DARK_THEME["purple"],
                     linewidth=2, linestyle=":", label="Future Forecast")
        axes[0].axvline(x=test_df["date"].iloc[-1], color=DARK_THEME["orange"],
                        linestyle="--", alpha=0.7, linewidth=1.5)

    axes[0].legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
                   labelcolor=DARK_THEME["text"])
    axes[0].set_ylabel("Sales ($)")
    if metrics:
        axes[0].set_xlabel(
            f"MAE: {metrics['MAE']} | RMSE: {metrics['RMSE']} | MAPE: {metrics['MAPE']}% | R²: {metrics['R2']}",
            color=DARK_THEME["accent"], fontsize=10)

    # Scatter: actual vs predicted
    axes[1].scatter(test_df["sales"].values, predictions,
                    alpha=0.4, color=DARK_THEME["accent"], s=15, label="Predictions")
    min_v = min(test_df["sales"].min(), predictions.min())
    max_v = max(test_df["sales"].max(), predictions.max())
    axes[1].plot([min_v, max_v], [min_v, max_v], color=DARK_THEME["green"],
                 linewidth=1.5, label="Perfect Prediction")
    axes[1].set_xlabel("Actual Sales ($)")
    axes[1].set_ylabel("Predicted Sales ($)")
    axes[1].set_title("Actual vs Predicted Scatter")
    axes[1].legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
                   labelcolor=DARK_THEME["text"])

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/lstm_forecast.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_THEME["bg"])
        plt.close()
        logger.info(f"📊 Saved: {path}")
    return path


# ─── 4. Model Comparison ─────────────────────────────────────────────────────

def plot_model_comparison(arima_metrics: dict, lstm_metrics: dict,
                          test_dates, actual, arima_preds, lstm_preds,
                          save: bool = True) -> str:
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.patch.set_facecolor(DARK_THEME["bg"])
    fig.suptitle("Model Comparison: ARIMA vs LSTM", color=DARK_THEME["text"],
                 fontsize=18, fontweight="bold", y=0.98)

    # Forecast overlay
    ax1 = fig.add_subplot(gs[0, :])
    _apply_dark_style(fig, [ax1])
    ax1.plot(test_dates, actual, color=DARK_THEME["text"], linewidth=1.5,
             label="Actual", alpha=0.9)
    ax1.plot(test_dates, arima_preds, color=DARK_THEME["orange"], linewidth=2,
             linestyle="--", label="ARIMA")
    ax1.plot(test_dates, lstm_preds, color=DARK_THEME["accent"], linewidth=2,
             linestyle="-.", label="LSTM")
    ax1.legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
               labelcolor=DARK_THEME["text"])
    ax1.set_title("Forecast Comparison")
    ax1.set_ylabel("Sales ($)")

    # Metrics bar charts
    metric_names = ["MAE", "RMSE", "MAPE"]
    ax2 = fig.add_subplot(gs[1, 0])
    _apply_dark_style(fig, [ax2])
    x = np.arange(len(metric_names))
    width = 0.35
    arima_vals = [arima_metrics[m] for m in metric_names]
    lstm_vals = [lstm_metrics[m] for m in metric_names]
    ax2.bar(x - width / 2, arima_vals, width, label="ARIMA",
            color=DARK_THEME["orange"], edgecolor=DARK_THEME["bg"])
    ax2.bar(x + width / 2, lstm_vals, width, label="LSTM",
            color=DARK_THEME["accent"], edgecolor=DARK_THEME["bg"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, color=DARK_THEME["text"])
    ax2.legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
               labelcolor=DARK_THEME["text"])
    ax2.set_title("Error Metrics (Lower = Better)")

    # R² bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    _apply_dark_style(fig, [ax3])
    r2_vals = [arima_metrics["R2"], lstm_metrics["R2"]]
    colors = [DARK_THEME["orange"], DARK_THEME["accent"]]
    bars = ax3.bar(["ARIMA", "LSTM"], r2_vals, color=colors, edgecolor=DARK_THEME["bg"])
    for bar, val in zip(bars, r2_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", color=DARK_THEME["text"],
                 fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 1.1)
    ax3.set_title("R² Score (Higher = Better)")
    ax3.set_ylabel("R²")

    path = f"{OUTPUT_DIR}/model_comparison.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_THEME["bg"])
        plt.close()
        logger.info(f"📊 Saved: {path}")
    return path


# ─── 5. Training History ─────────────────────────────────────────────────────

def plot_training_history(history, save: bool = True) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, axes)
    fig.suptitle("LSTM Training History", color=DARK_THEME["text"],
                 fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"], color=DARK_THEME["accent"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], color=DARK_THEME["orange"], label="Val Loss")
    axes[0].set_title("Loss (Huber)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
                   labelcolor=DARK_THEME["text"])

    axes[1].plot(history.history["mae"], color=DARK_THEME["green"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], color=DARK_THEME["red"], label="Val MAE")
    axes[1].set_title("Mean Absolute Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(facecolor=DARK_THEME["panel"], edgecolor=DARK_THEME["grid"],
                   labelcolor=DARK_THEME["text"])

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/training_history.png"
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_THEME["bg"])
        plt.close()
        logger.info(f"📊 Saved: {path}")
    return path


# ─── 6. Interactive Plotly Dashboard ─────────────────────────────────────────

def create_interactive_dashboard(df: pd.DataFrame, test_dates, actual,
                                  arima_preds, lstm_preds,
                                  arima_metrics: dict, lstm_metrics: dict) -> str:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Sales Timeline", "ARIMA vs LSTM Forecast",
                        "Model Error Metrics", "Actual vs Predicted"),
        specs=[[{"colspan": 2}, None],
               [{}, {}]],
        vertical_spacing=0.15,
    )

    # Row 1: full sales
    fig.add_trace(go.Scatter(x=df["date"], y=df["sales"],
                             name="All Sales", line=dict(color="#38bdf8", width=1),
                             fill="tozeroy", fillcolor="rgba(56,189,248,0.1)"),
                  row=1, col=1)

    # Row 2: forecast comparison
    fig.add_trace(go.Scatter(x=test_dates, y=actual, name="Actual",
                             line=dict(color="#e2e8f0", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=arima_preds, name="ARIMA",
                             line=dict(color="#fb923c", width=2, dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=test_dates, y=lstm_preds, name="LSTM",
                             line=dict(color="#a78bfa", width=2, dash="dot")), row=2, col=1)

    # Row 2: metrics bar
    metrics_df = pd.DataFrame([arima_metrics, lstm_metrics],
                               index=["ARIMA", "LSTM"])[["MAE", "RMSE", "MAPE"]]
    for col_name, color in zip(["MAE", "RMSE", "MAPE"], ["#38bdf8", "#34d399", "#fb923c"]):
        fig.add_trace(go.Bar(name=col_name, x=["ARIMA", "LSTM"],
                             y=metrics_df[col_name].values,
                             marker_color=color), row=2, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="monospace"),
        title=dict(text="Sales Forecast Dashboard", font=dict(size=22, color="#38bdf8")),
        height=750,
        legend=dict(bgcolor="#111827", bordercolor="#1e2d40"),
        barmode="group",
    )

    path = f"{OUTPUT_DIR}/interactive_dashboard.html"
    fig.write_html(path)
    logger.info(f"📊 Interactive dashboard saved: {path}")
    return path
