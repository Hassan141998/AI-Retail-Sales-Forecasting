# 📈 SalesPulse — AI Retail Sales Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Deploy-black?style=flat-square&logo=vercel)
![Neon](https://img.shields.io/badge/Neon-PostgreSQL-00E5A0?style=flat-square)

**ARIMA + Bidirectional LSTM · FastAPI · Neon PostgreSQL · Vercel · Animated Dashboard**

</div>

---

## 🗂 Project Structure

```
sales_forecast/
├── api/
│   └── server.py           # FastAPI REST API (Vercel-deployable)
├── data/
│   └── data_loader.py      # CSV / Neon DB / synthetic data loading
├── dashboard/
│   └── index.html          # Animated professional dashboard UI
├── models/
│   ├── arima_model.py      # ARIMA + Auto-ARIMA forecaster
│   ├── lstm_model.py       # Bidirectional LSTM forecaster
│   └── saved/              # Serialized model checkpoints
├── static/
│   └── plots/              # Generated matplotlib / plotly charts
├── tests/
│   └── test_pipeline.py    # Pytest test suite
├── utils/
│   ├── database.py         # Neon PostgreSQL ORM (SQLAlchemy)
│   └── visualizer.py       # Matplotlib dark-theme + Plotly dashboard
├── logs/                   # Rotating log files
├── .env                    # Local environment variables (git-ignored)
├── .env.example            # Template for new developers
├── train.py                # Main CLI training pipeline
├── requirements.txt        # Python dependencies
├── vercel.json             # Vercel deployment config
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone & Install

```bash
git clone https://github.com/your-username/sales-forecast.git
cd sales-forecast

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env with your Neon DB credentials and settings
```

### 3 — Train Models

```bash
# Train both ARIMA and LSTM (uses synthetic data by default)
python train.py --model both

# Train only ARIMA
python train.py --model arima

# Train LSTM only, save results to Neon DB
python train.py --model lstm --save-db
```

### 4 — Start API Server

```bash
python -m uvicorn api.server:app --reload --port 8000
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 5 — Open Dashboard

```bash
# Open dashboard/index.html in your browser directly, or serve it:
python -m http.server 3000 --directory dashboard
# Visit: http://localhost:3000
```

---

## 🤖 Models

### ARIMA (AutoRegressive Integrated Moving Average)
- Manual ARIMA with configurable (p, d, q) via `.env`
- **Auto-ARIMA** using `pmdarima` — automatically selects best order via AIC
- SARIMA support for seasonal retail patterns (weekly seasonality m=7)
- Augmented Dickey-Fuller stationarity test built-in
- 95% confidence interval generation

### Bidirectional LSTM
- **2-layer Stacked Bidirectional LSTM** architecture
- Huber loss (robust to outliers vs MSE)
- Batch Normalization + Dropout (0.1–0.2)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Auto-regressive multi-step future forecasting
- Configurable lookback window (default: 60 days)

### Evaluation Metrics
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |

---

## 🗄 Database — Neon PostgreSQL

**Why Neon?** Serverless PostgreSQL with branching, zero cold starts, and a generous free tier. Perfect for this workload.

### Setup

1. Create account at [neon.tech](https://neon.tech)
2. Create a new project → copy the connection string
3. Paste into `.env` as `DATABASE_URL`

```bash
# Initialize schema
python utils/database.py
```

### Schema

```sql
-- retail_sales        : raw sales data
-- forecast_results    : per-date predictions with metrics
-- model_runs          : training run history with all metrics
```

---

## ☁️ Deployment — Vercel

**Why Vercel?** Zero-config Python serverless functions, auto-SSL, CDN, preview deployments per PR.

### Deploy

```bash
npm i -g vercel
vercel login

# First deploy
vercel

# Production deploy
vercel --prod
```

### Environment Variables (Vercel Dashboard)

```
DATABASE_URL     →  your-neon-connection-string
APP_ENV          →  production
DEBUG            →  false
FORECAST_HORIZON →  30
```

> **Note**: Heavy model training should run locally or on a dedicated VM.
> Vercel functions are best for inference and serving pre-trained models.

---

## 📊 Visualization Outputs

| File | Description |
|------|-------------|
| `static/plots/sales_overview.png` | 3-panel raw data overview |
| `static/plots/arima_forecast.png` | ARIMA forecast + residuals |
| `static/plots/lstm_forecast.png` | LSTM forecast + scatter |
| `static/plots/model_comparison.png` | Side-by-side comparison |
| `static/plots/training_history.png` | LSTM loss/MAE curves |
| `static/plots/interactive_dashboard.html` | Plotly interactive dashboard |

---

## 🧪 Tests

```bash
pytest tests/ -v
pytest tests/ -v --tb=short   # Short traceback
pytest tests/test_pipeline.py::TestARIMA -v   # Run just ARIMA tests
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/forecast` | Run forecast |
| GET | `/data/summary` | Dataset statistics |
| GET | `/results/latest` | Recent model runs |
| GET | `/plots/{name}` | Serve a chart image |

### Forecast Request

```json
POST /forecast
{
  "model": "both",       // "arima" | "lstm" | "both"
  "horizon": 30,         // days to forecast
  "save_db": false       // save to Neon DB
}
```

---

## ⚙️ Configuration (.env)

```dotenv
# Model Tuning
ARIMA_ORDER_P=5
ARIMA_ORDER_D=1
ARIMA_ORDER_Q=0
LSTM_EPOCHS=50
LSTM_LOOKBACK=60
LSTM_UNITS=128

# Data
TRAIN_SPLIT=0.8
FORECAST_HORIZON=30
```

---

## 📦 Bringing Your Own Data

Drop a CSV at `./data/retail_sales.csv` with at minimum:

```csv
date,sales
2022-01-01,1042.50
2022-01-02,987.30
...
```

The loader auto-detects the file and falls back to synthetic data if not found.

---

## 🛠 PyCharm Setup

1. Open the project folder in PyCharm
2. Go to **Settings → Project → Python Interpreter** → add `.venv`
3. Set **Run Configuration** → Script: `train.py`, Parameters: `--model both`
4. For the API: Script: `api/server.py`
5. Install `.env` plugin for automatic env loading

---

## 🗺 Roadmap

- [ ] Prophet model integration
- [ ] Multi-store / multi-SKU support
- [ ] Streamlit interactive UI option
- [ ] Docker Compose setup
- [ ] GitHub Actions CI/CD pipeline
- [ ] Anomaly detection overlay

---

## 📜 License

MIT © 2024 — Built for learning and production use.
