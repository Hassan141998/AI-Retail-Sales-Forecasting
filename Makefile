# ============================================================
# Sales Forecast — Makefile
# ============================================================

.PHONY: install train train-arima train-lstm api test clean deploy

# Setup
install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "✅ Environment ready. Activate with: source .venv/bin/activate"

# Training
train:
	python train.py --model both

train-arima:
	python train.py --model arima

train-lstm:
	python train.py --model lstm

train-db:
	python train.py --model both --save-db

# API
api:
	uvicorn api.server:app --reload --port 8000

# Tests
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=. --cov-report=html

# Database
db-init:
	python utils/database.py

# Deploy
deploy:
	vercel --prod

# Clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov
	@echo "🧹 Cleaned."

# Help
help:
	@echo "Available commands:"
	@echo "  make install      — Create venv and install dependencies"
	@echo "  make train        — Train both ARIMA and LSTM"
	@echo "  make train-arima  — Train ARIMA only"
	@echo "  make train-lstm   — Train LSTM only"
	@echo "  make api          — Start FastAPI server"
	@echo "  make test         — Run pytest suite"
	@echo "  make deploy       — Deploy to Vercel"
	@echo "  make clean        — Remove cache files"
