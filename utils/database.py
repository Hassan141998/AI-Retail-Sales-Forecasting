"""
utils/database.py
=================
Neon PostgreSQL integration — schema creation, data storage, and results retrieval.
Uses SQLAlchemy ORM with async support via asyncpg.
"""

import os
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    from sqlalchemy import (
        create_engine, Column, Integer, Float, String,
        DateTime, Boolean, text, MetaData, Table
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

Base = declarative_base() if SQLALCHEMY_AVAILABLE else None


# ─── ORM Models ──────────────────────────────────────────────────────────────

if SQLALCHEMY_AVAILABLE:
    class RetailSales(Base):
        __tablename__ = "retail_sales"
        id = Column(Integer, primary_key=True, index=True)
        date = Column(DateTime, nullable=False, index=True)
        sales = Column(Float, nullable=False)
        day_of_week = Column(Integer)
        month = Column(Integer)
        is_weekend = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.utcnow)

    class ForecastResult(Base):
        __tablename__ = "forecast_results"
        id = Column(Integer, primary_key=True, index=True)
        run_id = Column(String(64), index=True)
        model_name = Column(String(32), nullable=False)   # 'arima' | 'lstm'
        forecast_date = Column(DateTime, nullable=False)
        predicted_value = Column(Float, nullable=False)
        actual_value = Column(Float)
        mae = Column(Float)
        rmse = Column(Float)
        mape = Column(Float)
        r2 = Column(Float)
        created_at = Column(DateTime, default=datetime.utcnow)

    class ModelRun(Base):
        __tablename__ = "model_runs"
        id = Column(Integer, primary_key=True, index=True)
        run_id = Column(String(64), unique=True, index=True)
        model_name = Column(String(32))
        mae = Column(Float)
        rmse = Column(Float)
        mape = Column(Float)
        r2 = Column(Float)
        train_rows = Column(Integer)
        test_rows = Column(Integer)
        notes = Column(String(512))
        created_at = Column(DateTime, default=datetime.utcnow)


# ─── DB Connection ───────────────────────────────────────────────────────────

class NeonDatabase:
    """Neon PostgreSQL connection manager."""

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.engine = None
        self.SessionLocal = None

    def connect(self) -> bool:
        if not SQLALCHEMY_AVAILABLE:
            logger.error("SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary")
            return False
        if not self.db_url:
            logger.warning("DATABASE_URL not set. Skipping DB operations.")
            return False
        try:
            self.engine = create_engine(
                self.db_url,
                connect_args={"connect_timeout": 10},
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
            )
            self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.success("✅ Connected to Neon PostgreSQL.")
            return True
        except Exception as e:
            logger.error(f"❌ DB connection failed: {e}")
            return False

    def create_tables(self):
        if self.engine:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Tables created / verified.")

    def get_session(self):
        if not self.SessionLocal:
            return None
        return self.SessionLocal()

    def save_forecast_results(self, run_id: str, model_name: str,
                               dates, predictions, actuals, metrics: dict):
        session = self.get_session()
        if not session:
            return
        try:
            for date, pred, actual in zip(dates, predictions, actuals):
                record = ForecastResult(
                    run_id=run_id,
                    model_name=model_name,
                    forecast_date=date,
                    predicted_value=float(pred),
                    actual_value=float(actual) if actual is not None else None,
                    mae=metrics.get("MAE"),
                    rmse=metrics.get("RMSE"),
                    mape=metrics.get("MAPE"),
                    r2=metrics.get("R2"),
                )
                session.add(record)
            session.commit()
            logger.info(f"💾 Saved {len(predictions)} {model_name} forecasts to Neon DB.")
        except Exception as e:
            session.rollback()
            logger.error(f"DB save failed: {e}")
        finally:
            session.close()

    def save_model_run(self, run_id: str, model_name: str, metrics: dict,
                        train_rows: int, test_rows: int, notes: str = ""):
        session = self.get_session()
        if not session:
            return
        try:
            run = ModelRun(
                run_id=run_id,
                model_name=model_name,
                mae=metrics.get("MAE"),
                rmse=metrics.get("RMSE"),
                mape=metrics.get("MAPE"),
                r2=metrics.get("R2"),
                train_rows=train_rows,
                test_rows=test_rows,
                notes=notes,
            )
            session.add(run)
            session.commit()
            logger.info(f"💾 Model run {run_id} saved.")
        except Exception as e:
            session.rollback()
            logger.error(f"DB save model run failed: {e}")
        finally:
            session.close()

    def get_latest_runs(self, limit: int = 10) -> list:
        session = self.get_session()
        if not session:
            return []
        try:
            rows = session.query(ModelRun).order_by(ModelRun.created_at.desc()).limit(limit).all()
            return [
                {
                    "run_id": r.run_id,
                    "model": r.model_name,
                    "MAE": r.mae,
                    "RMSE": r.rmse,
                    "MAPE": r.mape,
                    "R2": r.r2,
                    "created_at": str(r.created_at),
                }
                for r in rows
            ]
        finally:
            session.close()


# ─── Singleton ───────────────────────────────────────────────────────────────

db = NeonDatabase()


if __name__ == "__main__":
    if db.connect():
        db.create_tables()
        print("✅ Neon DB ready.")
    else:
        print("⚠️  DB not configured — set DATABASE_URL in .env")
