"""
Database Module for MLOps Serving Infrastructure
Handles asynchronous PostgreSQL connections using asyncpg, schema initialization,
and logs model predictions and drift reports.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import asyncpg

logger = logging.getLogger("mlops_database")

# DB configuration from environment variables (matching docker-compose)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres_password")
DB_DATABASE = os.getenv("DB_DATABASE", "mlops_db")

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}"
)

# Global pool instance
_pool: Optional[asyncpg.Pool] = None

# Local memory fallbacks if PostgreSQL is unavailable
mock_prediction_logs = []
mock_drift_reports = []


async def init_db():
    """Initializes the database pool and creates necessary tables."""
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=10,
            command_timeout=30.0
        )
        logger.info("PostgreSQL database pool initialized successfully.")
        
        # Create tables
        async with _pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    user_id VARCHAR(100),
                    brand_context TEXT,
                    predicted_bid DOUBLE PRECISION,
                    is_blocked BOOLEAN
                );
                
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metrics JSONB,
                    dataset_drift_detected BOOLEAN
                );
            """)
            logger.info("Database schemas verified/created successfully.")
    except Exception as e:
        logger.warning(
            f"Could not connect to PostgreSQL database ({e}). "
            "Database operations will fall back to local in-memory logging."
        )
        _pool = None


async def close_db():
    """Closes the database pool connection."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL database pool closed.")


class PredictionLogger:
    """Logs model prediction requests and responses."""

    @staticmethod
    async def log_prediction(user_id: str, brand_context: str, predicted_bid: float, is_blocked: bool):
        """Logs prediction parameters to database or falls back to local memory."""
        global _pool
        if _pool:
            try:
                async with _pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO prediction_logs (user_id, brand_context, predicted_bid, is_blocked)
                        VALUES ($1, $2, $3, $4)
                        """,
                        user_id, brand_context, predicted_bid, is_blocked
                    )
                return
            except Exception as e:
                logger.error(f"Failed to log prediction to PostgreSQL: {e}")
                
        # Local fallback
        mock_prediction_logs.append({
            "user_id": user_id,
            "brand_context": brand_context,
            "predicted_bid": predicted_bid,
            "is_blocked": is_blocked
        })
        logger.info(f"[Mock DB Log] Prediction logged: user={user_id}, bid={predicted_bid}, blocked={is_blocked}")


class DriftReportStore:
    """Saves data drift reports and alerts."""

    @staticmethod
    async def save_drift_report(metrics: Dict[str, Any], dataset_drift_detected: bool):
        """Persists the drift report JSON and indicator to database or falls back."""
        global _pool
        metrics_json = json.dumps(metrics)
        
        if _pool:
            try:
                async with _pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO drift_reports (metrics, dataset_drift_detected)
                        VALUES ($1, $2)
                        """,
                        metrics_json, dataset_drift_detected
                    )
                return
            except Exception as e:
                logger.error(f"Failed to log drift report to PostgreSQL: {e}")
                
        # Local fallback
        mock_drift_reports.append({
            "metrics": metrics,
            "dataset_drift_detected": dataset_drift_detected
        })
        logger.info(f"[Mock DB Log] Drift report saved: drift_detected={dataset_drift_detected}")
