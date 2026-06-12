"""
FastAPI Server Entrypoint
Implements async endpoints for real-time bid scoring, serving,
drift evaluation checks, and model management.
"""

import uuid
import logging
import pickle
from contextlib import asynccontextmanager
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks

from schemas import BidRequest, BidResponse, DriftCheckRequest, DriftCheckResponse
from drift_detector import PopulationDriftMonitor
from database import init_db, close_db, PredictionLogger, DriftReportStore
from drift_reporter import EvidentlyDriftReporter

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown database lifecycle events."""
    logger.info("Starting FastAPI application serving layer...")
    await init_db()
    yield
    logger.info("Shutting down FastAPI application serving layer...")
    await close_db()


app = FastAPI(
    title="AdTech ML Serving Infrastructure",
    description="Asynchronous serving layer for real-time bidding decisions, drift monitoring, and model management.",
    version="1.0.0",
    lifespan=lifespan
)

# Mock models & heuristics local parameters
MARKET_INFLECTION_PRICE = 2.0
COMPETITIVENESS_SLOPE = 1.5


@app.get("/health")
async def health_check():
    """Simple API health diagnostic check endpoint."""
    return {"status": "HEALTHY", "version": "1.0.0"}


@app.post("/bid", response_model=BidResponse)
async def score_and_bid(request: BidRequest, background_tasks: BackgroundTasks):
    """
    Computes utility-maximizing bid and targets creative.
    Applies real-time brand safety logic.
    """
    try:
        # Check brand safety
        brand_safety_status = "SAFE"
        unsafe_keywords = ["violence", "hate speech", "illegal", "terrorism"]
        
        if any(w in request.brand_context.lower() for w in unsafe_keywords):
            brand_safety_status = "UNSAFE"
            response = BidResponse(
                bid_price=0.0,
                should_bid=False,
                targeted_creative_id="none",
                brand_safety_status=brand_safety_status
            )
        else:
            # Optimize bid price: b* = (value - bid) * P_win(bid)
            # Assuming Value of impression = estimated_ctr * 50
            val = request.estimated_ctr * 50.0
            bid_price = min(val * 0.6, MARKET_INFLECTION_PRICE * 1.2)
            
            # Determine creative targeting using feature vectors
            targeted_creative = "crt_101" if request.user_features[0] > 0 else "crt_102"
            
            response = BidResponse(
                bid_price=round(bid_price, 2),
                should_bid=bid_price > 0.1,
                targeted_creative_id=targeted_creative,
                brand_safety_status=brand_safety_status
            )
            
        # Log prediction to PostgreSQL in the background to avoid blocking critical path
        background_tasks.add_task(
            PredictionLogger.log_prediction,
            request.user_id,
            request.brand_context,
            response.bid_price,
            not response.should_bid
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_and_save_evidently_report(baseline: List[float], production: List[float]):
    """Helper background task to calculate and archive Evidently AI drift reports."""
    try:
        ref_df = pd.DataFrame({"feature_value": baseline})
        curr_df = pd.DataFrame({"feature_value": production})
        
        report_id = str(uuid.uuid4())[:8]
        reporter = EvidentlyDriftReporter(reports_dir="reports")
        summary = reporter.generate_report(ref_df, curr_df, report_id)
        
        # Save metrics to PostgreSQL
        await DriftReportStore.save_drift_report(summary, summary["dataset_drift_detected"])
    except Exception as e:
        logger.error(f"Failed to generate and log Evidently drift report in background: {e}")


@app.post("/check-drift", response_model=DriftCheckResponse)
async def verify_data_drift(request: DriftCheckRequest, background_tasks: BackgroundTasks):
    """Calculates KS-test statistical differences and PSI metric values."""
    try:
        monitor = PopulationDriftMonitor()
        ks_stat, p_val = monitor.kolmogorov_smirnov_test(
            request.baseline_features, request.production_features
        )
        psi_val = monitor.population_stability_index(
            request.baseline_features, request.production_features
        )
        
        drift_detected = p_val < 0.05 or psi_val >= 0.25
        status_msg = "RETRAIN_MODEL" if drift_detected else "STABLE"
        
        # Schedule the heavy Evidently AI report generation and database storage to run in background
        background_tasks.add_task(
            run_and_save_evidently_report,
            request.baseline_features,
            request.production_features
        )
        
        return DriftCheckResponse(
            ks_statistic=round(ks_stat, 4),
            p_value=float(p_val),
            psi_metric=round(psi_val, 4),
            drift_detected=drift_detected,
            status=status_msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain-and-deploy")
async def retrain_and_deploy():
    """
    Triggers model retraining, uploads artifacts to GCS, and registers with Vertex AI.
    Demonstrates GCP production pipeline patterns with local sandboxed fallbacks.
    """
    try:
        # 1. Simulate training a bid-shading model
        model_artifact = {"model_name": "bid_shade_model", "coef": 1.5, "intercept": 0.2}
        local_path = "model.pkl"
        with open(local_path, "wb") as f:
            pickle.dump(model_artifact, f)
            
        # 2. Upload to Google Cloud Storage (GCS)
        gcs_uri = "gs://ml-sandbox-models/model.pkl"
        gcs_uploaded = False
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket("ml-sandbox-models")
            blob = bucket.blob("model.pkl")
            blob.upload_from_filename(local_path)
            gcs_uploaded = True
            logger.info("Successfully uploaded model artifact to GCS.")
        except Exception as e:
            logger.warning(f"GCS upload fallback: running locally ({e}). Saved to local model.pkl.")
            
        # 3. Register model with Vertex AI Model Registry
        vertex_registered = False
        try:
            from google.cloud import aiplatform
            aiplatform.init(project="sightly-mlops", location="us-central1")
            _ = aiplatform.Model.upload(
                display_name="bid-shading-model",
                artifact_uri="gs://ml-sandbox-models/",
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
            )
            vertex_registered = True
            logger.info("Successfully registered model in Vertex AI Model Registry.")
        except Exception as e:
            logger.warning(f"Vertex AI registration fallback: running locally ({e}).")
            
        return {
            "status": "SUCCESS",
            "model_version": "1.0.0",
            "gcs_uploaded": gcs_uploaded,
            "gcs_uri": gcs_uri if gcs_uploaded else f"local://{local_path}",
            "vertex_registered": vertex_registered
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
