"""
Unit tests for the MLOps-serving-infrastructure project.
Tests FastAPI endpoints, database loggers, KS/PSI monitors, and Evidently reports.
"""

import os
import shutil
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from drift_detector import PopulationDriftMonitor
from database import PredictionLogger, DriftReportStore, mock_prediction_logs, mock_drift_reports
from drift_reporter import EvidentlyDriftReporter

client = TestClient(app)


# ----------------------------------------------------
# 1. FastAPI Endpoint Tests
# ----------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "HEALTHY", "version": "1.0.0"}


def test_bid_endpoint_safe():
    payload = {
        "user_id": "usr_99",
        "brand_context": "Family friendly board games and toys",
        "estimated_ctr": 0.045,
        "campaign_id": "cmp_1",
        "user_features": [0.5, -0.2, 0.1]
    }
    response = client.post("/bid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["should_bid"]
    assert data["bid_price"] > 0.0
    assert data["brand_safety_status"] == "SAFE"
    assert data["targeted_creative_id"] == "crt_101" # features[0] > 0


def test_bid_endpoint_unsafe():
    payload = {
        "user_id": "usr_99",
        "brand_context": "Illegal hacking and weapons sale report",
        "estimated_ctr": 0.045,
        "campaign_id": "cmp_1",
        "user_features": [-0.5, -0.2, 0.1]
    }
    response = client.post("/bid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert not data["should_bid"]
    assert data["bid_price"] == 0.0
    assert data["brand_safety_status"] == "UNSAFE"


def test_check_drift_endpoint():
    payload = {
        "baseline_features": [1.0, 1.2, 1.3, 1.1, 1.2],
        "production_features": [1.0, 1.2, 1.3, 1.1, 1.2]
    }
    response = client.post("/check-drift", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert not data["drift_detected"]
    assert data["status"] == "STABLE"


def test_retrain_and_deploy_endpoint():
    response = client.post("/retrain-and-deploy")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "model_version" in data
    
    # Cleanup model file
    if os.path.exists("model.pkl"):
        try:
            os.remove("model.pkl")
        except OSError:
            pass


# ----------------------------------------------------
# 2. Custom Drift Detector Tests
# ----------------------------------------------------

def test_population_drift_monitor():
    monitor = PopulationDriftMonitor()
    
    # Identical distributions
    base = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    prod_stable = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
    
    ks, p_val = monitor.kolmogorov_smirnov_test(base, prod_stable)
    assert ks == 0.0
    assert p_val == 1.0
    
    psi = monitor.population_stability_index(base, prod_stable)
    assert psi == 0.0
    
    # Drifted distribution
    prod_drift = [0.6, 0.7, 0.8, 0.9, 1.0] * 20
    ks_dr, p_val_dr = monitor.kolmogorov_smirnov_test(base, prod_drift)
    assert ks_dr > 0.5
    assert p_val_dr < 0.05
    
    psi_dr = monitor.population_stability_index(base, prod_drift)
    assert psi_dr >= 0.25


# ----------------------------------------------------
# 3. Database Fallback Tests
# ----------------------------------------------------

def test_database_logging_fallback():
    import asyncio
    
    # Clear mocks
    mock_prediction_logs.clear()
    mock_drift_reports.clear()
    
    # Run async loggers synchronously using asyncio.run
    asyncio.run(PredictionLogger.log_prediction(
        user_id="usr_test_1",
        brand_context="Test brand context",
        predicted_bid=1.55,
        is_blocked=False
    ))
    
    asyncio.run(DriftReportStore.save_drift_report(
        metrics={"metric_drift": 0.33},
        dataset_drift_detected=True
    ))
    
    # Assert fallbacks populated
    assert len(mock_prediction_logs) == 1
    assert mock_prediction_logs[0]["user_id"] == "usr_test_1"
    assert mock_prediction_logs[0]["predicted_bid"] == 1.55
    
    assert len(mock_drift_reports) == 1
    assert mock_drift_reports[0]["dataset_drift_detected"]
    assert mock_drift_reports[0]["metrics"]["metric_drift"] == 0.33


# ----------------------------------------------------
# 4. Evidently AI Report Generation Tests
# ----------------------------------------------------

def test_evidently_drift_reporter():
    rng = np.random.default_rng(42)
    # Use larger dataset size so KS-test finds statistically significant drift
    ref = pd.DataFrame({"feat_1": rng.normal(loc=0.0, scale=1.0, size=100)})
    curr = pd.DataFrame({"feat_1": rng.normal(loc=2.0, scale=1.0, size=100)})
    
    reporter = EvidentlyDriftReporter(reports_dir="test_reports")
    summary = reporter.generate_report(ref, curr, report_id="test_run")
    
    assert summary["report_id"] == "test_run"
    assert os.path.exists(summary["html_path"])
    assert summary["dataset_drift_detected"]
    assert summary["number_of_drifted_features"] == 1
    
    # Cleanup
    if os.path.exists("test_reports"):
        shutil.rmtree("test_reports")

