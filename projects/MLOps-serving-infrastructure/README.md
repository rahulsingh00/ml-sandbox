# 🚀 MLOps serving & monitoring infrastructure

This project contains Python implementations for model serving and production monitoring pipelines, featuring async FastAPI endpoints, schema validation via Pydantic, Docker container configurations, and data drift calculations (Kolmogorov-Smirnov test and PSI).

## 📁 File Structure

*   [`main.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/MLOps-serving-infrastructure/main.py): High-performance async FastAPI application serving inference endpoints.
*   [`schemas.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/MLOps-serving-infrastructure/schemas.py): Pydantic input/output schemas for contract-safe APIs.
*   [`drift_detector.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/MLOps-serving-infrastructure/drift_detector.py): Computes Kolmogorov-Smirnov statistical tests and Population Stability Index (PSI) values.
*   [`Dockerfile`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/MLOps-serving-infrastructure/Dockerfile): Packaging structure for deploying containers to GCP (Vertex AI or Cloud Run).

## ⚙️ Installation & Setup

1.  Navigate to this project directory:
    ```bash
    cd projects/MLOps-serving-infrastructure
    ```
2.  Install requirements:
    ```bash
    pip3 install -r requirements.txt
    ```

## 🚀 Execution & Verification

Run the drift detection test:
```bash
python3 drift_detector.py
```
This will compute the Kolmogorov-Smirnov p-value and the PSI metric for two simulated feature distributions (baseline vs. shifted production data), outputting whether drift alarms are triggered.
