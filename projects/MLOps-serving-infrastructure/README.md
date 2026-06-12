# 🚀 MLOps Serving & Monitoring Infrastructure

This project contains Python implementations for model serving and production monitoring pipelines, featuring async FastAPI endpoints, schema validation via Pydantic V2, PostgreSQL logging, background drift reporter tasks, and container configurations for cloud deployment.

---

## 📁 File Structure

*   [`main.py`](main.py): High-performance async FastAPI application serving inference endpoints (`/bid`, `/check-drift`, `/health`, `/retrain-and-deploy`) using non-blocking asynchronous call handlers and background execution tasks.
*   [`schemas.py`](schemas.py): Pydantic input/output schema definitions validating contracts for bid requests, responses, and drift check requests.
*   [`database.py`](database.py): Asynchronous database connector managing a pool of connections (via `asyncpg`) to write inference logs and drift reports to PostgreSQL with thread-safe memory fallbacks.
*   [`drift_detector.py`](drift_detector.py): Calculates Kolmogorov-Smirnov statistical test p-values and Population Stability Index (PSI) metrics between baseline and production feature arrays.
*   [`drift_reporter.py`](drift_reporter.py): Leverages Evidently AI's data quality packages to analyze feature shifts and output static HTML drift dashboards.
*   [`Dockerfile`](Dockerfile): Hardened multi-stage Docker build utilizing non-root users, virtual environments, and automated `HEALTHCHECK` instructions.
*   [`docker-compose.yml`](docker-compose.yml): Setups a complete local environment including the FastAPI app server, a PostgreSQL database, and automated network mapping.
*   [`.dockerignore`](.dockerignore): Excludes unnecessary source directories to optimize container builds.

---

## ⚙️ Installation & Setup

1. Navigate to this project directory:
   ```bash
   cd projects/MLOps-serving-infrastructure
   ```
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 🚀 Execution & Verification

### 1. Run Drift Detector locally
```bash
python3 drift_detector.py
```
Computes KS-test stats and PSI for shifted numeric feature vectors, logging if alert limits are breached.

### 2. Start FastAPI Server locally
```bash
uvicorn main:app --reload --port 8000
```
This runs the API locally. You can send requests using `curl` or check the swagger docs at `http://127.0.0.1:8000/docs`.

### 3. Query the FastAPI Server
- **Health Check**:
  ```bash
  curl -X GET http://127.0.0.1:8000/health
  ```
- **Bid Optimization Endpoint**:
  ```bash
  curl -X POST http://127.0.0.1:8000/bid \
       -H "Content-Type: application/json" \
       -d '{"user_id":"usr_432","brand_context":"Nike shoes campaign","estimated_ctr":0.045,"campaign_id":"cmp_98","user_features":[0.12,-0.4,0.9]}'
  ```

### 4. Running the Complete Infrastructure via Docker Compose
Build and run the FastAPI server and PostgreSQL DB locally:
```bash
docker-compose up --build
```
This boots both the API and database container. Run `docker-compose down` to spin it down.

---

## 🧪 Running Unit Tests

This project includes a test suite under `tests/` validating FastAPI responses, async database connection hooks, invalid payloads, and retraining triggers.

### Run from Workspace Root (Recommended)
```bash
PYTHONPATH=projects/MLOps-serving-infrastructure python3 -m pytest projects/MLOps-serving-infrastructure/tests/ -v
```

### Run from Project Directory
```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
