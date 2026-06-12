# 📊 Dataset Acquisition & Simulators Guide

This guide describes how datasets are structured and acquired inside the ML Sandbox, covering programmatic synthetic simulators, public real-world benchmark datasets, and cloud enterprise adapters.

---

## ⚙️ Programmatic Synthetic Simulators

For local testing and offline verification, the sandbox includes built-in synthetic data generation engines. These ensure the code executes instantly without requiring external network calls or heavy downloads, while preserving realistic statistical properties (like multi-dimensional confounding and log-logistic win rates).

### 1. CTR Regression Simulator
- **Location**: `projects/ad-optimization-engine/ctr_regression.py` (`generate_synthetic_data()`)
- **Properties**: Generates a DataFrame containing campaign metrics:
  - `hour_of_day`: `[0, 23]` (int)
  - `day_of_week`: `[0, 6]` (int)
  - `device_type`: `["Mobile", "Desktop", "Tablet", "Connected TV"]` (string)
  - `placement`: `["Feed", "Banner", "Video Pre-roll", "Interstitial"]` (string)
  - `historical_ctr`: Log-normal distribution centered around 2%
  - `bid_floor`: Uniform distribution representing dynamic pricing floor
  - `actual_ctr`: Synthetic CTR incorporating linear effects of hour, day, historical CTR, and random noise.
- **Run command**:
  ```bash
  python3 projects/ad-optimization-engine/ctr_regression.py
  ```

### 2. Causal Confounded Observational Simulator
- **Location**: `projects/causal-uplift-experimenter/double_ml_estimator.py` (main test script)
- **Properties**: Simulates a partially linear model representing selection bias:
  - Covariates $X_1$ (normal) and $X_2$ (uniform) represent user profiles.
  - Treatment assignment $T$ is confounded by $X_1$ and $X_2$: $P(T=1) = \text{sigmoid}(X_1 + 2X_2)$.
  - True treatment effect (ATE) is set to 1.8.
  - Outcome $Y$ is determined by $Y = ATE \cdot T + 2\sin(X_1) + 3X_2^2 + \text{noise}$.
- **Run command**:
  ```bash
  python3 projects/causal-uplift-experimenter/double_ml_estimator.py
  ```

---

## 🌍 Real-World Public Benchmarks

To train production models on real-world data, the following public datasets are recommended and can be loaded into the folders.

### 1. Named Entity Recognition (NER)
- **Dataset**: CoNLL-2003 Shared Task Dataset
- **Purpose**: Benchmark Named Entity Recognition models.
- **Acquisition**: Download via HuggingFace `datasets` package:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("conll2003")
  ```

### 2. Sentiment Analysis
- **Dataset**: Stanford Sentiment Treebank (SST-2) or Yelp Open Dataset
- **Purpose**: Benchmark sentiment polarity classifiers.
- **Acquisition**: SST-2 is available via GLUE benchmark:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("glue", "sst2")
  ```

### 3. Multi-modal CLIP (Image/Text)
- **Dataset**: MS COCO (Common Objects in Context)
- **Purpose**: Zero-shot image classification and semantic search queries using CLIP.
- **Acquisition**: Download from official COCO dataset page (http://cocodataset.org/#download) or via `fiftyone` package.

---

## ☁️ Enterprise Cloud Data Connections

When deploying to production, stubs inside the sandbox are designed to integrate directly with enterprise cloud lakes.

### 1. Snowflake Data Warehouse
- **Connection**: `snowflake-connector-python` with `pandas` integration.
- **Query Structure**: Implemented in `ctr_regression.py` (`load_training_data_from_snowflake`):
  ```sql
  SELECT HOUR_OF_DAY, DAY_OF_WEEK, DEVICE_TYPE, PLACEMENT, HISTORICAL_CTR, BID_FLOOR, ACTUAL_CTR
  FROM AD_PERFORMANCE_METRICS
  SAMPLE (10) -- Bernoulli sampling for data scaling
  ```
- **Configuration**: Connection parameters dict must contain user, password, account, warehouse, database, schema.

### 2. Google Cloud Storage (GCS)
- **Client**: `google-cloud-storage` SDK.
- **Usage**: Models are serialized as `.joblib` files and uploaded to/downloaded from GCS buckets:
  ```python
  from google.cloud import storage
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  blob = bucket.blob("models/ctr_model.joblib")
  blob.upload_from_filename("ctr_model.joblib")
  ```
