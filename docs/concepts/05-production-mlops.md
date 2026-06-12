# 🧠 Concept: Production MLOps & Monitoring

This guide covers serving ML models at scale, design patterns for low-latency web services, containerization, and post-deployment monitoring for data drift.

---

## 1. High-Performance Model Serving with FastAPI
When serving models in production, the service must handle concurrent requests without blocking. FastAPI utilizes Python's `async`/`await` primitives and ASGI server (Uvicorn) to execute code concurrently.

*   **Concurrency vs. Parallelism**:
    *   *Async IO*: Frees up the event loop while waiting for database queries or downstream API requests to complete.
    *   *CPU-bound Inference*: Model forward passes (e.g. PyTorch/TensorFlow predictions) are CPU-bound and block the Python interpreter due to the Global Interpreter Lock (GIL). 
*   **Production Serving Architecture**:
    *   Perform heavy operations inside a process pool (e.g., using `concurrent.futures.ProcessPoolExecutor`) or utilize model serving frameworks like TorchServe or Triton.
    *   Use Pydantic for strict input/output payload validation.

---

## 2. Drift Detection & Model Monitoring
Once deployed, model performance degrades over time due to changes in user behavior or external context (e.g., cultural shifts). We monitor two primary types of drift:

### A. Data Drift (Covariate Shift)
The distribution of input features changes over time:
$$P(X_{\text{production}}) \neq P(X_{\text{baseline}})$$

#### Kolmogorov-Smirnov (KS) Test
A non-parametric statistical test used to determine if two continuous, one-dimensional distributions differ significantly.
*   **Null Hypothesis ($H_0$)**: The production feature sample is drawn from the same distribution as the baseline.
*   **Statistic $D$**: Represents the maximum vertical distance between the cumulative empirical distribution functions of the two samples.
    $$D = \sup_{x} |F_1(x) - F_2(x)|$$
*   If the p-value is less than a significance level (e.g. 0.05), we reject the null hypothesis and signal that data drift has occurred.

#### Population Stability Index (PSI)
Used to measure shifts in categorical variables or continuous variable buckets:
$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left(\frac{P_b}{Q_b}\right)$$
where $P_b$ is the proportion of records in bucket $b$ of the baseline population, and $Q_b$ is the proportion in the target population.
*   $\text{PSI} < 0.1$: No significant shift.
*   $0.1 \le \text{PSI} < 0.25$: Moderate shift; model should be checked.
*   $\text{PSI} \ge 0.25$: Significant shift; trigger model retraining.

### B. Concept Drift
The relationship between input features and target variables changes:
$$P(Y \mid X_{\text{production}}) \neq P(Y \mid X_{\text{baseline}})$$
This requires logging actual outcomes (e.g. click feedback) and comparing model performance metrics (AUC, F1-score) over time.

---

## 3. Data Integration (PostgreSQL & Snowflake)
*   **OLTP (PostgreSQL)**: Handles real-time transaction updates, serving user profile embeddings, and saving live logs. Needs low-latency indexes (e.g. `pgvector` for embedding searches).
*   **OLAP (Snowflake)**: Handles large-scale analytical queries, computing historical ad-performance aggregations, and preparing datasets for training pipelines.

---

## 🛠️ Sandbox Implementation
Check out the FastAPI serving and drift monitoring implementations in [`projects/MLOps-serving-infrastructure`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/MLOps-serving-infrastructure).
