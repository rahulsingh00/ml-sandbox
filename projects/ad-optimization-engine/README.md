# 📊 Ad Optimization Engine

This project contains Python implementations of ad-tech optimization systems, modeling bid price optimization, budget allocation under constraints, lookalike audience expansion, and dynamic creative selection using multi-armed bandits.

---

## 📁 File Structure

*   [`bid_optimizer.py`](bid_optimizer.py): Calculates the utility-maximizing bid price using probability of winning curves (log-logistic win rate modeling).
*   [`budget_allocator.py`](budget_allocator.py): Solves linear programming optimization problems using `scipy.optimize.linprog` to distribute advertising budgets across multiple platforms under ROI and spend constraints.
*   [`creative_bandit.py`](creative_bandit.py): Selects optimal ad creatives dynamically using a Thompson Sampling Multi-Armed Bandit model (Beta-Binomial conjugate distributions).
*   [`audience_targeter.py`](audience_targeter.py): Implements distance-based (KNN) and classification-based (Logistic Regression) lookalike audience modeling, including exclusion filters and frequency capping.
*   [`ctr_regression.py`](ctr_regression.py): Trains a `GradientBoostingRegressor` to predict continuous CTR values, including train/test splits, model evaluation metrics, and a Snowflake data retrieval connector stub with local fallback.

---

## ⚙️ Installation & Setup

1. Navigate to this project directory:
   ```bash
   cd projects/ad-optimization-engine
   ```
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 🚀 Execution & Verification

You can execute individual optimization modules with demo data:

### 1. Linear Programming Budget Allocator
```bash
python3 budget_allocator.py
```
Solves the optimal budget split across channels (Google, Meta, CTV) under predefined boundaries and outputs allocations.

### 2. Bid Price Utility Maximizer
```bash
python3 bid_optimizer.py
```
Computes and plots utility curves showing the optimal bid price given user value and win-rate curves.

### 3. Thompson Sampling Bandit
```bash
python3 creative_bandit.py
```
Simulates a multi-round dynamic creative selection game, showing the bandit learning CTRs and converging to the best creative.

---

## 🧪 Running Unit Tests

This project includes a test suite under `tests/` validating optimization algorithms, boundary conditions, and classifier bounds.

### Run from Workspace Root (Recommended)
```bash
PYTHONPATH=projects/ad-optimization-engine python3 -m pytest projects/ad-optimization-engine/tests/ -v
```

### Run from Project Directory
```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
