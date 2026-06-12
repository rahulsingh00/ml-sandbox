# 🧪 Causal Inference & Experimentation Sandbox

This project contains Python implementations of experimentation tools and causal models, featuring statistical power calculations, A/B testing statistical checks, propensity score matching (PSM), and Double Machine Learning (DML) for debiasing confounders.

---

## 📁 File Structure

*   [`power_calculator.py`](power_calculator.py): Determines sample size requirements, statistical power, and Minimum Detectable Effect (MDE) bounds using `statsmodels` power analysis objects.
*   [`stat_tests.py`](stat_tests.py): Executes statistical tests (two-sample z-test for proportions, t-test for continuous values, chi-square tests) and computes p-values and confidence intervals.
*   [`propensity_matcher.py`](propensity_matcher.py): Matches treated and control units using logistic regression propensity scores, greedy nearest-neighbor matching within a caliper, and evaluates balance diagnostics using Standardized Mean Difference (SMD).
*   [`double_ml_estimator.py`](double_ml_estimator.py): Implements Double Machine Learning (DML) with K-fold cross-fitting and Neyman-orthogonal score adjustments to isolate unbiased treatment effects (ATE) in confounded datasets.
*   [`uplift_model.py`](uplift_model.py): Implements meta-learners (S-Learner and T-Learner) using base scikit-learn regressors to estimate Individual Treatment Effects (ITE) and Conditional Average Treatment Effects (CATE) for uplift targeting.

---

## ⚙️ Installation & Setup

1. Navigate to this project directory:
   ```bash
   cd projects/causal-uplift-experimenter
   ```
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

---

## 🚀 Execution & Verification

You can execute individual causal and experimentation scripts:

### 1. Statistical Hypothesis Tests (A/B Test Evaluation)
```bash
python3 stat_tests.py
```
Simulates an A/B test run, computes p-values, conversion ratios, confidence intervals, and determines whether statistical significance has been reached.

### 2. Sample Size & Power Calculations
```bash
python3 power_calculator.py
```
Determines the required sample size given a target power, significance level, baseline conversion rate, and expected MDE.

### 3. Propensity Score Matcher (PSM)
```bash
python3 propensity_matcher.py
```
Simulates a confounded observational dataset, matches treated and control records, prints SMD covariate balance statistics before/after matching, and estimates the Average Treatment Effect on the Treated (ATT).

### 4. Double Machine Learning (DML)
```bash
python3 double_ml_estimator.py
```
Generates a confounded dataset, runs the cross-fit DML estimator using Gradient Boosting nuisance models, and outputs the debiased treatment effect estimate along with standard errors and confidence intervals.

### 5. Uplift Modeling (S/T-Learners)
```bash
python3 uplift_model.py
```
Trains S-Learner and T-Learner models, predicts uplift scores on simulated test sets, and outputs CATE distributions.

---

## 🧪 Running Unit Tests

This project includes a test suite under `tests/` validating power calculations, p-value formulas, PSM balance metrics, and DML convergence.

### Run from Workspace Root (Recommended)
```bash
PYTHONPATH=projects/causal-uplift-experimenter python3 -m pytest projects/causal-uplift-experimenter/tests/ -v
```

### Run from Project Directory
```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```
