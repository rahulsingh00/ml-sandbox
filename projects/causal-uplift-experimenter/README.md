# 🧪 Causal Inference & Experimentation Sandbox

This project contains Python implementations of experimentation tools and causal models, featuring statistical power calculations, A/B testing statistical checks, and Uplift model metalearners (T-Learner and S-Learner) for incrementality analysis.

## 📁 File Structure

*   [`power_calculator.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/causal-uplift-experimenter/power_calculator.py): Determines sample size and Minimum Detectable Effect (MDE) bounds using statsmodels.
*   [`stat_tests.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/causal-uplift-experimenter/stat_tests.py): Executes t-tests, chi-square tests, and computes p-values for A/B testing groups.
*   [`uplift_model.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/causal-uplift-experimenter/uplift_model.py): Implements S-Learner and T-Learner models using scikit-learn base learners to predict individual uplift.

## ⚙️ Installation & Setup

1.  Navigate to this project directory:
    ```bash
    cd projects/causal-uplift-experimenter
    ```
2.  Install requirements:
    ```bash
    pip3 install -r requirements.txt
    ```

## 🚀 Execution & Verification

Run the statistical test suite:
```bash
python3 stat_tests.py
```
This will run a simulated experiment, check if the treatment conversion rate shows statistical significance, and output p-values and confidence intervals.
