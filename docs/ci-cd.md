# 🚀 CI/CD & Local Pipeline Verification

This document outlines the coding standards, automated checks, and testing pipelines configured for the ML Sandbox.

---

## 🛠️ GitHub Actions Configuration

To avoid unwanted runner costs on GitHub, automatic CI triggers on commits and pull requests have been **disabled** in the workflow file [.github/workflows/ci.yml](file:///Users/rahulsingh/Work/ml-sandbox/.github/workflows/ci.yml).

The workflow is configured with `workflow_dispatch:`, which means it can still be run manually in the Actions tab of the GitHub repository interface, but it will never run automatically or incur automatic costs.

---

## 💻 Running Pipeline Verification Locally

A helper script is provided at the root of the repository to execute the exact same verification suite (linter, type checker, and all unit tests) locally in one command.

### Usage

Run the script from the root of the workspace:

```bash
./run_ci_locally.sh
```

### What the Script Checks

1. **Ruff Linter**: Performs style checks and code formatting verification across all sub-projects.
2. **Mypy Type Checker**: Runs strict static type checking under PEP 484 guidelines.
3. **Automated Unit Tests**: Executes the full test suite for each sub-project:
   - **Cultural Enrichment Pipeline**: Zero-shot NLP classifiers, stance/sentiment detectors, CLIP embeddings, and clustering tests.
   - **Ad Optimization Engine**: Bandit algorithms, bid optimizers, budget allocators, CTR regressors, and lookalike audience targeters.
   - **Causal Uplift Experimenter**: Propensity Score Matching (PSM), Double Machine Learning (DML) cross-fitting, A/B sizers, and hypothesis tests.
   - **MLOps Serving Infrastructure**: FastAPI endpoints, drift checking service, evidently reports, database logging, and retry logic.

---

## 🔍 Individual Validation Commands

If you wish to run specific parts of the pipeline manually:

### 1. Style & Lint Check (Ruff)
```bash
python3 -m ruff check .
```

### 2. Type Checking (Mypy)
```bash
python3 -m mypy --ignore-missing-imports projects/
```

### 3. Sub-Project Unit Tests
- **Cultural Enrichment Pipeline**:
  ```bash
  PYTHONPATH=projects/cultural-enrichment-pipeline python3 -m pytest projects/cultural-enrichment-pipeline/tests/ -v
  ```
- **Ad Optimization Engine**:
  ```bash
  PYTHONPATH=projects/ad-optimization-engine python3 -m pytest projects/ad-optimization-engine/tests/ -v
  ```
- **Causal Uplift Experimenter**:
  ```bash
  PYTHONPATH=projects/causal-uplift-experimenter python3 -m pytest projects/causal-uplift-experimenter/tests/ -v
  ```
- **MLOps Serving Infrastructure**:
  ```bash
  PYTHONPATH=projects/MLOps-serving-infrastructure python3 -m pytest projects/MLOps-serving-infrastructure/tests/ -v
  ```
