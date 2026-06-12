# 🎯 ML Lead Mastery Sandbox

A comprehensive learning workspace and multi-project sandbox designed to build, demonstrate, and master the advanced machine learning, optimization, and MLOps paradigms specified in the ML Lead requirements.

---

## 🏛️ Workspace Architecture & Sub-Projects

The repository contains four distinct sub-projects, each modeling a core business capability of the Brand Mentality® and ad optimization pipeline:

| Sub-Project | Capability Modeled | Core Technologies |
|:---|:---|:---|
| **[`cultural-enrichment-pipeline`](file:///Users/rahulsingh/Work/ml-sandbox/projects/cultural-enrichment-pipeline)** | Cultural data pipelines (NER, Stance, Topic Classification, CLIP Multi-modal, Text Embeddings) | HuggingFace, PyTorch, CLIP, sentence-transformers, Scikit-Learn |
| **[`ad-optimization-engine`](file:///Users/rahulsingh/Work/ml-sandbox/projects/ad-optimization-engine)** | Bid Optimization, Budget Allocation, Lookalike Audience Targeting, CTR Regression | SciPy Optimize, Scikit-Learn, Pandas, NumPy, Snowflake data connector |
| **[`causal-uplift-experimenter`](file:///Users/rahulsingh/Work/ml-sandbox/projects/causal-uplift-experimenter)** | A/B Testing Instrumentation, Propensity Score Matching (PSM), Double Machine Learning (DML), Uplift Modeling | Statsmodels, Scikit-Learn, SciPy, Pandas, NumPy, Seaborn |
| **[`MLOps-serving-infrastructure`](file:///Users/rahulsingh/Work/ml-sandbox/projects/MLOps-serving-infrastructure)** | High-throughput Async Serving, Evidently AI Drift Presets, PostgreSQL logging, Docker Compose | FastAPI, Docker, Evidently AI, PostgreSQL (asyncpg), Vertex AI, GCS |

---

## 📖 Documentation & Project Navigation Map

Use the index below to navigate all documentation guides, system reference manuals, and individual sub-project READMEs across the workspace:

### 📐 System Reference & Architecture
* 🧭 **[Docs Index README](docs/README.md)** — Core entry point for all study guides.
* 📊 **[JD Gap Analysis](docs/gap-analysis.md)** — Breakdown of skills gap scorecard and implemented modules.
* 📋 **[Agile Project Backlog](backlog.md)** — Epics, completed user stories, and future roadmap enhancements.
* 🎨 **[System Architecture Overview](docs/architecture/overview.md)** — End-to-end system design, boundaries, and flows.
* 🌳 **[Source Tree Map](docs/architecture/source-tree.md)** — Detailed map of files and folders mapping to JD skills.
* 🚀 **[CI/CD & Local Verification Guide](docs/ci-cd.md)** — Guide to GitHub Actions settings and local CI verification script.

### 🧠 Core Competency Study Guides
* 📝 **[01. Cultural Data & NLP](docs/concepts/01-cultural-data-nlp.md)** — NER, stance, topic taxonomies, brand safety.
* 🖼️ **[02. Multi-modal Enrichment](docs/concepts/02-multimodal-enrichment.md)** — CLIP image/video signal extraction, pooling.
* 📈 **[03. Ad Optimization Systems](docs/concepts/03-ad-optimization.md)** — LP solvers, constraints, utility bidding policies.
* 🔍 **[04. Causal Inference & A/B Testing](docs/concepts/04-causal-uplift-ab.md)** — Confounder controls, matching, A/B sizers, DML.
* 🐳 **[05. Production MLOps Engineering](docs/concepts/05-production-mlops.md)** — FastAPI concurrency, drift detection (KS-test, PSI).

### 🏗️ Sub-Project Readmes
* 📡 **[Cultural Enrichment Pipeline Readme](projects/cultural-enrichment-pipeline/README.md)**
* 💰 **[Ad Optimization Engine Readme](projects/ad-optimization-engine/README.md)**
* 🧪 **[Causal Uplift Experimenter Readme](projects/causal-uplift-experimenter/README.md)**
* 🎛️ **[MLOps Serving Infrastructure Readme](projects/MLOps-serving-infrastructure/README.md)**

---

## 🧪 Running Test Suites

Each sub-project includes a comprehensive `pytest` test suite covering its core mathematical, statistical, and serving layers. You can run all test suites locally with:

```bash
# Run Cultural Enrichment Pipeline tests
PYTHONPATH=projects/cultural-enrichment-pipeline python3 -m pytest projects/cultural-enrichment-pipeline/tests/ -v

# Run Ad Optimization Engine tests
PYTHONPATH=projects/ad-optimization-engine python3 -m pytest projects/ad-optimization-engine/tests/ -v

# Run Causal Uplift Experimenter tests
PYTHONPATH=projects/causal-uplift-experimenter python3 -m pytest projects/causal-uplift-experimenter/tests/ -v

# Run MLOps Serving Infrastructure tests
PYTHONPATH=projects/MLOps-serving-infrastructure python3 -m pytest projects/MLOps-serving-infrastructure/tests/ -v
```

---

## 🚀 Getting Started

1. **Install All Workspace Dependencies**:
   Install all combined back-end and front-end dashboard dependencies in one command:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Launch the Interactive Streamlit Dashboard**:
   Launch the interactive, multi-page web application to showcase the pipeline features:
   ```bash
   streamlit run dashboard/app.py
   ```
   This opens a web dashboard (typically at `http://localhost:8501`) presenting text enrichment models, CLIP image/video safety evaluators, budget LP optimization charts, and data drift dashboards.

3. **Run Local CI Verification**:
   Execute the linter, type-checker, and 31 unit tests in one command:
   ```bash
   ./run_ci_locally.sh
   ```

To review the Sightly ML Engineering Lead gap analysis, see the global **[Gap Analysis Doc](docs/gap-analysis.md)** and the **[Agile Project Backlog](backlog.md)**.

