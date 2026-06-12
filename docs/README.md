# 📚 ML Sandbox — Documentation Index

> Living documentation for the multi-project machine learning, optimization, and causal inference sandbox designed for the ML Engineering Lead role.

---

## 📐 Architecture & System Reference

| Document | Description |
|:---|:---|
| **[System Architecture Overview](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/architecture/overview.md)** | End-to-end design, data pipelines, sub-project boundaries, and data flows. |
| **[Source Tree Map](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/architecture/source-tree.md)** | Annotated workspace structure mapping files to specific job requirements. |
| **[Dataset & Simulators Guide](datasets.md)** | How to generate mock data or download real-world public datasets for testing. |

---

## 📊 Gap Analysis & Progress Tracking

| Document | Description |
|:---|:---|
| **[Gap Analysis](gap-analysis.md)** | Detailed skills gap analysis against the [Sightly ML Engineering Lead](https://app.trinethire.com/companies/22914-sightly-enterprises-inc/jobs/121118-machine-learning-engineering-lead) job posting. Tracks what's implemented, what's partial/mocked, and what's missing — updated as implementations progress. |
| **[Agile Project Backlog](../backlog.md)** | Epics, completed user stories, and future roadmap enhancements. |

---

## 🛠️ Build & Quality Pipelines

| Document | Description |
|:---|:---|
| **[CI/CD & Local Verification](ci-cd.md)** | Details of GitHub Actions workflows, manual triggers, and running verification suites locally. |

---

## 🧠 Core Competency Guides

Each guide provides a detailed conceptual walkthrough, mathematical formulation (where applicable), and links to practical implementations inside the sandbox.

| Concept | Topics Covered | Related Sub-Project |
|:---|:---|:---|
| **[01. Cultural Data & NLP](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/concepts/01-cultural-data-nlp.md)** | NER, Topic modeling, Stance & Sentiment detection, Clustering, Brand Safety | `cultural-enrichment-pipeline` |
| **[02. Multi-modal Enrichment](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/concepts/02-multimodal-enrichment.md)** | Image & Video signals, CLIP embeddings, temporal pooling, visual brand safety | `cultural-enrichment-pipeline` |
| **[03. Ad Optimization Systems](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/concepts/03-ad-optimization.md)** | Linear Programming, bidding policies, budget constraint equations, utility functions | `ad-optimization-engine` |
| **[04. Causal Inference & A/B Testing](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/concepts/04-causal-uplift-ab.md)** | Hypothesis framing, power analysis, A/B/n testing, confounders, S/T/X-Learners | `causal-uplift-experimenter` |
| **[05. Production MLOps Engineering](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/docs/concepts/05-production-mlops.md)** | FastAPI async, dockerization, database schemas, drift metrics (KS-Test, PSI) | `MLOps-serving-infrastructure` |

---

## 🏗️ Documentation & Coding Standards

1. **Self-Documenting Implementations**: Every Python script in the sub-projects must contain clear docstrings explaining the underlying algorithm and its complexity.
2. **Reproducibility**: Each project must contain a `requirements.txt` or setup script to ensure quick environment setup.
3. **Data-Centric Design**: When mock data is generated for notebooks or tests, it must be statistically grounded (e.g. simulating true uplift or ad performance distributions) rather than pure random noise.
