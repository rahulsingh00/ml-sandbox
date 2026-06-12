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

## 📚 Study Guide & Concept Roadmap

To systematically master the theoretical and practical concepts, refer to our curated documentation guides in [`docs/`](file:///Users/rahulsingh/Work/ml-sandbox/docs):

1. **[Cultural Data & NLP](file:///Users/rahulsingh/Work/ml-sandbox/docs/concepts/01-cultural-data-nlp.md)**: Stance detection, NER, embedding spaces, clustering, and brand safety.
2. **[Multi-modal Enrichment](file:///Users/rahulsingh/Work/ml-sandbox/docs/concepts/02-multimodal-enrichment.md)**: Visual/Video signals, CLIP, frame embeddings, and multimodal classifiers.
3. **[Ad Optimization Systems](file:///Users/rahulsingh/Work/ml-sandbox/docs/concepts/03-ad-optimization.md)**: Mathematical formulations, constraints, objective functions, and bidding policies.
4. **[Causal Inference & Experimentation](file:///Users/rahulsingh/Work/ml-sandbox/docs/concepts/04-causal-uplift-ab.md)**: Power analysis, confounding variables, propensity score matching, and Uplift models (Metalearners).
5. **[Production MLOps Engineering](file:///Users/rahulsingh/Work/ml-sandbox/docs/concepts/05-production-mlops.md)**: FastAPI async patterns, data/concept drift detection (KS-test, PSI), PostgreSQL schemas, and Dockerizing workloads.

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

Ensure you have Python 3.9+ installed. To review the Sightly ML Engineering Lead gap analysis, see the global **[Gap Analysis Doc](file:///Users/rahulsingh/Work/ml-sandbox/docs/gap-analysis.md)**.

