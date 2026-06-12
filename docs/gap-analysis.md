# 🎯 ML Engineering Lead — Gap Analysis

**Job Posting**: [Sightly Enterprises — ML Engineering Lead](https://app.trinethire.com/companies/22914-sightly-enterprises-inc/jobs/121118-machine-learning-engineering-lead)
**Last Updated**: June 12, 2026

---

## Executive Summary

The ml-sandbox has a **solid architectural blueprint** — all four sub-projects map well to the JD's "What you'll work on" areas, and the docs/concept guides provide good theoretical coverage. However, a deep code inspection reveals that **many implementations are lightweight prototypes** with heuristic fallbacks, mock data, and placeholder imports rather than real model inference. The project currently demonstrates **~55-60%** of the required technical skills with working code.

> [!IMPORTANT]
> The README and docs describe a more complete system than what the actual code delivers. Several key libraries (transformers, torch, sentence-transformers, Pillow, pandas) are declared as dependencies but **never actually called** in the code.

---

## Coverage Scorecard

### Legend
- 🟢 **Solid** — Working code with real logic
- 🟡 **Partial** — Placeholder/heuristic/stub, or mentioned in docs only
- 🔴 **Missing** — Not present in code or docs

---

### Cultural Enrichment Pipeline (`projects/cultural-enrichment-pipeline`)

> [!WARNING]
> This project has only **3 Python files (~224 lines total)**. Most HuggingFace models are imported but never instantiated. The pipeline relies on rule-based heuristics and mock data.

| JD Requirement | Actual State | Rating | Details |
|:---|:---|:---:|:---|
| NER / Entity Extraction | Regex-based brand detection (hardcoded list) | 🟡 | No HuggingFace NER model (`dslim/bert-base-NER` not used). Just regex + hashtag extraction |
| Stance Classification | Rule-based keyword scanning | 🟡 | Scans for "boycott/cancel/hate" vs "love/recommend/support". No NLI model |
| Topic Classification | Placeholder only | 🔴 | `zero_shot_classifier = None` — no actual implementation |
| Sentiment Analysis | Heuristic word-list counting | 🟡 | Counts positive/negative keywords. HuggingFace pipeline imported but **never called** |
| Embeddings | Not implemented | 🔴 | `sentence-transformers` in requirements but never imported in code |
| Clustering | TF-IDF + KMeans | 🟢 | Real scikit-learn implementation. Only genuine ML in this project |
| Brand Safety | Keyword blacklist scanning | 🟡 | Simple unsafe-word list. No `toxic-bert` or model-based scoring |
| Multi-modal (CLIP) | **Fully mocked** | 🔴 | 512-dim random vectors seeded by filename length. No actual CLIP model loaded, no real images processed |
| Video Enrichment | Not present | 🔴 | No video processing at all |

---

### Ad Optimization Engine (`projects/ad-optimization-engine`)

> [!NOTE]
> This project is the **strongest of the four** — all three modules contain real mathematical optimization logic, though it's still a prototype (~231 lines across 3 files).

| JD Requirement | Actual State | Rating | Details |
|:---|:---|:---:|:---|
| Bid Optimization | Sigmoid win-probability + bounded scalar minimization | 🟢 | Real `scipy.optimize.minimize_scalar`. Models expected utility U(bid) = (value - bid) × P_win(bid) |
| Budget Allocation | LP via `scipy.optimize.linprog` (HiGHS) | 🟢 | Real LP with budget constraints, diversity limits, minimum spend floors |
| Creative Selection | Thompson Sampling MAB | 🟢 | Real Beta-Bernoulli bandits with proper Bayesian updating |
| Audience Targeting | Not implemented | 🔴 | No module exists despite being mentioned in docs |
| Objective Functions & Constraints | Well-articulated | 🟢 | Each module clearly defines objective, constraints, and tradeoffs |

---

### Causal Uplift Experimenter (`projects/causal-uplift-experimenter`)

> [!NOTE]
> Covers experimentation basics but is missing the advanced causal inference methods described in the README/docs (PSM, DoubleML, IPW, X-Learner).

| JD Requirement | Actual State | Rating | Details |
|:---|:---|:---:|:---|
| A/B Testing | Z-test for proportions | 🟢 | Manual implementation using `scipy.stats.norm.cdf`. Two-sided test |
| Power Analysis | Sample size & power calculation | 🟢 | Real `statsmodels.TTestIndPower`. Cohen's d effect sizes |
| Sequential Testing | Not implemented | 🔴 | No O'Brien-Fleming or Pocock alpha-spending |
| Propensity Score Matching | Not implemented | 🔴 | Not present despite being in docs |
| Double ML | Not implemented | 🔴 | Not present despite being in docs |
| Meta-learners (Uplift) | S-Learner + T-Learner | 🟢 | Both implemented with LogisticRegression base learner. Synthetic heterogeneous treatment effects |
| IPW / Doubly Robust | Not implemented | 🔴 | Not present |
| Multiple Comparison Correction | Not implemented | 🔴 | No Bonferroni, FDR, etc. |
| Bootstrap CIs | Not implemented | 🔴 | Mentioned in README but not coded |

---

### MLOps Serving Infrastructure (`projects/MLOps-serving-infrastructure`)

> [!WARNING]
> The FastAPI + drift detection core is functional, but PostgreSQL, Evidently AI, CI/CD, model registry, and automated retraining are all absent.

| JD Requirement | Actual State | Rating | Details |
|:---|:---|:---:|:---|
| FastAPI + async/await | 3 async endpoints (`/health`, `/bid`, `/check-drift`) | 🟢 | Real async handlers, Pydantic validation, error handling |
| Pydantic Schemas | Request/response models | 🟢 | `BidRequest`, `BidResponse`, `DriftCheckRequest`, `DriftCheckResponse` |
| Data Drift (KS-test) | Real implementation | 🟢 | Wraps `scipy.stats.ks_2samp`, with demo scenarios |
| Data Drift (PSI) | Real implementation | 🟢 | Custom decile-binned PSI with epsilon smoothing |
| Evidently AI | Not present | 🔴 | Drift detection is custom scipy/numpy only |
| Docker | Basic Dockerfile | 🟡 | Functional but no multi-stage build, no `.dockerignore`, no non-root user, no health check |
| PostgreSQL | Not present | 🔴 | No database code, no SQLAlchemy/psycopg2, no connection pooling |
| Model Registry / Versioning | Not present | 🔴 | Endpoint uses inline heuristic, no real model loading |
| CI/CD | Not present | 🔴 | No GitHub Actions, no pipeline config of any kind |
| Model Monitoring | Drift check endpoint only | 🟡 | Returns "RETRAIN_MODEL" status but no actual trigger/scheduler |
| Automated Retraining | Not present | 🔴 | No pipeline, scheduler, or Pub/Sub integration |
| GCP Cloud Run Deployment | Not present | 🔴 | README mentions it but no deployment config |

---

### Cross-Cutting Skills

| JD Requirement | Actual State | Rating | Details |
|:---|:---|:---:|:---|
| **Python** | All code is Python | 🟢 | Clean, well-structured |
| **scikit-learn** | Used in 2 projects | 🟢 | KMeans, TF-IDF, LogisticRegression |
| **PyTorch** | Listed as dependency, **never used** | 🔴 | `torch` appears in requirements.txt but is never imported |
| **TensorFlow** | Not present | 🔴 | Not even in requirements |
| **HuggingFace Transformers** | Imported but **never called** | 🔴 | `transformers.pipeline` is imported conditionally but no model is ever instantiated |
| **NumPy** | Actively used | 🟢 | Used in all projects |
| **pandas** | In requirements but **never imported** | 🔴 | Unused dependency in 3 of 4 projects |
| **SciPy** | Actively used | 🟢 | Optimizers (linprog, minimize_scalar), stats (ks_2samp, norm.cdf) |
| **Formal Tests (pytest)** | No test files anywhere | 🔴 | Only `__main__` demo blocks. No `tests/` dirs, no assertions |
| **Snowflake** | Docs mention only | 🔴 | No code, no connector |
| **Vertex AI** | Docs mention only | 🔴 | No integration |
| **GCS** | Docs mention only | 🔴 | No `google-cloud-storage` usage |
| **Regression tasks** | Not present | 🔴 | Only classification examples |
| **Distributed Processing** | Not present | 🔴 | No Spark, Dask, or Ray |
| **Recommendation / Ranking** | Not present | 🔴 | Nice-to-have, but absent |
| **LLM familiarity** | Not demonstrated | 🔴 | Nice-to-have, but absent |
| **Type hints / strong typing** | Pydantic models in MLOps | 🟡 | Pydantic used in one project; other projects lack type annotations |
| **Git / code review / CI/CD** | Git repo exists | 🟡 | Repo exists but no CI/CD config, no branching strategy, no PR templates |

---

## What's Actually Strong ✅

Despite the gaps, these areas have **genuine, working implementations**:

1. **Mathematical optimization** (ad-optimization-engine) — LP, bounded minimization, Thompson Sampling with proper Bayesian updating
2. **Statistical testing** — Z-test for proportions, power analysis with statsmodels
3. **Uplift modeling** — S-Learner and T-Learner with heterogeneous treatment effects
4. **Drift detection** — KS-test and PSI with real scipy implementations
5. **FastAPI async serving** — Working async endpoints with Pydantic validation
6. **TF-IDF + KMeans clustering** — Real scikit-learn pipeline
7. **Documentation & concept guides** — Thorough theoretical coverage across all 5 areas
8. **Architecture design** — Well-articulated system diagrams and data flows

---

## Priority-Ordered Recommendations

### 🔴 Critical Gaps (Core Requirements — High Interview Risk)

| # | Gap | Recommendation | Effort | Status |
|---|---|---|---|---|
| 1 | **HuggingFace models are never used** | Replace heuristics in `text_enrichment.py` with real `pipeline("ner")`, `pipeline("sentiment-analysis")`, `pipeline("zero-shot-classification")` calls | Medium | ⬜ TODO |
| 2 | **PyTorch is never used** | Ensure at least one module does real `torch` inference (the HuggingFace fix above will trigger this) | Low (via #1) | ⬜ TODO |
| 3 | **No formal tests** | Add `pytest` test suites to all 4 projects. Even 2-3 tests per module with assertions would be a major improvement | Medium | ⬜ TODO |
| 4 | **CLIP is fully mocked** | Load real CLIP model (`openai/clip-vit-base-patch32`) in `multimodal_enrichment.py`, process a real image | Medium | ⬜ TODO |
| 5 | **No PostgreSQL integration** | Add `asyncpg` or `psycopg2` connection + prediction logging schema to MLOps project | Medium | ⬜ TODO |
| 6 | **No CI/CD pipeline** | Add a `.github/workflows/ci.yml` with lint, type-check, and test steps | Low | ⬜ TODO |
| 7 | **No audience targeting module** | Add KNN/cosine-similarity lookalike modeling to ad-optimization-engine | Medium | ⬜ TODO |

### 🟡 Important Gaps (Strengthen Profile)

| # | Gap | Recommendation | Effort | Status |
|---|---|---|---|---|
| 8 | **Docker needs hardening** | Add multi-stage build, `.dockerignore`, non-root user, `HEALTHCHECK` to Dockerfile | Low | ⬜ TODO |
| 9 | **No Snowflake integration** | Add `snowflake-connector-python` example for training data reads | Low | ⬜ TODO |
| 10 | **No GCS/Vertex AI code** | Add model artifact upload/download with `google-cloud-storage`; stub a Vertex AI pipeline | Medium | ⬜ TODO |
| 11 | **Missing causal inference methods** | Add Propensity Score Matching and/or DoubleML to causal-uplift-experimenter | Medium | ⬜ TODO |
| 12 | **No regression example** | Add a CTR/ROAS regression model (could fit in ad-optimization-engine) | Low | ⬜ TODO |
| 13 | **No TensorFlow** | Add a small `tf.keras` text classifier alongside the PyTorch models | Low | ⬜ TODO |
| 14 | **No Evidently AI** | Add Evidently data drift reports to MLOps project alongside custom KS/PSI | Low | ⬜ TODO |

### 🟢 Nice-to-Haves

| # | Gap | Recommendation | Effort | Status |
|---|---|---|---|---|
| 15 | LLM integration | Add prompt-based enrichment or RAG example | Low | ⬜ TODO |
| 16 | Recommendation / ranking | Add LightGBM ranker or collaborative filtering module | Medium | ⬜ TODO |
| 17 | pandas usage | Actually use pandas (currently listed but never imported) | Low | ⬜ TODO |
| 18 | Type hints | Add type annotations across all modules | Low | ⬜ TODO |
| 19 | Sequential testing | Add alpha-spending functions (O'Brien-Fleming, Pocock) | Medium | ⬜ TODO |

---

## What the README/Docs Describe vs. What Code Actually Does

This is worth noting because an interviewer who reviews the code will notice discrepancies:

| Claim (README/Docs) | Reality (Code) |
|---|---|
| "HuggingFace, PyTorch, CLIP, Scikit-Learn" | Only scikit-learn is actually used |
| "Entity extraction via `bert-base-NER`" | Regex matching against a hardcoded brand list |
| "Zero-shot stance via `bart-large-mnli`" | Rule-based keyword scanning |
| "CLIP multi-modal embeddings" | Random vectors seeded by filename length |
| "Evidently AI for drift monitoring" | Custom KS/PSI with scipy only |
| "PostgreSQL for prediction logging" | No database code at all |
| "CI/CD pipelines" | No pipeline config files |
| "Comprehensive test suites" | No test files anywhere |
| "PSM, DoubleML, IPW, Meta-learners" | Only S-Learner and T-Learner exist |

> [!CAUTION]
> This discrepancy between documentation claims and actual code is a risk in a technical interview. Recommend either **implementing the missing pieces** or **updating the docs to accurately reflect the current state** before discussing the project.

---

## Summary Table

| Category | Items Covered | Items Partial/Mock | Items Missing | Coverage |
|---|:---:|:---:|:---:|:---:|
| **Cultural Enrichment (NLP/Multi-modal)** | 1 | 4 | 4 | ~25% |
| **Ad Optimization** | 3 | 0 | 1 | ~75% |
| **Causal Inference / Experimentation** | 3 | 0 | 5 | ~38% |
| **MLOps / Production Engineering** | 4 | 2 | 6 | ~33% |
| **Cross-cutting Tools & Platforms** | 4 | 2 | 9 | ~27% |
| **Overall** | **15** | **8** | **25** | **~55%** |

> [!TIP]
> The **fastest wins** are items #1 (use real HuggingFace models), #3 (add pytest tests), #6 (add CI/CD YAML), and #8 (harden Docker). These four changes alone would jump coverage from ~55% to ~70% with relatively low effort.
