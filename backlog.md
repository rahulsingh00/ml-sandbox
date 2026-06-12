# 📋 ML Lead Mastery Sandbox — Agile Project Backlog

This backlog outlines the Epics, User Stories, and future roadmap of the ML Sandbox. It serves as a tool for planning sprint deliverables and aligning technical implementation with business outcomes.

---

## 🏛️ Business Epics

### Epic 1: Cultural Context & Semantic Enrichment (Sub-project: `cultural-enrichment-pipeline`)
*   **Goal**: Extract deep NLP and multimodal properties from unstructured social streams to determine context, stance, sentiment, and safety relative to brands.
*   **Business Value**: Protects brand safety, targets cultural contextual alignments in real-time, and reduces media wastage.

### Epic 2: Real-time Bidding & Budget Optimization (Sub-project: `ad-optimization-engine`)
*   **Goal**: Implement mathematical optimization models to divide budget resources and adjust bidding bids to maximize expected utility.
*   **Business Value**: Maximizes conversion counts under hard constraints, ensures optimal bid values on ad networks, and manages creative explore/exploit learning loops.

### Epic 3: Incrementality & Causal Experimentation Platform (Sub-project: `causal-uplift-experimenter`)
*   **Goal**: instrument rigorous hypothesis testing, power analyses, and causal inference methods to measure true conversion uplift.
*   **Business Value**: Isolates treatment effects from complex selection bias/confounders to prove true marketing return-on-ad-spend (ROAS).

### Epic 4: Async Model Serving & Quality Monitoring (Sub-project: `MLOps-serving-infrastructure`)
*   **Goal**: Serve low-latency inference endpoints via FastAPI, log predictions, and calculate population drift stats.
*   **Business Value**: Ensures reliable high-throughput serving, logs analytics details, and fires alarms when production distributions shift away from baseline training data.

---

## 📝 User Stories & Status

### 🎨 Epic 1: Cultural Enrichment
*   **[COMPLETED]** *User Story 1.1: Named Entity Recognition*
    - **Description**: As a Campaign Strategist, I want to extract brands, people, and locations from social posts, so that I can automatically classify campaigns.
    - **Verification**: `tests/test_text_enrichment.py` (`test_ner_pipeline`)
*   **[COMPLETED]** *User Story 1.2: Zero-shot Stance Classification*
    - **Description**: As a Brand Manager, I want to classify if a post is favorable or critical of a target brand, so I can respond appropriately.
    - **Verification**: `tests/test_text_enrichment.py` (`test_classify_stance`)
*   **[COMPLETED]** *User Story 1.3: Visual Safety Checks*
    - **Description**: As a Brand Safety Officer, I want to flag images/videos containing unsafe objects, so that I can prevent ad placement on toxic media.
    - **Verification**: `tests/test_multimodal.py` (`test_evaluate_image_safety`, `test_evaluate_video_safety`)

### 📊 Epic 2: Ad Optimization
*   **[COMPLETED]** *User Story 2.1: LP Budget Allocator*
    - **Description**: As a Media Planner, I want to allocate spend across Search, Social, and TV to maximize conversions under constraints, so I can optimize my return.
    - **Verification**: `tests/test_ad_engine.py` (`test_budget_allocator_feasible`)
*   **[COMPLETED]** *User Story 2.2: Thompson Sampling Bandit*
    - **Description**: As a Creative Director, I want to dynamically serve the best performing ad creative, so that I can maximize campaign CTRs.
    - **Verification**: `tests/test_ad_engine.py` (`test_creative_bandit`)
*   **[COMPLETED]** *User Story 2.3: Lookalike Targeter*
    - **Description**: As an Audience Engineer, I want to identify candidate profiles similar to my seed audience, so that I can target high-probability users.
    - **Verification**: `tests/test_ad_engine.py` (`test_audience_targeter`)

### 🧪 Epic 3: Causal Experimentation
*   **[COMPLETED]** *User Story 3.1: Propensity Score Matcher*
    - **Description**: As an Experimentation Lead, I want to match control and treatment users in observational logs, so that I can eliminate confounder bias.
    - **Verification**: `tests/test_causal.py` (`test_propensity_matcher`)
*   **[COMPLETED]** *User Story 3.2: Double ML Estimation*
    - **Description**: As a Chief Statistician, I want to run debiased treatment calculations (DML), so that I can isolate the true incremental effect of the campaign.
    - **Verification**: `tests/test_causal.py` (`test_double_ml_estimator`)

### 🚀 Epic 4: MLOps Serving
*   **[COMPLETED]** *User Story 4.1: High-Performance Endpoint serving*
    - **Description**: As a serving Engineer, I want to query a low-latency endpoint with Pydantic contracts, so that I can run real-time predictions.
    - **Verification**: `tests/test_mlops.py` (`test_bid_endpoint_safe`)
*   **[COMPLETED]** *User Story 4.2: Evidendly AI Drift snapshots*
    - **Description**: As a Model Monitor, I want to view visual reports comparing baseline and production features, so I can identify model degradation.
    - **Verification**: `tests/test_mlops.py` (`test_trigger_drift_check`)

---

## 🔮 Future Enhancements & Product Backlog

These items represent future roadmap features to extend the capabilities of the sandbox:

| Task / Story | Epic Alignment | Priority | Complexity | Description |
|:---|:---|:---:|:---:|:---|
| **[BACKLOG]** LLM RAG Pipeline | Epic 1 | High | Medium | Integrate a local LLM (e.g. Llama-3 via Ollama) and a Vector database (ChromaDB) to perform retrieval-augmented generation for brand safety policies. |
| **[BACKLOG]** Recommendation System | Epic 2 | Medium | High | Implement a matrix factorization (Collaborative Filtering) or Deep Learning recommendation model to serve personalized creatives. |
| **[BACKLOG]** Vertex AI Retraining | Epic 4 | Medium | Medium | Complete the Vertex AI pipeline integration to trigger automatic model retraining in Google Cloud Platform when drift alerts are triggered. |
| **[BACKLOG]** Sequential A/B Testing | Epic 3 | Low | Medium | Implement sequential probability ratio tests (SPRT) to allow early stopping of A/B tests. |
