# 📂 Source Tree Map

This document outlines the file structure of the ML Mastery Sandbox.

```
ml-sandbox/
├── README.md                                    # Root workspace guide
├── docs/                                        # Global documentation index
│   ├── README.md                                # Docs landing page
│   ├── architecture/
│   │   ├── overview.md                          # End-to-end system design & data flows
│   │   └── source-tree.md                       # This file map
│   └── concepts/
│       ├── 01-cultural-data-nlp.md              # Concept: NLP Enrichment models
│       ├── 02-multimodal-enrichment.md          # Concept: Vision-Language & Video
│       ├── 03-ad-optimization.md                # Concept: Mathematical Optimization
│       ├── 04-causal-uplift-ab.md               # Concept: Experimentation & Uplift Modeling
│       └── 05-production-mlops.md               # Concept: FastAPI, Docker, and Drift Detection
└── projects/
    ├── cultural-enrichment-pipeline/            # Project 1: Cultural Pipeline
    │   ├── README.md                            # Setup and code walkthrough
    │   ├── requirements.txt                     # Project-specific dependencies
    │   ├── text_enrichment.py                   # NER, Stance, Topic Classification
    │   ├── multimodal_enrichment.py             # CLIP-based safety & topic matching
    │   └── clustering.py                        # Sentence embedding & discovery clustering
    ├── ad-optimization-engine/                  # Project 2: Ad Optimizer
    │   ├── README.md                            # Optimization formulations guide
    │   ├── requirements.txt                     # Scipy, NumPy, PuLP
    │   ├── bid_optimizer.py                     # Bid pricing optimization solver
    │   ├── budget_allocator.py                  # Linear programming budget distributor
    │   └── creative_bandit.py                   # Multi-armed bandit for ad selection
    ├── causal-uplift-experimenter/              # Project 3: Causal Experimentation
    │   ├── README.md                            # Statistics & Causal frameworks guide
    │   ├── requirements.txt                     # Statsmodels, CausalML
    │   ├── power_calculator.py                  # Power calculations (MDE, sample size)
    │   ├── stat_tests.py                        # A/B significance test suite
    │   └── uplift_model.py                      # Double ML / Metalearner implementations
    └── MLOps-serving-infrastructure/            # Project 4: Serving & Drift
        ├── README.md                            # Production MLOps guide
        ├── requirements.txt                     # FastAPI, evidently, uvicorn
        ├── main.py                              # Async FastAPI serving endpoints
        ├── drift_detector.py                    # Kolmogorov-Smirnov & PSI drift calculator
        ├── schemas.py                           # Pydantic input/output schemas
        ├── Dockerfile                           # Service containerization
        └── docker-compose.yml                   # Multi-service local environment setup
```
