"""
Unit tests for the ad-optimization-engine project.
Tests bid optimization, budget allocation, Thompson Sampling bandits,
audience targeting, and CTR regression.
"""

import numpy as np
import pandas as pd
import pytest

from bid_optimizer import BidOptimizer
from budget_allocator import BudgetAllocator
from creative_bandit import ThompsonSamplingBandit
from audience_targeter import AudienceTargeter
from ctr_regression import CTRPredictor


# ----------------------------------------------------
# 1. Bid Optimizer Tests
# ----------------------------------------------------

def test_bid_optimizer():
    optimizer = BidOptimizer(k=1.5, b0=2.0)
    
    # Test win probability bounds
    assert 0.0 <= optimizer.win_probability(1.0) <= 1.0
    assert 0.0 <= optimizer.win_probability(3.0) <= 1.0
    
    # Expected utility of bid >= value should be <= 0
    assert optimizer.expected_utility(4.0, 3.0) <= 0.0
    
    # Test optimal bid bounds
    opt_bid = optimizer.optimize_bid(value=4.0)
    assert 0.0 <= opt_bid <= 4.0
    
    # Test zero value yields zero bid
    assert optimizer.optimize_bid(0.0) == 0.0


# ----------------------------------------------------
# 2. Budget Allocator Tests
# ----------------------------------------------------

def test_budget_allocator_feasible():
    allocator = BudgetAllocator(total_budget=10000.0)
    result = allocator.allocate()
    
    assert result["success"]
    assert result["total_spend"] <= 10000.0
    
    allocations = result["allocations"]
    # Verify Google Search constraint (Google >= 1000)
    assert allocations["Google Search"] >= 1000.0
    
    # Verify CTV diversity constraint (CTV <= 40% of budget = 4000)
    assert allocations["Connected TV"] <= 4000.0


def test_budget_allocator_infeasible():
    # If budget is less than the minimum Google spend of $1000, it should be infeasible
    allocator = BudgetAllocator(total_budget=500.0)
    result = allocator.allocate()
    assert not result["success"]


# ----------------------------------------------------
# 3. Creative Bandit Tests
# ----------------------------------------------------

def test_creative_bandit():
    bandit = ThompsonSamplingBandit(num_creatives=3)
    
    # Initially priors are uniform, estimated CTR should be 0.5
    ctrs = bandit.get_estimated_ctrs()
    assert ctrs == [0.5, 0.5, 0.5]
    
    # Select creative should return valid index
    chosen = bandit.select_creative()
    assert chosen in [0, 1, 2]
    
    # Update prior with a success (click)
    bandit.update_prior(1, click_observed=True)
    assert bandit.alphas[1] == 2.0
    assert bandit.betas[1] == 1.0
    assert bandit.get_estimated_ctrs()[1] == 2.0 / 3.0
    
    # Update prior with a failure (no click)
    bandit.update_prior(2, click_observed=False)
    assert bandit.alphas[2] == 1.0
    assert bandit.betas[2] == 2.0
    assert bandit.get_estimated_ctrs()[2] == 1.0 / 3.0


# ----------------------------------------------------
# 4. Audience Targeter Tests
# ----------------------------------------------------

def test_audience_targeter():
    rng = np.random.default_rng(42)
    seeds = rng.normal(loc=[0.8, 0.8, 0.8], scale=0.1, size=(10, 3))
    seed_ids = [f"s_{i}" for i in range(10)]
    
    targeter = AudienceTargeter(seeds, seed_ids)
    
    # Train lookalike classifier
    negatives = rng.normal(loc=[0.2, 0.2, 0.2], scale=0.1, size=(20, 3))
    targeter.train_classifier(negatives)
    
    # Candidates to select
    candidates = np.array([
        [0.85, 0.85, 0.85],      # High similarity
        [-0.85, -0.85, -0.85],   # Low similarity (opposite direction)
        [0.80, 0.80, 0.80]       # High similarity (but will exclude/cap)
    ])

    candidates_df = pd.DataFrame({"user_id": ["u1", "u2", "u3"]})
    
    # Exclude user 3
    targeter.add_exclusions(["u3"])
    
    # Target audience
    selected = targeter.select_audience(
        candidates_df, candidates, method="knn", similarity_threshold=0.7, max_frequency=3
    )
    
    # u1 should be selected, u2 should fail threshold, u3 should fail exclusion
    targeted_ids = selected["user_id"].tolist()
    assert "u1" in targeted_ids
    assert "u2" not in targeted_ids
    assert "u3" not in targeted_ids
    
    # Verify frequency capping
    targeter.record_impression("u1")
    targeter.record_impression("u1")
    targeter.record_impression("u1") # 3 impressions
    
    selected_capped = targeter.select_audience(
        candidates_df, candidates, method="knn", similarity_threshold=0.7, max_frequency=3
    )
    assert "u1" not in selected_capped["user_id"].tolist()


# ----------------------------------------------------
# 5. CTR Regression Tests
# ----------------------------------------------------

def test_ctr_regression():
    predictor = CTRPredictor()
    
    # Generate synthetic training data
    df = predictor.generate_synthetic_data(num_samples=100)
    assert len(df) == 100
    assert "actual_ctr" in df.columns
    
    # Fit model
    metrics = predictor.fit(df)
    assert "test_mse" in metrics
    assert "test_r2" in metrics
    assert predictor.is_trained
    
    # Predict
    sample_inference = pd.DataFrame({
        "hour_of_day": [12],
        "day_of_week": [2],
        "device_type": ["mobile"],
        "placement": ["banner"],
        "historical_ctr": [0.015],
        "bid_floor": [1.0]
    })
    preds = predictor.predict(sample_inference)
    assert len(preds) == 1
    assert isinstance(preds[0], float)
