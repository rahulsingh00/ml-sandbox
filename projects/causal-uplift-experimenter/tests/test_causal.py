"""
Unit tests for the causal-uplift-experimenter project.
Tests A/B testing z-proportions, Cohen's d power, S/T-learners,
Propensity Score Matching (PSM), and Double Machine Learning (DML).
"""

import numpy as np
import pandas as pd

from stat_tests import ABTestEvaluator
from power_calculator import ExperimentSizer
from uplift_model import SLearnerUplift, TLearnerUplift
from propensity_matcher import PropensityMatcher
from double_ml_estimator import DoubleMLEstimator


# ----------------------------------------------------
# 1. Statistical Significance Tests
# ----------------------------------------------------

def test_ab_test_evaluator():
    # Significant difference
    res_sig = ABTestEvaluator.z_test_proportions(
        successes_a=80, trials_a=1000, successes_b=150, trials_b=1000
    )
    assert res_sig["is_significant"]
    assert res_sig["relative_lift"] > 0.0
    
    # Non-significant difference
    res_nonsig = ABTestEvaluator.z_test_proportions(
        successes_a=100, trials_a=1000, successes_b=105, trials_b=1000
    )
    assert not res_nonsig["is_significant"]


# ----------------------------------------------------
# 2. Power Analysis Tests
# ----------------------------------------------------

def test_experiment_sizer():
    # We must mock or import numpy if it's referenced in power_calculator.py
    import sys
    import numpy as np
    sys.modules['power_calculator'].np = np
    
    sizer = ExperimentSizer()
    
    # Large effect size requires smaller sample size
    n_large = sizer.calculate_sample_size(effect_size=0.5)
    # Small effect size requires larger sample size
    n_small = sizer.calculate_sample_size(effect_size=0.1)
    
    assert n_large < n_small
    assert n_large > 0
    
    # Verify calculate_power returns float between 0 and 1
    pwr = sizer.calculate_power(sample_size=100, effect_size=0.3)
    assert 0.0 <= pwr <= 1.0


# ----------------------------------------------------
# 3. Uplift Models (S-Learner, T-Learner) Tests
# ----------------------------------------------------

def test_uplift_models():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    W = rng.binomial(n=1, p=0.5, size=200)
    # Binary outcome
    y = rng.binomial(n=1, p=0.5, size=200)
    
    # S-Learner
    s_learner = SLearnerUplift()
    s_learner.fit(X, W, y)
    s_uplift = s_learner.predict_uplift(X)
    assert s_uplift.shape == (200,)
    
    # T-Learner
    t_learner = TLearnerUplift()
    t_learner.fit(X, W, y)
    t_uplift = t_learner.predict_uplift(X)
    assert t_uplift.shape == (200,)


# ----------------------------------------------------
# 4. Propensity Score Matching (PSM) Tests
# ----------------------------------------------------

def test_propensity_matcher():
    rng = np.random.default_rng(42)
    n = 100
    w_cov = rng.normal(0, 1, n)
    # Confounded treatment assignment
    logit = 0.5 * w_cov
    prob = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, prob)
    # Outcome
    outcome = 2.0 * treatment + w_cov + rng.normal(0, 0.1, n)
    
    df_cov = pd.DataFrame({"w": w_cov})
    
    matcher = PropensityMatcher(caliper_scale=0.2)
    scores = matcher.estimate_propensity_scores(df_cov, treatment)
    assert len(scores) == n
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    
    matched = matcher.perform_matching(treatment)
    assert len(matched) > 0
    
    balance = matcher.check_covariate_balance(df_cov, treatment)
    smd_before = balance.loc[0, "smd_before"]
    smd_after = balance.loc[0, "smd_after"]
    # Matching should generally improve balance (reduce SMD)
    assert abs(smd_after) < abs(smd_before)
    
    effects = matcher.estimate_effects(outcome)
    assert "att" in effects
    assert "att_se" in effects


# ----------------------------------------------------
# 5. Double Machine Learning (DML) Tests
# ----------------------------------------------------

def test_double_ml_estimator():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(0, 1, n)
    X = pd.DataFrame({"x": x})
    treatment = rng.binomial(1, 0.5, n)
    outcome = 1.5 * treatment + 2.0 * x + rng.normal(0, 0.1, n)
    
    dml = DoubleMLEstimator(k_folds=3)
    results = dml.estimate_effect(X, treatment, outcome)
    
    assert "ate" in results
    assert "se" in results
    assert "ci_lower" in results
    assert "ci_upper" in results
    assert results["ci_lower"] <= results["ate"] <= results["ci_upper"]
