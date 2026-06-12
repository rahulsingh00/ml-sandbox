"""
Uplift Modeling Module
Implements S-Learner and T-Learner architectures to estimate Individual
Treatment Effects (ITE) and identify persuadable target cohorts.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SLearnerUplift:
    """S-Learner (Single Model) approach for Uplift Modeling."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray) -> None:
        """Trains the single model containing features X and treatment indicator W."""
        # Concatenate treatment indicator as a feature
        features = np.column_stack((X, W))
        self.model.fit(features, y)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Estimates uplift: P(Y=1 | X, W=1) - P(Y=1 | X, W=0)."""
        features_treated = np.column_stack((X, np.ones(X.shape[0])))
        features_control = np.column_stack((X, np.zeros(X.shape[0])))
        
        # Predict probability of conversion (class index 1)
        p_treated = self.model.predict_proba(features_treated)[:, 1]
        p_control = self.model.predict_proba(features_control)[:, 1]
        
        return p_treated - p_control


class TLearnerUplift:
    """T-Learner (Two Models) approach for Uplift Modeling."""

    def __init__(self):
        self.model_treated = LogisticRegression()
        self.model_control = LogisticRegression()

    def fit(self, X: np.ndarray, W: np.ndarray, y: np.ndarray) -> None:
        """Trains separate models for treated and control groups."""
        X_treated = X[W == 1]
        y_treated = y[W == 1]
        
        X_control = X[W == 0]
        y_control = y[W == 0]
        
        self.model_treated.fit(X_treated, y_treated)
        self.model_control.fit(X_control, y_control)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Estimates uplift: P(Y=1 | X, W=1) - P(Y=1 | X, W=0)."""
        p_treated = self.model_treated.predict_proba(X)[:, 1]
        p_control = self.model_control.predict_proba(X)[:, 1]
        return p_treated - p_control


if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate observational dataset (1000 records, 3 features)
    n_samples = 1000
    X_sim = np.random.normal(size=(n_samples, 3))
    
    # Treatment assignment (50/50 probability)
    W_sim = np.random.binomial(n=1, p=0.5, size=n_samples)
    
    # Outcome Y: conversion rate depends on features, plus treatment interaction.
    # User cohort with X[:, 0] > 0 is highly persuadable (treatment has a large positive effect)
    base_score = 0.2 * X_sim[:, 0] - 0.1 * X_sim[:, 1]
    true_uplift = np.where(X_sim[:, 0] > 0, 0.3, -0.05)
    
    prob_conversion = 1.0 / (1.0 + np.exp(-(base_score + W_sim * true_uplift)))
    y_sim = np.random.binomial(n=1, p=prob_conversion)
    
    # Train models
    s_learner = SLearnerUplift()
    s_learner.fit(X_sim, W_sim, y_sim)
    
    t_learner = TLearnerUplift()
    t_learner.fit(X_sim, W_sim, y_sim)
    
    # Estimate uplift on new test users
    X_test = np.array([
        [1.5, 0.2, -0.5],   # Expected high persuadable (X[0] > 0)
        [-1.2, 0.5, 0.1]    # Expected sleeping dog/lost cause (X[0] <= 0)
    ])
    
    s_uplift = s_learner.predict_uplift(X_test)
    t_uplift = t_learner.predict_uplift(X_test)
    
    print("=== UPLIFT MODEL PREDICTIONS RUN ===\n")
    print("Test User 1 (High positive X[0]):")
    print(f"  S-Learner Uplift: {s_uplift[0] * 100:+.2f}%")
    print(f"  T-Learner Uplift: {t_uplift[0] * 100:+.2f}%")
    
    print("\nTest User 2 (Negative X[0]):")
    print(f"  S-Learner Uplift: {s_uplift[1] * 100:+.2f}%")
    print(f"  T-Learner Uplift: {t_uplift[1] * 100:+.2f}%")
    print("-" * 50)
