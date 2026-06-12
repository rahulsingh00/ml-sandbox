"""
Double Machine Learning (DML) Estimator Module
Implements DML for debiased treatment effects using K-fold cross-fitting.
Features Neyman-orthogonal scores for estimating treatment effects (ATE/ATT) with confidence intervals.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from scipy.stats import norm


class DoubleMLEstimator:
    """Estimates debiased treatment effects using K-fold cross-fitting and Neyman-orthogonal scores."""

    def __init__(self, k_folds: int = 5, random_state: int = 42):
        """
        Initializes the DoubleMLEstimator.
        
        Args:
            k_folds: Number of folds for cross-fitting.
            random_state: Seed for reproducibility.
        """
        self.k_folds = k_folds
        self.random_state = random_state
        
        # Treatment effect estimate
        self.ate = None
        self.se = None
        self.ci_lower = None
        self.ci_upper = None
        self.p_value = None

    def estimate_effect(
        self, 
        X: pd.DataFrame, 
        treatment: np.ndarray, 
        outcome: np.ndarray,
        model_y: Any = None,
        model_t: Any = None
    ) -> Dict[str, float]:
        """
        Estimates the treatment effect (ATE) using cross-fitting.
        
        Args:
            X: Covariates DataFrame, shape [N, D]
            treatment: Binary treatment indicator, shape [N]
            outcome: Continuous outcome variable, shape [N]
            model_y: Custom regressor for outcome Y. Defaults to GradientBoostingRegressor.
            model_t: Custom classifier for treatment T. Defaults to GradientBoostingClassifier.
            
        Returns:
            Dictionary containing estimated ATE, standard error, 95% CI, p-value.
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        n_samples = len(treatment)
        
        # Initialize default base learners
        if model_y is None:
            model_y = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=self.random_state)
        if model_t is None:
            model_t = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=self.random_state)
            
        # Arrays to store residuals
        residuals_y = np.zeros(n_samples)
        residuals_t = np.zeros(n_samples)
        
        # K-fold cross-fitting setup
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in kf.split(X_np):
            # Split data
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = outcome[train_idx], outcome[val_idx]
            t_train, t_val = treatment[train_idx], treatment[val_idx]
            
            # Clone model structures
            fold_model_y = sklearn_clone(model_y)
            fold_model_t = sklearn_clone(model_t)
            
            # Step 1: Fit nuisance model for Y (Outcome on Covariates X)
            fold_model_y.fit(X_train, y_train)
            pred_y = fold_model_y.predict(X_val)
            residuals_y[val_idx] = y_val - pred_y
            
            # Step 2: Fit nuisance model for T (Treatment on Covariates X)
            fold_model_t.fit(X_train, t_train)
            if hasattr(fold_model_t, "predict_proba"):
                pred_t = fold_model_t.predict_proba(X_val)[:, 1]
            else:
                pred_t = fold_model_t.predict(X_val)
            residuals_t[val_idx] = t_val - pred_t
            
        # Step 3: Solve for treatment effect (theta / ATE) using Neyman-orthogonal score
        # Regress outcome residuals (residuals_y) on treatment residuals (residuals_t) without intercept
        denominator = np.dot(residuals_t, residuals_t)
        if denominator == 0:
            raise ValueError("Treatment residuals have zero variance. Nuisance models may be overfitted.")
            
        self.ate = np.dot(residuals_t, residuals_y) / denominator
        
        # Step 4: Asymptotic variance and standard error estimation
        # Score contribution: psi_i = (residuals_y_i - ATE * residuals_t_i) * residuals_t_i
        scores = (residuals_y - self.ate * residuals_t) * residuals_t
        
        # J is the expected value of denominator (mean of treatment residuals squared)
        J = np.mean(residuals_t**2)
        
        # Variance of ATE is mean(scores^2) / (J^2 * n)
        variance = np.mean(scores**2) / (J**2 * n_samples)
        self.se = np.sqrt(variance)
        
        # Confidence intervals
        z_crit = norm.ppf(0.975) # 95% confidence interval
        self.ci_lower = self.ate - z_crit * self.se
        self.ci_upper = self.ate + z_crit * self.se
        
        # P-value calculation (two-sided)
        z_stat = self.ate / self.se
        self.p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
        
        return {
            "ate": float(self.ate),
            "se": float(self.se),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "p_value": float(self.p_value)
        }


def sklearn_clone(estimator: Any) -> Any:
    """Helper to clone a scikit-learn estimator using sklearn's base clone."""
    from sklearn.base import clone
    return clone(estimator)


if __name__ == "__main__":
    print("=== DOUBLE MACHINE LEARNING (DML) ESTIMATOR TEST ===\n")
    
    # Generate confounded observational dataset (partially linear model)
    # Y = theta * T + g(X) + U
    # T = m(X) + V
    rng = np.random.default_rng(42)
    n = 1200
    
    # Covariates X (confounders)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-1, 1, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    
    # Non-linear confounding functions
    g_x = 2.0 * np.sin(x1) + 3.0 * (x2**2)
    m_x = 1.0 / (1.0 + np.exp(-(x1 + 2.0 * x2)))
    
    # Treatment assignment
    treatment = rng.binomial(n=1, p=m_x)
    
    # True treatment effect
    true_ate = 1.8
    
    # Outcome Y
    outcome = true_ate * treatment + g_x + rng.normal(0, 0.5, n)
    
    # Naive regression estimate (biased)
    naive_ate = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])
    print(f"Naive Treatment Effect (Confounded): {naive_ate:.4f} (True Effect: {true_ate})")
    
    # Run Double ML Estimator
    dml = DoubleMLEstimator(k_folds=5)
    results = dml.estimate_effect(X, treatment, outcome)
    
    print("\nDouble ML Causal Estimates:")
    print(f"  Estimated ATE: {results['ate']:.4f}")
    print(f"  Standard Error: {results['se']:.4f}")
    print(f"  95% Confidence Interval: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print(f"  P-value: {results['p_value']:.4e}")
