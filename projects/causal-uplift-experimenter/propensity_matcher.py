"""
Propensity Score Matching (PSM) Module
Estimates treatment effects by matching treated and control units with similar propensity scores.
Includes covariate balance diagnostics (SMD) and treatment effect estimation (ATT, ATE).
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PropensityMatcher:
    """Performs Propensity Score Matching to estimate causal effects in observational studies."""

    def __init__(self, caliper_scale: float = 0.2):
        """
        Initializes the PropensityMatcher.
        
        Args:
            caliper_scale: Standard deviation fraction to set matching threshold (caliper).
        """
        self.caliper_scale = caliper_scale
        self.propensity_model = LogisticRegression(solver="liblinear", random_state=42)

        self.propensity_scores: Optional[np.ndarray] = None
        self.matched_indices: Optional[List[Tuple[int, int]]] = None  # List of tuples: (treated_idx, control_idx)

    def estimate_propensity_scores(self, X: pd.DataFrame, treatment: np.ndarray) -> np.ndarray:
        """
        Fits a logistic regression to predict treatment probability (propensity score) from covariates.
        """
        self.propensity_model.fit(X, treatment)
        # Probability of being treated (class 1)
        scores = self.propensity_model.predict_proba(X)[:, 1]
        self.propensity_scores = scores
        return scores

    def perform_matching(self, treatment: np.ndarray, caliper: Optional[float] = None) -> List[Tuple[int, int]]:
        """
        Performs greedy nearest-neighbor matching without replacement within a caliper.
        
        Args:
            treatment: Binary treatment indicator array (0 or 1)
            caliper: Maximum absolute distance in propensity score for a match. 
                     If None, defaults to caliper_scale * std(propensity_scores).
                     
        Returns:
            List of matched tuples: (treated_original_index, control_original_index)
        """
        if self.propensity_scores is None:
            raise ValueError("Propensity scores must be estimated before matching.")

        # Compute default caliper if not provided
        if caliper is None:
            caliper = self.caliper_scale * np.std(self.propensity_scores)
            
        treated_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]
        
        # Sort treated indices by propensity score (descending) to match highest-probability units first
        treated_scores = self.propensity_scores[treated_indices]
        sorted_treated_order = np.argsort(treated_scores)[::-1]
        sorted_treated_indices = treated_indices[sorted_treated_order]
        
        available_controls = set(control_indices)
        matched_pairs = []
        
        for t_idx in sorted_treated_indices:
            t_score = self.propensity_scores[t_idx]
            
            # Find closest available control unit
            best_c_idx = None
            min_dist = float('inf')
            
            for c_idx in available_controls:
                c_score = self.propensity_scores[c_idx]
                dist = abs(t_score - c_score)
                if dist < min_dist:
                    min_dist = dist
                    best_c_idx = c_idx
            
            # Match if within caliper
            if best_c_idx is not None and min_dist <= caliper:
                matched_pairs.append((t_idx, best_c_idx))
                available_controls.remove(best_c_idx)  # Without replacement
                
        self.matched_indices = matched_pairs
        print(f"Matched {len(matched_pairs)} pairs from {len(treated_indices)} treated and {len(control_indices)} controls (caliper={caliper:.4f}).")
        return matched_pairs

    def check_covariate_balance(self, X: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
        """
        Computes Standardized Mean Difference (SMD) for all covariates before and after matching.
        SMD < 0.1 is standard indicator of a well-balanced matched sample.
        """
        if self.matched_indices is None:
            raise ValueError("Matching must be performed before checking balance.")
            
        matched_t_idx = [pair[0] for pair in self.matched_indices]
        matched_c_idx = [pair[1] for pair in self.matched_indices]
        
        balance_metrics = []
        
        for col in X.columns:
            # Check if column is numeric. If not, convert to float (or one-hot)
            x_col = X[col].values.astype(float)
            
            # Unmatched groups
            t_unmatched = x_col[treatment == 1]
            c_unmatched = x_col[treatment == 0]
            
            # Matched groups
            t_matched = x_col[matched_t_idx]
            c_matched = x_col[matched_c_idx]
            
            # Mean and Var before matching
            mean_t_un = np.mean(t_unmatched)
            mean_c_un = np.mean(c_unmatched)
            var_t_un = np.var(t_unmatched, ddof=1)
            var_c_un = np.var(c_unmatched, ddof=1)
            
            # Mean and Var after matching
            mean_t_mat = np.mean(t_matched)
            mean_c_mat = np.mean(c_matched)
            var_t_mat = np.var(t_matched, ddof=1)
            var_c_mat = np.var(c_matched, ddof=1)
            
            # SMD formulas
            smd_before = (mean_t_un - mean_c_un) / np.sqrt((var_t_un + var_c_un) / 2.0)
            
            pooled_std_after = np.sqrt((var_t_mat + var_c_mat) / 2.0)
            smd_after = (mean_t_mat - mean_c_mat) / (pooled_std_after if pooled_std_after > 0 else 1.0)
            
            balance_metrics.append({
                "covariate": col,
                "mean_treated_before": mean_t_un,
                "mean_control_before": mean_c_un,
                "smd_before": smd_before,
                "mean_treated_after": mean_t_mat,
                "mean_control_after": mean_c_mat,
                "smd_after": smd_after
            })
            
        return pd.DataFrame(balance_metrics)

    def estimate_effects(self, outcome: np.ndarray) -> Dict[str, float]:
        """
        Estimates the Average Treatment Effect on the Treated (ATT) from matched pairs.
        """
        if self.matched_indices is None:
            raise ValueError("Matching must be performed before estimating effects.")
            
        matched_t_idx = [pair[0] for pair in self.matched_indices]
        matched_c_idx = [pair[1] for pair in self.matched_indices]
        
        y_treated = outcome[matched_t_idx]
        y_control = outcome[matched_c_idx]
        
        att = float(np.mean(y_treated - y_control))
        
        # Simple standard error of the matched difference
        differences = y_treated - y_control
        se_att = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
        
        return {
            "att": att,
            "att_se": se_att,
            "t_statistic": att / se_att if se_att > 0 else 0.0
        }


if __name__ == "__main__":
    print("=== PROPENSITY SCORE MATCHING CAUSAL TEST ===\n")
    
    # Generate synthetic observational dataset
    # Confounder W: age/tenure that affects both treatment assignment and outcome
    rng = np.random.default_rng(42)
    n = 1000
    
    w_covariate = rng.normal(loc=0.0, scale=1.0, size=n)
    
    # Propensity score (treatment assignment depends on confounder W)
    # Treated units are older/have higher tenure
    logit = -0.5 + 1.2 * w_covariate
    prop_true = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(n=1, p=prop_true)
    
    # Outcome Y: treatment effect of 2.5, but confounded by W (W adds +3.0)
    treatment_effect = 2.5
    outcome = 10.0 + treatment_effect * treatment + 3.0 * w_covariate + rng.normal(loc=0.0, scale=0.5, size=n)
    
    df_covariates = pd.DataFrame({"confounder_w": w_covariate})
    
    # Simple naive comparison before matching (severely biased due to W confounder)
    naive_diff = np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0])
    print(f"Naive Treatment Effect (Confounded): {naive_diff:.4f} (True Effect: {treatment_effect})")
    
    # Run PSM
    matcher = PropensityMatcher(caliper_scale=0.1)
    matcher.estimate_propensity_scores(df_covariates, treatment)
    matcher.perform_matching(treatment)
    
    # Check balance
    balance = matcher.check_covariate_balance(df_covariates, treatment)
    print("\nCovariate Balance Diagnostic:")
    print(balance[["covariate", "smd_before", "smd_after"]])
    
    # Estimate treatment effects
    effects = matcher.estimate_effects(outcome)
    print("\nCausal Estimates:")
    print(f"  Estimated ATT: {effects['att']:.4f} (SE: {effects['att_se']:.4f})")
    print(f"  T-statistic: {effects['t_statistic']:.2f}")
