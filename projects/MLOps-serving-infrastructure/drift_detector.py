"""
Drift Detection Module
Computes Kolmogorov-Smirnov (KS) test and Population Stability Index (PSI)
to monitor baseline features against production inputs for covariate shift.
"""

from typing import Tuple, List
import numpy as np
from scipy import stats


class PopulationDriftMonitor:
    """Monitors live model features and compares distributions against a baseline."""

    @staticmethod
    def kolmogorov_smirnov_test(baseline: List[float], production: List[float]) -> Tuple[float, float]:
        """
        Executes a two-sample Kolmogorov-Smirnov test.
        Returns:
            ks_statistic: maximum difference between cumulative empirical distributions.
            p_value: probability of observing such difference under Null Hypothesis.
        """
        res = stats.ks_2samp(baseline, production)
        return float(res.statistic), float(res.pvalue)

    @staticmethod
    def population_stability_index(
        baseline: List[float], production: List[float], num_buckets: int = 10
    ) -> float:
        """
        Computes the Population Stability Index (PSI).
        Estimates distribution stability using decile binning.
        """
        base_arr = np.array(baseline)
        prod_arr = np.array(production)
        
        # Determine quantile-based bucket boundaries from baseline population
        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(base_arr, percentiles)
        buckets[0] = -np.inf
        buckets[-1] = np.inf
        
        # Calculate frequency distributions
        base_counts, _ = np.histogram(base_arr, bins=buckets)
        prod_counts, _ = np.histogram(prod_arr, bins=buckets)
        
        # Convert frequencies to relative proportions (add small epsilon to avoid divide-by-zero)
        p = base_counts / len(base_arr)
        q = prod_counts / len(prod_arr)
        
        # Handle zero frequencies safely via smoothing epsilon
        eps = 1e-4
        p = np.where(p == 0, eps, p)
        q = np.where(q == 0, eps, q)
        
        # Re-normalize
        p /= np.sum(p)
        q /= np.sum(q)
        
        # Calculate PSI formula
        psi_value = np.sum((p - q) * np.log(p / q))
        return float(psi_value)


if __name__ == "__main__":
    np.random.seed(42)
    
    # Simulate baseline features (e.g. historical CTR prediction scores)
    base_dist = np.random.normal(loc=0.03, scale=0.005, size=1000).tolist()
    
    # Scenario 1: Stable production data (same distribution, different sample size)
    stable_prod = np.random.normal(loc=0.03, scale=0.005, size=500).tolist()
    
    # Scenario 2: Drifted production data (significant mean increase/shift)
    drifted_prod = np.random.normal(loc=0.035, scale=0.006, size=500).tolist()
    
    monitor = PopulationDriftMonitor()
    
    # Evaluate stable scenario
    ks_stat_1, p_val_1 = monitor.kolmogorov_smirnov_test(base_dist, stable_prod)
    psi_1 = monitor.population_stability_index(base_dist, stable_prod)
    
    # Evaluate drifted scenario
    ks_stat_2, p_val_2 = monitor.kolmogorov_smirnov_test(base_dist, drifted_prod)
    psi_2 = monitor.population_stability_index(base_dist, drifted_prod)
    
    print("=== MODEL PRODUCTION DRIFT MONITOR RUN ===\n")
    print(f"Baseline Population Size: {len(base_dist)}")
    print("-" * 55)
    print("📈 SCENARIO 1: STABLE POPULATION COMPARISON")
    print(f"  KS Statistic:          {ks_stat_1:.4f}")
    print(f"  KS Test P-Value:       {p_val_1:.5f} (Drift: {p_val_1 < 0.05})")
    print(f"  PSI Metric:            {psi_1:.4f}")
    print(f"  PSI Verdict:           {'STABLE' if psi_1 < 0.1 else 'SHIFT DETECTED'}")
    
    print("-" * 55)
    print("⚠️ SCENARIO 2: DRIFTED POPULATION COMPARISON")
    print(f"  KS Statistic:          {ks_stat_2:.4f}")
    print(f"  KS Test P-Value:       {p_val_2:.5f} (Drift: {p_val_2 < 0.05})")
    print(f"  PSI Metric:            {psi_2:.4f}")
    print(f"  PSI Verdict:           {'STABLE' if psi_2 < 0.1 else 'SHIFT DETECTED'}")
    print("-" * 55)
