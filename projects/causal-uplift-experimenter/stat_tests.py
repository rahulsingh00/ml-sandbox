"""
Statistical Significance Testing Module
Executes standard statistical tests to evaluate A/B testing experiment outcomes.
"""

from typing import Dict, Any
import numpy as np
from scipy import stats


class ABTestEvaluator:
    """Computes statistical significance metrics for A/B testing campaigns."""

    @staticmethod
    def z_test_proportions(
        successes_a: int, trials_a: int, successes_b: int, trials_b: int
    ) -> Dict[str, Any]:
        """
        Performs a two-sided Z-test for difference in proportions.
        A is Control, B is Treatment.
        """
        p_a = successes_a / trials_a
        p_b = successes_b / trials_b
        
        # Pooled proportion
        p_pooled = (successes_a + successes_b) / (trials_a + trials_b)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/trials_a + 1/trials_b))
        
        # Z-statistic
        z_stat = (p_b - p_a) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Relative lift
        lift = (p_b - p_a) / p_a if p_a > 0 else 0.0
        
        return {
            "rate_control": round(p_a, 4),
            "rate_treatment": round(p_b, 4),
            "relative_lift": round(lift, 4),
            "z_stat": round(z_stat, 4),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05
        }


if __name__ == "__main__":
    # Scenario: A/B testing campaign for Brand Mentality targeting strategy
    control_clicks = 820
    control_impressions = 40000
    
    treatment_clicks = 940
    treatment_impressions = 40000
    
    evaluator = ABTestEvaluator()
    report = evaluator.z_test_proportions(
        successes_a=control_clicks,
        trials_a=control_impressions,
        successes_b=treatment_clicks,
        trials_b=treatment_impressions
    )
    
    print("=== A/B TESTING STATISTICAL EVALUATION RUN ===\n")
    print(f"Control (A):   {control_clicks:,} clicks / {control_impressions:,} impressions")
    print(f"Treatment (B): {treatment_clicks:,} clicks / {treatment_impressions:,} impressions")
    print("\nStatistical Report:")
    print(f"  Control CTR:          {report['rate_control'] * 100:.3f}%")
    print(f"  Treatment CTR:        {report['rate_treatment'] * 100:.3f}%")
    print(f"  Relative Lift:        {report['relative_lift'] * 100:.2f}%")
    print(f"  Z-Statistic:          {report['z_stat']}")
    print(f"  P-Value:              {report['p_value']:.5f}")
    print(f"  Reject Null (p < 5%): {report['is_significant']}")
    print("-" * 50)
