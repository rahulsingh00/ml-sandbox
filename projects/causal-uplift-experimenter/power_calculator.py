"""
Power Calculator Module
Performs statistical power analysis for experiment design,
helping to size control and treatment groups before launch.
"""

from statsmodels.stats.power import TTestIndPower


class ExperimentSizer:
    """Computes required sample size, power, and Minimum Detectable Effect (MDE)."""

    def __init__(self):
        self.power_analysis = TTestIndPower()

    def calculate_sample_size(self, effect_size: float, alpha: float = 0.05, power: float = 0.80) -> int:
        """
        Calculates sample size needed per group.
        effect_size: Cohen's d value (difference in means / standard deviation)
        """
        n = self.power_analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative='two-sided'
        )
        return int(np.ceil(n)) if 'np' in globals() else int(n + 1)

    def calculate_power(self, sample_size: int, effect_size: float, alpha: float = 0.05) -> float:
        """Calculates post-hoc or planned statistical power for a given sample size."""
        power = self.power_analysis.solve_power(
            effect_size=effect_size,
            nobs1=sample_size,
            alpha=alpha,
            ratio=1.0,
            alternative='two-sided'
        )
        return float(power)


if __name__ == "__main__":
    import numpy as np  # Ensure imported inside execution block
    sizer = ExperimentSizer()
    
    # We want to detect a 5% relative lift in CTR.
    # Say baseline CTR = 2.0%, new CTR = 2.1%.
    # Cohen's d for proportions can be approximated or standard effect sizes evaluated.
    effect_sizes = [0.01, 0.02, 0.05, 0.10]
    
    print("=== EXPERIMENT DESIGN & POWER ANALYSIS RUN ===\n")
    print("Given Significance Level (alpha) = 5%")
    print("Desired Statistical Power (1 - beta) = 80%\n")
    
    for es in effect_sizes:
        n_req = sizer.calculate_sample_size(effect_size=es)
        print(f"For Cohen's d Effect Size = {es:.3f}:")
        print(f"  Required sample size per group: {n_req:,} users")
        print("-" * 50)
