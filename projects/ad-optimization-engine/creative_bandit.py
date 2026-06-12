"""
Creative Selection Module
Implements a Thompson Sampling Multi-Armed Bandit (MAB) algorithm
to dynamically optimize selection of ad creatives based on click performance.
"""

from typing import List, Tuple
import numpy as np


class ThompsonSamplingBandit:
    """Thompson Sampling Solver for dynamic creative optimization."""

    def __init__(self, num_creatives: int):
        self.num_creatives = num_creatives
        # Initialize Beta prior distributions (alpha=1, beta=1 is Uniform prior)
        self.alphas = np.ones(num_creatives)
        self.betas = np.ones(num_creatives)

    def select_creative(self) -> int:
        """Draw samples from Beta distribution for each arm; select index of maximum."""
        samples = [np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.num_creatives)]
        return int(np.argmax(samples))

    def update_prior(self, creative_id: int, click_observed: bool) -> None:
        """Update alpha (successes) or beta (failures) based on user interaction."""
        if click_observed:
            self.alphas[creative_id] += 1
        else:
            self.betas[creative_id] += 1

    def get_estimated_ctrs(self) -> List[float]:
        """Calculates expected CTR (mean of the Beta distribution) for each creative."""
        return [float(self.alphas[i] / (self.alphas[i] + self.betas[i])) for i in range(self.num_creatives)]


if __name__ == "__main__":
    # Simulation parameters
    np.random.seed(42)
    # Underlying true click-through rates for 3 ad creatives
    true_ctrs = [0.02, 0.05, 0.03]  # Creative 1 (index 1) is the best
    num_trials = 1000
    
    bandit = ThompsonSamplingBandit(num_creatives=3)
    selections = [0, 0, 0]
    clicks = [0, 0, 0]
    
    # Run simulation loop
    for _ in range(num_trials):
        chosen_id = bandit.select_creative()
        selections[chosen_id] += 1
        
        # Simulate click based on true CTR probability
        click = np.random.rand() < true_ctrs[chosen_id]
        if click:
            clicks[chosen_id] += 1
            
        bandit.update_prior(chosen_id, click)
        
    estimated_ctrs = bandit.get_estimated_ctrs()
    
    print("=== THOMPSON SAMPLING BANDIT SIMULATION ===\n")
    print(f"Total Trials Simulated: {num_trials}")
    print(f"True CTR targets:       {true_ctrs}\n")
    
    for idx in range(3):
        print(f"Creative {idx}:")
        print(f"  Times Displayed:  {selections[idx]}")
        print(f"  Clicks Obtained:  {clicks[idx]}")
        print(f"  Empirical CTR:    {(clicks[idx]/max(1, selections[idx]))*100:.2f}%")
        print(f"  Estimated CTR:    {estimated_ctrs[idx]*100:.2f}%")
        print("-" * 40)
