"""
Bid Optimization Module
Calculates the utility-maximizing bid price in an ad auction context
using mathematical optimization.
"""

import numpy as np
from scipy.optimize import minimize_scalar


class BidOptimizer:
    """Optimizes real-time auction bids to maximize advertiser utility."""

    def __init__(self, k: float = 2.0, b0: float = 1.5):
        """
        k: Slope of the win probability curve (competitiveness factor)
        b0: Inflection point where win probability is 50%
        """
        self.k = k
        self.b0 = b0

    def win_probability(self, bid: float) -> float:
        """
        Computes win probability P_win(bid) using a logistic curve.
        Range is [0, 1].
        """
        return 1.0 / (1.0 + np.exp(-self.k * (bid - self.b0)))

    def expected_utility(self, bid: float, value: float) -> float:
        """
        Computes expected utility: U(bid) = (value - bid) * P_win(bid).
        Under First-Price auction rules.
        """
        # If bid is higher than value, utility is negative
        return (value - bid) * self.win_probability(bid)

    def optimize_bid(self, value: float) -> float:
        """
        Solves for the optimal bid b* that maximizes expected utility.
        """
        def objective(b):
            return -self.expected_utility(b, value)
        
        # Bounds: bid must be between 0 and the actual value of impression
        res = minimize_scalar(objective, bounds=(0, value), method='bounded')

        
        if res.success:
            return float(res.x)
        return 0.0


if __name__ == "__main__":
    optimizer = BidOptimizer(k=1.5, b0=2.0)
    
    # Test for various estimated impression values
    test_values = [1.0, 2.5, 4.0, 6.0]
    
    print("=== REAL-TIME BID OPTIMIZATION RUN ===\n")
    print(f"Auction market inflection point (b0): ${optimizer.b0}")
    print(f"Win probability steepness (k): {optimizer.k}\n")
    
    for val in test_values:
        opt_bid = optimizer.optimize_bid(val)
        win_prob = optimizer.win_probability(opt_bid)
        expected_ut = optimizer.expected_utility(opt_bid, val)
        
        print(f"Impression Value: ${val:.2f}")
        print(f"  Optimal Bid:    ${opt_bid:.2f}")
        print(f"  Win Probability: {win_prob * 100:.1f}%")
        print(f"  Expected Profit: ${expected_ut:.2f}")
        print("-" * 45)
