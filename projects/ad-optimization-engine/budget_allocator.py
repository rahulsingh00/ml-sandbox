"""
Budget Allocation Module
Uses Linear Programming (LP) to distribute advertising spend across channels
to maximize conversions under resource constraints.
"""

import numpy as np
from scipy.optimize import linprog


class BudgetAllocator:
    """Allocates ad budgets across platforms using linear programming."""

    def __init__(self, total_budget: float):
        self.total_budget = total_budget

    def allocate(self) -> dict:
        """
        Solves the LP problem.
        Channels: [0: Google Search, 1: Meta Social, 2: Connected TV (CTV)]
        
        Objective: Maximize conversions
           Conversions = 0.12 * x_0 + 0.08 * x_1 + 0.15 * x_2
           (represented as minimizing the negative coefficients)
        
        Constraints:
           x_0 + x_1 + x_2 <= Total Budget (Budget limit)
           x_2 <= 0.40 * Total Budget      (Diversity limit: Max 40% on CTV)
           x_0 >= 1000                     (Minimum spend on Google)
        """
        # Objective coefficients (negative because linprog minimizes)
        c = [-0.12, -0.08, -0.15]
        
        # Inequality constraints A_ub * x <= b_ub
        A_ub = [
            [1, 1, 1],   # x_0 + x_1 + x_2 <= total_budget
            [0, 0, 1]    # x_2 <= 0.40 * total_budget
        ]
        b_ub = [
            self.total_budget,
            0.40 * self.total_budget
        ]
        
        # Bounds for each variable
        bounds = [
            (1000, None),                   # Google: min $1000, no max
            (0, None),                      # Meta: min $0, no max
            (0, None)                       # CTV: min $0, no max
        ]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            allocations = res.x
            return {
                "success": True,
                "allocations": {
                    "Google Search": round(allocations[0], 2),
                    "Meta Social": round(allocations[1], 2),
                    "Connected TV": round(allocations[2], 2),
                },
                "expected_conversions": round(-res.fun, 1),
                "total_spend": round(np.sum(allocations), 2)
            }
        
        return {"success": False, "message": res.message}


if __name__ == "__main__":
    budget = 10000.0
    allocator = BudgetAllocator(total_budget=budget)
    result = allocator.allocate()
    
    print("=== CAMPAIGN BUDGET ALLOCATION RUN ===\n")
    print(f"Total Portfolio Budget: ${budget:.2f}")
    if result["success"]:
        print(f"Total Spend Allocated:  ${result['total_spend']:.2f}")
        print(f"Expected Conversions:   {result['expected_conversions']}")
        print("\nOptimal Channel Allocations:")
        for channel, amt in result["allocations"].items():
            pct = (amt / budget) * 100
            print(f"  - {channel}: ${amt:.2f} ({pct:.1f}%)")
    else:
        print(f"Solver error: {result['message']}")
    print("-" * 45)
