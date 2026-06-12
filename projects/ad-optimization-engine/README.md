# 📊 Ad Optimization Engine

This project contains Python implementations of ad-tech optimization systems, modeling bid price optimization, budget allocation under constraints, and dynamic creative selection using multi-armed bandits.

## 📁 File Structure

*   [`bid_optimizer.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/ad-optimization-engine/bid_optimizer.py): Calculates the utility-maximizing bid price using probability of winning curves.
*   [`budget_allocator.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/ad-optimization-engine/budget_allocator.py): Solves linear programming problems using `scipy.optimize.linprog` to distribute advertising budget across multiple platforms.
*   [`creative_bandit.py`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/ad-optimization-engine/creative_bandit.py): Selects optimal ad creatives dynamically using a Thompson Sampling Multi-Armed Bandit model.

## ⚙️ Installation & Setup

1.  Navigate to this project directory:
    ```bash
    cd projects/ad-optimization-engine
    ```
2.  Install requirements:
    ```bash
    pip3 install -r requirements.txt
    ```

## 🚀 Execution & Verification

Run the budget allocation solver:
```bash
python3 budget_allocator.py
```
This will solve a linear programming model representing budget constraints and output the optimal dollar allocations for Google, Meta, and CTV campaigns.
