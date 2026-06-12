# 🧠 Concept: Causal Inference & Experimentation

This guide covers experimentation design, statistical significance testing, and causal inference techniques like uplift modeling to isolate actual incremental ad-driven behavior.

---

## 1. A/B Testing & Power Analysis
Running online experiments requires careful statistical planning to prevent false positives ($\alpha$ error) or false negatives ($\beta$ error).

### Hypothesis Formulation
*   **Null Hypothesis ($H_0$)**: The new ad strategy does not change the conversion rate ($p_{\text{control}} = p_{\text{treatment}}$).
*   **Alternative Hypothesis ($H_1$)**: The new ad strategy changes the conversion rate ($p_{\text{control}} \neq p_{\text{treatment}}$).

### Statistical Power & Sample Size
Statistical Power ($1 - \beta$) is the probability of correctly rejecting the null hypothesis when a true effect exists (typically set to 80%).

To calculate the required sample size $n$ per group for a two-sample t-test of proportions, we use:
$$n \approx \frac{2 \bar{p}(1-\bar{p})(Z_{1-\alpha/2} + Z_{1-\beta})^2}{(p_{\text{treatment}} - p_{\text{control}})^2}$$
where $\bar{p} = \frac{p_{\text{control}} + p_{\text{treatment}}}{2}$, $Z_{1-\alpha/2}$ is the critical value for significance level $\alpha$ (typically 5%, $Z_{0.975} \approx 1.96$), and $Z_{1-\beta}$ is the critical value for power.

---

## 2. Causal Inference vs. Correlation
In advertising, users who see an ad and then convert might have converted anyway (e.g. searching for the brand directly). 
*   **Correlation**: Users who see ads convert at a higher rate.
*   **Causal Effect**: The ad *caused* the increase in conversions.

### Confounders & Selection Bias
Confounders ($X$) are variables that affect both the treatment assignment ($W \in \{0, 1\}$) and the outcome ($Y$).
Example: High-intent shoppers search for the brand ($X$), are targeted with retargeting ads ($W$), and convert ($Y$). If we do not control for shopping intent, we overestimate the ad's impact.

To adjust for confounders in observational data, we use methods like **Propensity Score Matching (PSM)**, where we estimate the probability of receiving treatment given features $X$:
$$e(X) = P(W = 1 \mid X)$$
and match treatment/control users with similar propensity scores.

---

## 3. Uplift Modeling (Incremental Targeting)
Uplift modeling estimates the Individual Treatment Effect (ITE) or Conditional Average Treatment Effect (CATE) to target users who are *persuadable* (only buy if they see the ad), avoiding *sure things* (buy anyway) or *lost causes* (never buy).

$$\text{Uplift} = E[Y \mid X, W=1] - E[Y \mid X, W=0]$$

### Metalearning Architectures

#### A. S-Learner (Single Learner)
A single machine learning model $M$ is trained on all data, including the treatment indicator $W$ as a feature:
$$\hat{Y} = M(X, W)$$
The uplift is estimated by predicting the difference:
$$\text{Uplift}(X) = M(X, 1) - M(X, 0)$$

#### B. T-Learner (Two Learners)
Two separate models are trained: $M_0$ on the control group ($W=0$) and $M_1$ on the treatment group ($W=1$).
$$\hat{Y}_0 = M_0(X), \quad \hat{Y}_1 = M_1(X)$$
The uplift is:
$$\text{Uplift}(X) = M_1(X) - M_0(X)$$
*Pro*: Prevents the treatment effect from being washed out by other strong features (which can happen in S-Learner if $W$ is given low feature importance).

---

## 🛠️ Sandbox Implementation
Check out the significance testing and uplift models in [`projects/causal-uplift-experimenter`](file:///Users/rahulsingh/.gemini/antigravity/scratch/ml-sandbox/projects/causal-uplift-experimenter).
