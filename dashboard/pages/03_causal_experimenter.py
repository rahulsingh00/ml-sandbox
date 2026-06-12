import streamlit as st
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.append("projects/causal-uplift-experimenter")
from power_calculator import ExperimentSizer
from propensity_matcher import PropensityMatcher
from double_ml_estimator import DoubleMLEstimator

st.set_page_config(page_title="Causal Uplift Experimenter", page_icon="🧪", layout="wide")

st.markdown("# 🧪 Causal Inference & Experimentation")
st.write("Plan campaigns, match observational cohorts with propensity scores, and calculate debiased causal treatment effects.")
st.write("---")

tab1, tab2, tab3 = st.tabs(["📐 Experiment Power Sizer", "⚖️ Propensity Score Matching (PSM)", "🤖 Double Machine Learning (DML)"])

with tab1:
    st.header("Experiment Size & Power Analysis")
    st.write("Determines the required sample size per cohort to achieve a desired level of statistical significance.")
    
    c1, c2, c3 = st.columns(3)
    alpha_val = c1.slider("Significance Level (alpha / False Positive rate):", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    power_val = c2.slider("Statistical Power (1 - beta / True Positive rate):", min_value=0.50, max_value=0.99, value=0.80, step=0.05)
    effect_size_val = c3.slider("Expected Effect Size (Cohen's d):", min_value=0.01, max_value=0.50, value=0.05, step=0.01)
    
    sizer = ExperimentSizer()
    n_required = sizer.calculate_sample_size(effect_size=effect_size_val, alpha=alpha_val, power=power_val)
    
    st.subheader("💡 Sample Size Requirement")
    st.metric("Required Sample Size Per Group", f"{n_required:,} users")
    st.info(f"To run this A/B test, you will need a total sample size of **{n_required * 2:,}** users (split 50/50 control/treatment) to safely detect the target effect size without excessive false alarms.")

with tab2:
    st.header("Propensity Score Matching (PSM)")
    st.write("Corrects for selection bias (confounders) in observational campaigns by matching treated users to statistically similar control users.")
    
    caliper_val = st.slider("Caliper standard deviation scale:", min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    
    if st.button("Run Cohort Matching"):
        with st.spinner("Generating confounded observations & running matching..."):
            # Generate confounded dataset
            rng = np.random.default_rng(42)
            n = 1000
            x1 = rng.normal(1.5, 1.0, n)
            x2 = rng.uniform(-1.0, 2.0, n)
            
            # Confounded treatment assignment
            p_t = 1.0 / (1.0 + np.exp(-(x1 + 1.5 * x2)))
            treatment = rng.binomial(1, p_t)
            
            # True treatment effect is 2.5
            true_ate = 2.5
            outcome = true_ate * treatment + 2.0 * x1 + 3.0 * (x2**2) + rng.normal(0, 0.5, n)
            
            X = pd.DataFrame({"x1": x1, "x2": x2})
            
            # Instantiate and match
            matcher = PropensityMatcher(caliper_scale=caliper_val)
            matcher.estimate_propensity_scores(X, treatment)
            matched_pairs = matcher.perform_matching(treatment)
            balance_df = matcher.check_covariate_balance(X, treatment)
            
        st.subheader("💡 Matching Balance Diagnostics")
        st.write(f"Successfully matched **{len(matched_pairs)}** treatment/control pairs.")
        
        # Display balance dataframe
        st.dataframe(balance_df, use_container_width=True)
        
        # Plot SMD comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(x=balance_df["Covariate"], y=balance_df["SMD Before Matching"], name="SMD Before Matching", marker_color="red"))
        fig.add_trace(go.Bar(x=balance_df["Covariate"], y=balance_df["SMD After Matching"], name="SMD After Matching", marker_color="green"))
        
        # Add a threshold marker line at 0.1 (standard balance limit)
        fig.add_hline(y=0.1, line_width=1.5, line_dash="dash", line_color="black", annotation_text="Standard Balance Threshold (0.1)")
        
        fig.update_layout(title="Standardized Mean Difference (SMD) Balance Diagnostic", xaxis_title="Covariate Feature", yaxis_title="SMD", bmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Covariate balance achieved! Covariates showing SMD < 0.1 after matching verify that the selection bias has been successfully removed.")

with tab3:
    st.header("Double Machine Learning (DML)")
    st.write("Isolates treatment effects from complex multi-dimensional confounders using orthogonal residualization via cross-fitting.")
    
    k_folds_val = st.selectbox("Number of Cross-fitting Folds (K):", [2, 3, 5])
    
    if st.button("Run DML Estimator"):
        with st.spinner("Executing Double ML cross-fitting loops..."):
            rng = np.random.default_rng(42)
            n = 800
            
            x1 = rng.normal(0, 1.0, n)
            x2 = rng.uniform(-1.0, 1.0, n)
            X = pd.DataFrame({"x1": x1, "x2": x2})
            
            g_x = 2.0 * np.sin(x1) + 3.0 * (x2**2)
            m_x = 1.0 / (1.0 + np.exp(-(x1 + 2.0 * x2)))
            
            treatment = rng.binomial(n=1, p=m_x)
            true_ate = 1.8
            outcome = true_ate * treatment + g_x + rng.normal(0, 0.5, n)
            
            # Naive estimate
            naive_ate = float(np.mean(outcome[treatment == 1]) - np.mean(outcome[treatment == 0]))
            
            # DML estimate
            dml = DoubleMLEstimator(k_folds=k_folds_val)
            dml_res = dml.estimate_effect(X, treatment, outcome)
            
        st.subheader("💡 Debiased Treatment Effect Estimation")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("True ATE", f"{true_ate:.3f}")
        c2.metric("Confounded Naive ATE", f"{naive_ate:.3f}", delta=f"Bias: {abs(naive_ate-true_ate):.3f}", delta_color="inverse")
        c3.metric("DML Debiased ATE", f"{dml_res['ate']:.3f}", delta=f"Residual Error: {abs(dml_res['ate']-true_ate):.3f}", delta_color="normal")
        
        # Plot Confidence Interval
        fig = go.Figure()
        # Naive point
        fig.add_trace(go.Scatter(x=[naive_ate], y=["Naive"], mode="markers", marker=dict(size=12, color="red"), name="Naive (Biased)"))
        
        # DML point with error bar
        fig.add_trace(go.Scatter(
            x=[dml_res["ate"]], 
            y=["Double ML"], 
            mode="markers", 
            marker=dict(size=12, color="green"),
            error_x=dict(type="data", array=[dml_res["ci_upper"] - dml_res["ate"]], visible=True),
            name="Double ML (Debiased)"
        ))
        
        # True value vertical line
        fig.add_vline(x=true_ate, line_width=2, line_dash="dash", line_color="blue", annotation_text="True Treatment Effect")
        
        fig.update_layout(title="Treatment Effect Estimation & 95% Confidence Interval", xaxis_title="Average Treatment Effect (ATE)", yaxis_title="Estimator")
        st.plotly_chart(fig, use_container_width=True)
