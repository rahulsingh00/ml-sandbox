import streamlit as st
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.append("projects/ad-optimization-engine")
from budget_allocator import BudgetAllocator
from bid_optimizer import BidOptimizer
from creative_bandit import ThompsonSamplingBandit

st.set_page_config(page_title="Ad Optimization Engine", page_icon="📊", layout="wide")

st.markdown("# 📊 Ad Optimization Engine")
st.write("Optimize advertising budget allocation, calculate utility-maximizing bid prices, and simulate multi-armed bandits.")
st.write("---")

tab1, tab2, tab3 = st.tabs(["💰 Budget Allocation", "📈 Bid Price Optimization", "🎰 Creative Bandit Selector"])

with tab1:
    st.header("Linear Programming Budget Allocation")
    st.write("Distributes ad budget across Google Search, Meta Social, and CTV platforms under spend and diversity constraints to maximize overall conversion counts.")
    
    total_budget = st.slider("Total Campaign Budget ($):", min_value=2000, max_value=100000, value=25000, step=1000)
    
    # Run LP allocation
    allocator = BudgetAllocator(total_budget=total_budget)
    res = allocator.allocate()
    
    if res.get("success"):
        st.subheader("💡 Optimal Allocations")
        
        # Metric dashboard
        c1, c2, c3 = st.columns(3)
        c1.metric("Google Search", f"${res['allocations']['Google Search']:,}")
        c2.metric("Meta Social", f"${res['allocations']['Meta Social']:,}")
        c3.metric("Connected TV (CTV)", f"${res['allocations']['Connected TV']:,}")
        
        # Plot budget split
        allocations_dict = res["allocations"]
        fig_df = pd.DataFrame(list(allocations_dict.items()), columns=["Platform", "Allocated Spend"])
        fig = px.pie(fig_df, values="Allocated Spend", names="Platform", title="Budget Distribution Share", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Budget allocation solver failed. Check if minimum constraint configurations are feasible.")

with tab2:
    st.header("Utility-Maximizing Bid Price")
    st.write("Under First-Price auction rules, bid price is optimized against expected utility: `Utility = (Value - Bid) * P_win(Bid)`.")
    
    col1, col2 = st.columns(2)
    with col1:
        impression_value = st.slider("Estimated Value of Impression ($):", min_value=0.5, max_value=10.0, value=5.0, step=0.1)
        b0_mid = st.slider("Market Inflection Point b0 ($):", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        k_slope = st.slider("Market Competitiveness Factor (k):", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        
    optimizer = BidOptimizer(k=k_slope, b0=b0_mid)
    optimal_bid = optimizer.optimize_bid(impression_value)
    
    # Generate curves for plotting
    bids = np.linspace(0.0, impression_value, 100)
    p_wins = [optimizer.win_probability(b) for b in bids]
    utilities = [optimizer.expected_utility(b, impression_value) for b in bids]
    
    with col2:
        st.subheader("💡 Optimal Bid Result")
        st.metric("Utility-Maximizing Bid Price", f"${optimal_bid:.2f}", help="Maximizes expected value minus cost")
        
    # Plot curves
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bids, y=p_wins, name="Win Probability (P_win)", yaxis="y1", line=dict(color="blue", dash="dash")))
    fig.add_trace(go.Scatter(x=bids, y=utilities, name="Expected Utility", yaxis="y2", line=dict(color="green", width=3)))
    
    # Optimal bid vertical line
    fig.add_vline(x=optimal_bid, line_width=2, line_dash="dash", line_color="red", annotation_text="Optimal Bid")
    
    fig.update_layout(
        title="Expected Utility & Win Probability vs Bid Price",
        xaxis=dict(title="Bid Price ($)"),
        yaxis=dict(title="Win Probability", titlefont=dict(color="blue"), tickfont=dict(color="blue")),
        yaxis2=dict(title="Expected Utility ($)", titlefont=dict(color="green"), tickfont=dict(color="green"), anchor="x", overlaying="y", side="right")
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Thompson Sampling Multi-Armed Bandit")
    st.write("Dynamic Creative Selection: Solves the exploration-exploitation trade-off to find the creative with the highest Click-Through Rate (CTR).")
    
    c1, c2, c3 = st.columns(3)
    c1_ctr = c1.slider("Underlying CTR for Creative 1 (%):", min_value=0.5, max_value=15.0, value=2.0, step=0.1) / 100.0
    c2_ctr = c2.slider("Underlying CTR for Creative 2 (%):", min_value=0.5, max_value=15.0, value=5.0, step=0.1) / 100.0
    c3_ctr = c3.slider("Underlying CTR for Creative 3 (%):", min_value=0.5, max_value=15.0, value=3.0, step=0.1) / 100.0
    
    num_trials = st.number_input("Number of simulated ad impressions to run:", min_value=100, max_value=5000, value=1000, step=100)
    
    if st.button("Run Bandit Simulation"):
        bandit = ThompsonSamplingBandit(num_creatives=3)
        true_ctrs = [c1_ctr, c2_ctr, c3_ctr]
        
        selections = [0, 0, 0]
        clicks = [0, 0, 0]
        
        history = []
        
        # Run loop
        for t in range(1, num_trials + 1):
            chosen_id = bandit.select_creative()
            selections[chosen_id] += 1
            
            # Simulate real click based on true probabilities
            click = np.random.rand() < true_ctrs[chosen_id]
            if click:
                clicks[chosen_id] += 1
                
            bandit.update_prior(chosen_id, click)
            
            if t % 50 == 0 or t == num_trials:
                estimated = bandit.get_estimated_ctrs()
                history.append({
                    "Trial": t,
                    "Creative 1 (Est CTR)": estimated[0] * 100.0,
                    "Creative 2 (Est CTR)": estimated[1] * 100.0,
                    "Creative 3 (Est CTR)": estimated[2] * 100.0,
                })
                
        st.subheader("💡 Bandit Outcomes")
        
        r1, r2, r3 = st.columns(3)
        r1.metric("Creative 1", f"Pulled {selections[0]} times", f"Clicks: {clicks[0]} (Est: {bandit.get_estimated_ctrs()[0]*100:.2f}%)")
        r2.metric("Creative 2", f"Pulled {selections[1]} times", f"Clicks: {clicks[1]} (Est: {bandit.get_estimated_ctrs()[1]*100:.2f}%)")
        r3.metric("Creative 3", f"Pulled {selections[2]} times", f"Clicks: {clicks[2]} (Est: {bandit.get_estimated_ctrs()[2]*100:.2f}%)")
        
        # Plot learning curves
        hist_df = pd.DataFrame(history)
        fig = px.line(hist_df, x="Trial", y=["Creative 1 (Est CTR)", "Creative 2 (Est CTR)", "Creative 3 (Est CTR)"], 
                      title="CTR Convergence Rates Over Time (Thompson Learning Curves)",
                      labels={"value": "CTR (%)", "variable": "Creative"})
        st.plotly_chart(fig, use_container_width=True)
