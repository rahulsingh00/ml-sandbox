import streamlit as st
import sys
import os
import pandas as pd
import numpy as np

sys.path.append("projects/MLOps-serving-infrastructure")
from main import app
from fastapi.testclient import TestClient
from drift_detector import FeatureDriftDetector
from drift_reporter import EvidentlyDriftReporter

st.set_page_config(page_title="MLOps Serving & Monitoring", page_icon="🚀", layout="wide")

st.markdown("# 🚀 MLOps serving & monitoring infrastructure")
st.write("Simulate async FastAPI serving endpoints, database logger calls, and Evidently AI data drift audits.")
st.write("---")

tab1, tab2 = st.tabs(["⚡ FastAPI Serving Client", "🛡️ Evidently AI Drift Monitoring"])

# Initialize client in-process
@st.cache_resource
def get_api_client():
    return TestClient(app)

client = get_api_client()

with tab1:
    st.header("FastAPI Serving Endpoint Client")
    st.write("Simulates API calls to the serving layer. Queries the in-process FastAPI application container.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Send Bid Request (`/bid`)")
        user_id = st.text_input("User ID:", value="usr_201")
        brand_context = st.text_input("Brand Context (Category safety checks):", value="Nike shoe campaign launch")
        estimated_ctr = st.slider("Estimated CTR:", min_value=0.001, max_value=0.20, value=0.045, step=0.005)
        campaign_id = st.text_input("Campaign ID:", value="cmp_12")
        
        # User features
        feat1 = st.number_input("User Feature 1:", value=0.12)
        feat2 = st.number_input("User Feature 2:", value=-0.40)
        feat3 = st.number_input("User Feature 3:", value=0.90)
        
    if st.button("Submit Bid Request"):
        payload = {
            "user_id": user_id,
            "brand_context": brand_context,
            "estimated_ctr": estimated_ctr,
            "campaign_id": campaign_id,
            "user_features": [feat1, feat2, feat3]
        }
        
        with st.spinner("Submitting request to FastAPI..."):
            response = client.post("/bid", json=payload)
            
        with col2:
            st.subheader("💡 API Response JSON")
            st.code(response.text, language="json")
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("brand_safety_status") == "SAFE":
                    st.success("Bid Placed successfully!")
                else:
                    st.error("Bid blocked by Brand Safety filter!")
                    
    st.write("---")
    st.subheader("🔧 System Diagnostics (`/health`)")
    if st.button("Check Server Health"):
        response = client.get("/health")
        st.code(response.text, language="json")

with tab2:
    st.header("Evidently AI Feature Drift Audits")
    st.write("Simulates feature drift detection. Shift production inputs relative to baseline training data and run drift reports.")
    
    drift_shift = st.slider("Introduce Drift (Production feature mean shift offset):", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    
    if st.button("Run Evidently AI Audit"):
        with st.spinner("Generating datasets and computing drift metrics..."):
            # Generate baseline reference data (1000 samples)
            rng = np.random.default_rng(42)
            ref_data = rng.normal(0.0, 1.0, (500, 3))
            ref_df = pd.DataFrame(ref_data, columns=["Feature_A", "Feature_B", "Feature_C"])
            
            # Generate production current data (1000 samples) with drift shift
            curr_data = rng.normal(0.0, 1.0, (500, 3))
            curr_data[:, 0] += drift_shift  # Apply drift to Feature_A
            curr_data[:, 1] += (drift_shift * 0.5) # Apply partial drift to Feature_B
            curr_df = pd.DataFrame(curr_data, columns=["Feature_A", "Feature_B", "Feature_C"])
            
            # 1. Run FeatureDriftDetector (PSI / KS)
            detector = FeatureDriftDetector()
            psi_a = detector.calculate_psi(ref_df["Feature_A"].values, curr_df["Feature_A"].values)
            ks_stat, p_val = detector.calculate_ks_test(ref_df["Feature_A"].values, curr_df["Feature_A"].values)
            
            # 2. Run Evidently report
            os.makedirs("reports", exist_ok=True)
            reporter = EvidentlyDriftReporter(reports_dir="reports")
            metrics = reporter.generate_report(ref_df, curr_df, report_id="streamlit_demo")
            
        st.subheader("💡 Drift Summary Metrics")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Feature A PSI Metric", f"{psi_a:.4f}", 
                  help="PSI > 0.25 indicates significant population shift",
                  delta="High Drift" if psi_a > 0.25 else "Stable",
                  delta_color="inverse" if psi_a > 0.25 else "normal")
        c2.metric("Feature A KS-test P-value", f"{p_val:.4e}",
                  help="P-value < 0.05 indicates statistical drift",
                  delta="Drift Detected" if p_val < 0.05 else "Stable",
                  delta_color="inverse" if p_val < 0.05 else "normal")
        c3.metric("Evidently Global Drift Status", "DRIFT DETECTED" if metrics.get("dataset_drift_detected") else "STABLE")
        
        # Read and display Evidently html report
        html_path = "reports/drift_report_streamlit_demo.html"
        if os.path.exists(html_path):
            st.write("---")
            st.subheader("📊 Evidently AI Dashboard Snapshots")
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
