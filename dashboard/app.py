import streamlit as st

st.set_page_config(
    page_title="Sightly ML Sandbox Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f7fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    .card-body {
        color: #4a5568;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 ML Lead Mastery Sandbox</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Brand Mentality® & Ad Optimization Engine</div>', unsafe_allow_html=True)

st.markdown("""
This interactive dashboard demonstrates the core machine learning, optimization, and causal inference components built for the **Sightly ML Engineering Lead** workspace. 
It bridges the gap between complex model code and real-world campaign outcomes.

Use the sidebar navigation to explore the different layers of the pipeline in action.
""")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎨 Page 1: Cultural Enrichment Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">Extracts brand signals, stance, topics, sentiment, and visual/multimodal context from unstructured data. Uses pre-trained transformers and CLIP to verify brand safety.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Page 2: Ad Optimization Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">Solves budget allocations under platform constraints using Linear Programming, adjusts bid price utilities, and matches lookalike target profiles with KNN classifiers. Also runs Thomson-Sampling Bandit simulations.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Page 3: Causal Uplift Experimenter</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">Runs A/B testing statistical checks, power sizing calculations, Propensity Score Matching (PSM) for observational balance, and Double Machine Learning (DML) for debiased treatment effects.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🚀 Page 4: MLOps Serving & Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-body">Validates API contracts, tests async model serving endpoints, and runs data drift monitors (PSI, KS-Test) with interactive Evidently AI dashboard generation.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.markdown("### 🗺️ Business Problem Solved")
st.info("""
**The Challenge**: Brand campaign performance suffers from unstructured and noisy data (social posts, creative assets), target audience overlap, unmeasured marketing uplift, and production serving drift.

**The Solution Demonstrated Here**:
1. **Understand Context**: Extract precise cultural indicators using text NLP and image/video CLIP embeddings.
2. **Optimize Decisions**: Maximize utility and allocate budgets mathematically using Linear Programming and bandit learning.
3. **Validate Uplift**: Prove that campaigns cause performance uplift (ATT, ATE) using PSM and Double ML.
4. **Scale Safely**: Deploy and monitor servers in high-throughput environments, alerting when data distribution drift occurs.
""")
