"""
streamlit_dashboard_production.py - FIXED VERSION

Streamlit dashboard that talks to Flask backend:
- /predict : trust score + local SHAP
- /anomaly : anomaly result
- /recommend : neighbors
- /shap-global : global SHAP importance

Works with FIXED backend that uses stable KernelExplainer.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BACKEND_URL = "https://manufacturer-backend.onrender.com"


st.set_page_config(
    page_title="Manufacturer Trust Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Manufacturer Trust Evaluation Dashboard")
st.markdown("With SHAP Explainability & Deep Learning Analysis")

# -------------------------------------------------------------------
# 1. Helper functions
# -------------------------------------------------------------------

@st.cache_data
def get_global_shap():
    """Fetch global SHAP importance"""
    try:
        r = requests.get(f"{BACKEND_URL}/shap-global", timeout=100)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error fetching SHAP global importance: {e}")
        return None


def post_json(endpoint, payload):
    """Post JSON to backend"""
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=150)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error calling {endpoint}: {e}")
        return None


def check_backend_health():
    """Check if backend is running"""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False


# -------------------------------------------------------------------
# 2. Check backend status
# -------------------------------------------------------------------

if not check_backend_health():
    st.error("❌ Backend server is not running! Please start it first:")
    st.code("python flask_backend_production.py", language="bash")
    st.stop()

st.success("✅ Backend connected successfully!")

# -------------------------------------------------------------------
# 3. Sidebar: Input form
# -------------------------------------------------------------------

st.sidebar.header("📊 Input Manufacturer Features")

default_values = {
    "revenue_m": 120.0,
    "profit_margin": 18.0,
    "operational_eff": 0.85,
    "compliance_score": 92.0,
    "cust_rating": 4.5,
    "cust_feedback_count": 150,
    "sustainability_idx": 0.8,
    "innovation_idx": 0.7,
    "delivery_timeliness": 0.9,
    "after_sales": 4.2,
    "debt_to_equity": 0.8,
    "roa": 12.0,
    "defect_rate": 0.02,
    "capacity_util": 0.9,
    "on_time_delivery": 0.92,
    "warranty_claim_rate": 0.03,
    "reg_violation_count": 0,
    "certifications_count": 3,
    "market_share": 0.12,
    "employee_count": 800,
    "training_hours_per_emp": 12.0,
    "it_system_maturity": 0.7,
    "supplier_diversity": 0.5,
    "supply_chain_resilience": 0.75,
    "carbon_intensity": 0.4,
    "renewable_energy_ratio": 0.6,
    "region_risk_index": 0.2,
    "currency_volatility": 0.3,
    "avg_lead_time_days": 15.0,
    "expedite_ratio": 0.1,
    "it_incident_rate": 0.05,
    "data_quality_score": 0.8,
    "contract_breach_count": 0,
    "legal_dispute_count": 0,
    "brand_reputation": 0.75,
    "partnership_years": 6,
    "mgmt_stability_index": 0.8,
    "innovation_pipeline_strength": 0.7,
    "cybersecurity_maturity": 0.7,
    "digitization_level": 0.8,
}

user_input = {}
for i, (key, val) in enumerate(default_values.items()):
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        if isinstance(val, float):
            user_input[key] = st.number_input(
                key, value=float(val), step=0.1, label_visibility="visible"
            )
        else:
            user_input[key] = st.number_input(
                key, value=int(val), step=1, label_visibility="visible"
            )

st.sidebar.markdown("---")
st.sidebar.write("✨ Click buttons in main panel to run inference.")

# -------------------------------------------------------------------
# 4. Initialize session state
# -------------------------------------------------------------------

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_anomaly" not in st.session_state:
    st.session_state.last_anomaly = None
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = None

# -------------------------------------------------------------------
# 5. Tabs
# -------------------------------------------------------------------

tab_pred, tab_explain, tab_anom, tab_rec = st.tabs(
    ["🔮 Prediction", "🧠 Explainability (SHAP)", "🚨 Anomaly", "🧭 Recommendations"]
)

# 5.1 Prediction Tab
with tab_pred:
    st.subheader("Predict Trust Score")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Predict Trust Score", use_container_width=True):
            with st.spinner("Computing trust score..."):
                result = post_json("/predict", user_input)
                if result is not None:
                    st.session_state.last_prediction = result
                    st.success("✅ Prediction complete!")
                else:
                    st.error("❌ Prediction failed.")
    
    with col2:
        if st.button("📊 Reset", use_container_width=True):
            st.session_state.last_prediction = None
            st.rerun()

    # Display prediction result
    if st.session_state.last_prediction is not None:
        result = st.session_state.last_prediction
        score = result["trust_score"]
        
        st.markdown("---")
        
        # Trust score metric
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trust Score (0–1)", f"{score:.3f}")
        with col2:
            trust_pct = score * 100
            if trust_pct >= 75:
                status = "🟢 HIGH"
            elif trust_pct >= 50:
                status = "🟡 MEDIUM"
            else:
                status = "🔴 LOW"
            st.metric("Trust Level", status)
        with col3:
            st.metric("Confidence %", f"{min(score*100, 100):.1f}%")

        # Progress bar
        st.progress(min(max(score, 0), 1))
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={'text': "Trust Score (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# 5.2 Explainability Tab
with tab_explain:
    st.subheader("🧠 Feature Importance Analysis (SHAP)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Local Explanation** (for current prediction)")
    with col2:
        if st.button("🔄 Refresh", key="refresh_local"):
            pass

    # Local SHAP
    if st.session_state.last_prediction is not None:
        result = st.session_state.last_prediction
        local_shap = result.get("local_shap", [])
        
        if local_shap:
            df_local = pd.DataFrame(local_shap)
            df_local = df_local.sort_values("abs_shap", ascending=True).tail(15)
            
            fig = px.bar(
                df_local,
                x="abs_shap",
                y="feature",
                color="shap_value",
                color_continuous_scale="RdBu",
                orientation="h",
                title="Local Feature Contributions (|SHAP|, colored by sign)",
                labels={"abs_shap": "Absolute SHAP Value", "feature": "Feature"}
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show details table
            with st.expander("📋 Detailed Values"):
                st.dataframe(df_local, use_container_width=True)
        else:
            st.info("⚠️ Local SHAP explanation not available. SHAP may still be computing.")
    else:
        st.info("💡 Run a prediction first to see local SHAP explanation.")

    st.markdown("---")
    
    # Global SHAP
    st.write("**Global Importance** (across all validation data)")
    
    global_data = get_global_shap()
    if global_data is not None:
        df_global = pd.DataFrame(global_data["global_importance"])
        df_global = df_global.sort_values("mean_abs_shap", ascending=True).tail(15)
        
        fig2 = px.bar(
            df_global,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global Feature Importance (Mean |SHAP|)",
            labels={"mean_abs_shap": "Mean Absolute SHAP", "feature": "Feature"}
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("❌ Global SHAP importance not available.")


# 5.3 Anomaly Tab
with tab_anom:
    st.subheader("🚨 Anomaly Detection")
    st.write("Detects unusual manufacturers based on reconstruction error from autoencoder.")
    
    if st.button("🔎 Check Anomaly", use_container_width=True):
        with st.spinner("Checking for anomalies..."):
            result = post_json("/anomaly", user_input)
            if result is not None:
                st.session_state.last_anomaly = result
                st.success("✅ Anomaly check complete!")
            else:
                st.error("❌ Anomaly check failed.")

    if st.session_state.last_anomaly is not None:
        result = st.session_state.last_anomaly
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Reconstruction Error", f"{result['reconstruction_error']:.6f}")
        with col2:
            st.metric("Threshold (95%ile)", f"{result['threshold']:.6f}")
        with col3:
            if result["is_anomaly"]:
                st.metric("Status", "🚨 ANOMALY")
            else:
                st.metric("Status", "✅ NORMAL")

        if result["is_anomaly"]:
            st.error(
                f"⚠️ This manufacturer is flagged as ANOMALOUS. "
                f"Reconstruction error ({result['reconstruction_error']:.6f}) exceeds threshold ({result['threshold']:.6f})."
            )
        else:
            st.success(
                f"✅ This manufacturer appears NORMAL. "
                f"Reconstruction error is within expected range."
            )


# 5.4 Recommendations Tab
with tab_rec:
    st.subheader("🧭 Similar Manufacturers")
    st.write("Find the top 5 most similar manufacturers in the dataset using embedding similarity.")
    
    if st.button("🔍 Get Recommendations", use_container_width=True):
        with st.spinner("Finding similar manufacturers..."):
            result = post_json("/recommend", user_input)
            if result is not None:
                st.session_state.last_recommendations = result
                st.success("✅ Recommendations ready!")
            else:
                st.error("❌ Recommendation search failed.")

    if st.session_state.last_recommendations is not None:
        result = st.session_state.last_recommendations
        df_neighbors = pd.DataFrame(result["neighbors"])
        
        st.markdown("---")
        
        # Display as table
        st.dataframe(df_neighbors, use_container_width=True)
        
        # Visualize similarity
        fig = px.bar(
            df_neighbors,
            x="similarity",
            y="index",
            orientation="h",
            title="Similarity Scores to Nearest Neighbors",
            labels={"similarity": "Cosine Similarity", "index": "Manufacturer Index"},
            color="similarity",
            color_continuous_scale="Greens"
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    ### 📖 About This Dashboard
    
    This dashboard demonstrates:
    - **Deep Learning** predictions using 6-layer neural networks
    - **SHAP Explainability** for feature importance analysis
    - **Anomaly Detection** using autoencoders
    - **Recommendations** using embedding similarity
    
    All powered by TensorFlow/Keras and SHAP libraries.
    """
)