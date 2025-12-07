"""
🏭 Manufacturer Trust Evaluation Dashboard - PRODUCTION VERSION

Complete Streamlit dashboard with:
- SHAP explainability
- Layman-friendly explanations
- Anomaly detection
- Recommendations
- Beautiful UI/UX
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="Manufacturer Trust Evaluator",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# BACKEND CONFIG
# =====================================================================

BACKEND_URL = "http://localhost:5000"

@st.cache_data
def get_global_shap():
    """Fetch global SHAP importance"""
    try:
        r = requests.get(f"{BACKEND_URL}/shap-global", timeout=100)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def post_json(endpoint, payload):
    """Post JSON to backend"""
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=150)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Backend error: {str(e)}")
        return None

def check_backend_health():
    """Check if backend is running"""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

# =====================================================================
# HEADER & STATUS
# =====================================================================

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("# 🏭 Manufacturer Trust Evaluator")
    st.markdown("### AI-Powered Trust Score Analysis with SHAP Explainability")

with col3:
    if check_backend_health():
        st.success("✅ Backend Online")
    else:
        st.error("❌ Backend Offline")
        st.info("Start backend: `python flask_backend_production.py`")
        st.stop()

st.markdown("---")

# =====================================================================
# SIDEBAR - INPUT FORM
# =====================================================================

st.sidebar.markdown("# 📊 Manufacturer Profile")
st.sidebar.markdown("Enter manufacturer details below:")

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
for key, val in default_values.items():
    if isinstance(val, float):
        user_input[key] = st.sidebar.number_input(
            key.replace("_", " ").title(),
            value=float(val),
            step=0.1,
            format="%.2f"
        )
    else:
        user_input[key] = st.sidebar.number_input(
            key.replace("_", " ").title(),
            value=int(val),
            step=1
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### ✨ Instructions")
st.sidebar.markdown("""
1. **Adjust** parameters in the form
2. **Click** "Predict" button
3. **Explore** results in tabs
4. **Read** easy explanations
""")

# =====================================================================
# SESSION STATE
# =====================================================================

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_anomaly" not in st.session_state:
    st.session_state.last_anomaly = None
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = None

# =====================================================================
# TABS
# =====================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "🧠 Explainability",
    "🚨 Anomaly Detection",
    "🧭 Recommendations"
])

# ===========================
# TAB 1: PREDICTION
# ===========================

with tab1:
    st.markdown("## 🔮 Trust Score Prediction")
    st.markdown("Get AI-powered trust evaluation for your manufacturer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Predict Trust Score", use_container_width=True, key="btn_predict"):
            with st.spinner("⏳ Computing trust score..."):
                result = post_json("/predict", user_input)
                if result is not None:
                    st.session_state.last_prediction = result
                    st.balloons()
                    st.success("✅ Prediction complete!")
    
    with col2:
        if st.button("📊 Reset All", use_container_width=True):
            st.session_state.last_prediction = None
            st.rerun()
    
    with col3:
        st.markdown("**Last Updated:** " + datetime.now().strftime("%H:%M:%S"))
    
    st.markdown("---")
    
    # Display Results
    if st.session_state.last_prediction is not None:
        result = st.session_state.last_prediction
        score = result.get("trust_score", 0)
        
        # Main Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Trust Score",
                f"{score:.1%}",
                delta=f"{(score*100):.1f}/100"
            )
        
        with col2:
            if score >= 0.8:
                status = "🟢 EXCELLENT"
                delta_color = "off"
            elif score >= 0.6:
                status = "🟡 GOOD"
                delta_color = "off"
            elif score >= 0.4:
                status = "🟠 FAIR"
                delta_color = "inverse"
            else:
                status = "🔴 POOR"
                delta_color = "inverse"
            st.metric("📈 Status", status)
        
        with col3:
            confidence = min(score*100, 100)
            st.metric("🎯 Confidence", f"{confidence:.1f}%")
        
        with col4:
            st.metric("🤖 Model", "TensorFlow 2.20")
        
        st.markdown("")
        
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            title={'text': "Trust Score", 'font': {'size': 24}},
            delta={'reference': 75},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccc"},
                    {'range': [40, 60], 'color': "#ffffcc"},
                    {'range': [60, 80], 'color': "#ccffcc"},
                    {'range': [80, 100], 'color': "#ccffff"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        fig_gauge.update_layout(height=400, font={'size': 18})
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Summary Cards
        st.markdown("### 📋 Quick Summary")
        
        if score >= 0.8:
            summary_text = "🌟 Excellent manufacturer! High trust and reliability. Low risk for partnerships."
            color = "green"
        elif score >= 0.6:
            summary_text = "✅ Good manufacturer. Reliable performance with manageable risks."
            color = "blue"
        elif score >= 0.4:
            summary_text = "⚠️ Fair manufacturer. Several areas need improvement before engagement."
            color = "orange"
        else:
            summary_text = "❌ High risk manufacturer. Significant concerns require resolution."
            color = "red"
        
        st.info(summary_text)
    else:
        st.info("💡 Click '🚀 Predict Trust Score' to get started!")

# ===========================
# TAB 2: EXPLAINABILITY
# ===========================

with tab2:
    st.markdown("## 🧠 Feature Importance & Easy Explanation")
    st.markdown("Understand what drives the trust score")
    
    if st.session_state.last_prediction is not None:
        result = st.session_state.last_prediction
        local_shap = result.get("local_shap", [])
        trust_score = result.get("trust_score", 0)
        
        if local_shap and len(local_shap) > 0:
            # SHAP Chart
            st.markdown("### 📊 SHAP Feature Importance Chart")
            
            df_local = pd.DataFrame(local_shap)
            df_local_top = df_local.sort_values("abs_shap", ascending=True).tail(15)
            
            fig = px.bar(
                df_local_top,
                x="abs_shap",
                y="feature",
                color="shap_value",
                color_continuous_scale="RdBu",
                orientation="h",
                title="Top 15 Features Contributing to Trust Score",
                labels={
                    "abs_shap": "Impact (|SHAP Value|)",
                    "feature": "Feature",
                    "shap_value": "Contribution"
                }
            )
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View All Feature Values"):
                st.dataframe(
                    df_local.sort_values("abs_shap", ascending=False),
                    use_container_width=True
                )
            
            st.markdown("---")
            
            # Easy Explanation
            st.markdown("### 📝 Easy-to-Understand Explanation")
            
            df_all_shap = pd.DataFrame(local_shap)
            positive = df_all_shap[df_all_shap['shap_value'] > 0.001].sort_values('shap_value', ascending=False)
            negative = df_all_shap[df_all_shap['shap_value'] < -0.001].sort_values('shap_value', ascending=True)
            
            # Trust Level
            if trust_score >= 0.8:
                trust_level = "🟢 EXCELLENT (Very Trustworthy)"
            elif trust_score >= 0.6:
                trust_level = "🟡 GOOD (Trustworthy)"
            elif trust_score >= 0.4:
                trust_level = "🟠 FAIR (Somewhat Risky)"
            else:
                trust_level = "🔴 POOR (High Risk)"
            
            st.markdown(f"## {trust_level}")
            st.markdown(f"**Trust Score: {trust_score*100:.1f}/100**")
            
            st.markdown("---")
            
            # Strengths & Weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ What's Working Well")
                st.markdown("*Factors increasing trust*")
                
                if len(positive) > 0:
                    for idx, (i, row) in enumerate(positive.head(5).iterrows()):
                        feature_name = row['feature'].replace("_", " ").title()
                        contribution = row['shap_value']
                        st.write(f"**{idx+1}. {feature_name}**")
                        st.write(f"   💚 Contribution: `+{contribution:.4f}`")
                else:
                    st.info("No significant positive factors identified")
            
            with col2:
                st.markdown("### ❌ Areas for Improvement")
                st.markdown("*Factors decreasing trust*")
                
                if len(negative) > 0:
                    for idx, (i, row) in enumerate(negative.head(5).iterrows()):
                        feature_name = row['feature'].replace("_", " ").title()
                        contribution = abs(row['shap_value'])
                        st.write(f"**{idx+1}. {feature_name}**")
                        st.write(f"   ❌ Impact: `-{contribution:.4f}`")
                else:
                    st.success("✨ No significant weaknesses! Excellent performance!")
            
            st.markdown("---")
            
            # Recommendations
            st.markdown("### 🎯 Recommended Actions")
            
            if len(negative) > 0:
                recommendations = []
                for idx, (i, row) in enumerate(negative.head(3).iterrows()):
                    feature = row['feature'].lower()
                    feature_display = row['feature'].replace("_", " ").title()
                    
                    if "defect" in feature:
                        rec = f"🔧 **Reduce {feature_display}**: Invest in quality control"
                    elif "breach" in feature or "complaint" in feature:
                        rec = f"🛡️ **Address {feature_display}**: Implement robust processes"
                    elif "violation" in feature or "compliance" in feature:
                        rec = f"⚖️ **Ensure {feature_display}**: Meet regulatory requirements"
                    elif "rating" in feature or "reputation" in feature:
                        rec = f"🌟 **Improve {feature_display}**: Focus on customer satisfaction"
                    elif "debt" in feature or "margin" in feature:
                        rec = f"💰 **Strengthen {feature_display}**: Improve financial health"
                    else:
                        rec = f"📌 **Priority: {feature_display}**: Key area for improvement"
                    
                    recommendations.append(rec)
                
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("🎉 **Excellent!** No specific recommendations needed - all areas strong!")
            
            st.markdown("---")
            
            # Key Insight
            st.markdown("### 💡 Key Insight")
            
            try:
                insight_parts = []
                
                if len(positive) > 0:
                    top_pos = positive.iloc['feature'].replace('_', ' ').title()
                    insight_parts.append(f"**{top_pos}** is your strongest area")
                
                if len(negative) > 0:
                    top_neg = negative.iloc['feature'].replace('_', ' ').title()
                    if len(insight_parts) > 0:
                        insight_parts.append(f"while **{top_neg}** needs the most attention.")
                    else:
                        insight_parts.append(f"**{top_neg}** is the main focus area.")
                else:
                    insight_parts.append("and all factors are contributing positively!")
                
                insight = " ".join(insight_parts)
                st.info(f"💭 {insight}")
            except:
                st.info("Analysis complete!")
        else:
            st.info("⚠️ No SHAP data available. Re-run prediction.")
    else:
        st.info("💡 Run prediction first to see explainability analysis.")
    
    st.markdown("---")
    
    # Global SHAP
    st.markdown("### 🌍 Global Feature Importance")
    st.markdown("*Which features matter most across all manufacturers?*")
    
    global_data = get_global_shap()
    if global_data is not None:
        df_global = pd.DataFrame(global_data["global_importance"])
        df_global_top = df_global.sort_values("mean_abs_shap", ascending=True).tail(15)
        
        fig2 = px.bar(
            df_global_top,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global Feature Importance (Mean |SHAP|)",
            labels={"mean_abs_shap": "Mean Absolute SHAP", "feature": "Feature"},
            color="mean_abs_shap",
            color_continuous_scale="Viridis"
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Global SHAP data not available")

# ===========================
# TAB 3: ANOMALY DETECTION
# ===========================

with tab3:
    st.markdown("## 🚨 Anomaly Detection")
    st.markdown("Identifies unusual manufacturers based on autoencoder reconstruction error")
    
    if st.button("🔍 Check for Anomalies", use_container_width=True, key="btn_anomaly"):
        with st.spinner("🔎 Analyzing for anomalies..."):
            result = post_json("/anomaly", user_input)
            if result is not None:
                st.session_state.last_anomaly = result
                st.success("✅ Anomaly check complete!")
    
    st.markdown("---")
    
    if st.session_state.last_anomaly is not None:
        result = st.session_state.last_anomaly
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Reconstruction Error",
                f"{result['reconstruction_error']:.6f}"
            )
        
        with col2:
            st.metric(
                "🎯 Threshold (95%ile)",
                f"{result['threshold']:.6f}"
            )
        
        with col3:
            if result["is_anomaly"]:
                st.metric("🚨 Status", "ANOMALY")
            else:
                st.metric("✅ Status", "NORMAL")
        
        st.markdown("")
        
        if result["is_anomaly"]:
            st.error(f"""
            ⚠️ **ANOMALY DETECTED**
            
            Reconstruction error: `{result['reconstruction_error']:.6f}`
            Threshold: `{result['threshold']:.6f}`
            
            This manufacturer shows unusual patterns and deserves closer scrutiny.
            """)
        else:
            st.success(f"""
            ✅ **NORMAL PROFILE**
            
            Reconstruction error: `{result['reconstruction_error']:.6f}`
            Threshold: `{result['threshold']:.6f}`
            
            This manufacturer follows expected patterns in the dataset.
            """)
    else:
        st.info("💡 Click '🔍 Check for Anomalies' to analyze unusual patterns")

# ===========================
# TAB 4: RECOMMENDATIONS
# ===========================

with tab4:
    st.markdown("## 🧭 Similar Manufacturers")
    st.markdown("Find comparable manufacturers using embedding similarity")
    
    if st.button("🔎 Find Similar Manufacturers", use_container_width=True, key="btn_recommend"):
        with st.spinner("🔄 Finding similar manufacturers..."):
            result = post_json("/recommend", user_input)
            if result is not None:
                st.session_state.last_recommendations = result
                st.success("✅ Recommendations ready!")
    
    st.markdown("---")
    
    if st.session_state.last_recommendations is not None:
        result = st.session_state.last_recommendations
        df_neighbors = pd.DataFrame(result["neighbors"])
        
        st.markdown("### Top 5 Similar Manufacturers")
        
        # Table view
        st.dataframe(df_neighbors, use_container_width=True)
        
        st.markdown("")
        
        # Visualization
        fig = px.bar(
            df_neighbors,
            x="similarity",
            y="index",
            orientation="h",
            title="Similarity Scores to Nearest Neighbors",
            labels={"similarity": "Cosine Similarity", "index": "Manufacturer ID"},
            color="similarity",
            color_continuous_scale="Greens"
        )
        fig.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Average Similarity:** {df_neighbors['similarity'].mean():.4f}")
    else:
        st.info("💡 Click '🔎 Find Similar Manufacturers' to get recommendations")

# =====================================================================
# FOOTER
# =====================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <h3>📖 About This Dashboard</h3>
    <p>
    🔬 <b>Deep Learning Models:</b> 6-layer neural networks for trust prediction<br>
    🧠 <b>SHAP Explainability:</b> KernelExplainer for feature importance<br>
    🔍 <b>Anomaly Detection:</b> Autoencoder-based analysis<br>
    🎯 <b>Recommendations:</b> Embedding similarity matching<br>
    </p>
    <p style='color: #999; font-size: 12px;'>
    Powered by TensorFlow 2.20 | SHAP | Streamlit<br>
    Built for B2B Manufacturer Trust Evaluation
    </p>
    </div>
""", unsafe_allow_html=True)
