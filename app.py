import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from llm_utils import explain_with_openai_for_row, build_input_from_template

# === Page setup ===
st.set_page_config(page_title="Climformatics – Heat Risk Demo", layout="wide")
st.title("🔥 Climformatics – Heat-Related Hospitalization Forecast")

st.markdown("""
This interactive demo shows:
- Temperature and hospitalization spike trends
- A predictive model with SHAP explanation
- GPT-generated narrative for decision support
""")

# === Section 1: Embedded trend plot ===
st.subheader("📈 Temperature–Spike Trends")
with open("Assets/tem_spike_trends.html", "r", encoding="utf-8") as f:
    html_string = f.read()
components.html(html_string, height=600)

# === Section 2: User inputs ===
st.subheader("🧪 Try Prediction and Explanation")

# Load assets
pipeline = joblib.load("Models/hsp_pred_pipeline.pkl")
default_template = joblib.load("Models/default_input_template.pkl")
explainer = joblib.load("Models/explainer.pkl")

# Get county options
available_counties = sorted(default_template["county"].unique())

# Build UI inputs
user_inputs = {
    'CLRSKY_SFC_SW_DWN': st.slider("Clear Sky SW Down", -3.0, 3.0, 0.0, 0.1),
    'CLRSKY_SFC_PAR_TOT': st.slider("PAR Total", -3.0, 3.0, 0.0, 0.1),
    'T2M_roll3_mean': st.slider("T2M Rolling Mean", -3.0, 3.0, 0.0, 0.1),
    'T2M_lag1': st.slider("T2M Lag-1", -3.0, 3.0, 0.0, 0.1),
    'T2MWET': st.slider("T2MWET", -3.0, 3.0, 0.0, 0.1),
    'T2MWET_lag1': st.slider("T2MWET Lag-1", -3.0, 3.0, 0.0, 0.1),
    'T2MWET_roll3_mean': st.slider("T2MWET Rolling Mean", -3.0, 3.0, 0.0, 0.1),
    'T2M_max7': st.slider("T2M Max Last 7 Days", -3.0, 3.0, 0.0, 0.1),
    'T2MWET_max7': st.slider("T2MWET Max Last 7 Days", -3.0, 3.0, 0.0, 0.1),
    'month': st.slider("Month", 1, 12, 7),
    'T2M_T2MWET_interact': st.slider("T2M × T2MWET", -3.0, 3.0, 0.0, 0.1),
    'county': st.selectbox("County", available_counties)
}

# Build model-ready input row
X_row = build_input_from_template(default_template, user_inputs)

# Explanation audience
audience = st.selectbox("Explanation audience", ["general", "policy_maker", "scientific"])

if st.button("Generate explanation"):
    # GPT-generated explanation
    explanation = explain_with_openai_for_row(explainer, pipeline, X_row, audience)
    st.markdown(f"### 🧠 Explanation\n{explanation}")

    # SHAP waterfall plot
    st.subheader("🔍 SHAP Waterfall Explanation")
    shap_value = explainer(X_row)[0]
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_value, max_display=10, show=False)
    st.pyplot(fig)
