import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import shap
import os
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
available_counties = ['alameda', 'butte', 'contra costa', 'el dorado', 'fresno',
       'humboldt', 'imperial', 'kern', 'lake', 'los angeles', 'madera',
       'marin', 'mendocino', 'merced', 'monterey', 'napa', 'nevada',
       'orange', 'placer', 'riverside', 'sacramento', 'san bernardino',
       'san diego', 'san francisco', 'san joaquin', 'san luis obispo',
       'san mateo', 'santa barbara', 'santa clara', 'santa cruz',
       'shasta', 'siskiyou', 'solano', 'sonoma', 'stanislaus', 'tehama',
       'tulare', 'tuolumne', 'ventura', 'yolo']

# Build UI inputs
user_inputs = {
    'CLRSKY_SFC_SW_DWN': st.slider("Clear-Sky Surface Shortwave Radiation (Anomaly)", -3.0, 3.0, 0.0, 0.1),
    'CLRSKY_SFC_PAR_TOT': st.slider("Clear-Sky PAR (Photosynthetically Active Radiation)", -3.0, 3.0, 0.0, 0.1),
    'T2M_roll3_mean': st.slider("2m Temperature – 3-Day Rolling Mean", -3.0, 3.0, 0.0, 0.1),
    'T2M_lag1': st.slider("2m Temperature – 1 Day Ago", -3.0, 3.0, 0.0, 0.1),
    'T2MWET': st.slider("Wet-Bulb Temperature (Current)", -3.0, 3.0, 0.0, 0.1),
    'T2MWET_lag1': st.slider("Wet-Bulb Temperature – 1 Day Ago", -3.0, 3.0, 0.0, 0.1),
    'T2MWET_roll3_mean': st.slider("Wet-Bulb Temp – 3-Day Rolling Mean", -3.0, 3.0, 0.0, 0.1),
    'T2M_max7': st.slider("Max 2m Temperature – Past 7 Days", -3.0, 3.0, 0.0, 0.1),
    'T2MWET_max7': st.slider("Max Wet-Bulb Temp – Past 7 Days", -3.0, 3.0, 0.0, 0.1),
    'month': st.slider("Calendar Month", 1, 12, 7),
    'T2M_T2MWET_interact': st.slider("Interaction: 2m Temp × Wet-Bulb Temp", -3.0, 3.0, 0.0, 0.1),
    'county': st.selectbox("Target County", available_counties)
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
    st.subheader("🔍 SHAP Waterfall")
    X_sparse = pipeline.named_steps['preprocess'].transform(X_row)
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    X_row_transformed = pd.DataFrame(X_sparse.toarray(), columns=feature_names)
    
    shap_value = explainer(X_row_transformed)[0]

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_value, max_display=10, show=False)
    st.pyplot(fig)



