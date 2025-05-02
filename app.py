import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
from llm_utils import explain_with_openai_for_row  # Your LLM explanation function

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
components.iframe("https://your-username.github.io/climformatics-visuals/tem_spike_trends.html", height=600)

# === Section 2: User inputs ===
st.subheader("🧪 Try Prediction and Explanation")

# Load assets
pipeline = joblib.load("Models/hsp_pred_pipeline.pkl")
default_template = joblib.load("Models/default_input_template.pkl")
explainer = joblib.load("Models/explainer.pkl")

# Input fields
temp = st.slider("T2M (mean temp)", -3.0, 3.0, 0.0, 0.1)
wet = st.slider("T2MWET (wet bulb)", -3.0, 3.0, 0.0, 0.1)
county = st.selectbox("County", [col for col in default_template if "county_" in col])

# Construct input row
X_row = default_template.copy()
X_row["T2M"] = temp
X_row["T2MWET"] = wet
for col in X_row:
    if col.startswith("county_"):
        X_row[col] = 1 if col == county else 0
X_row_df = pd.DataFrame([X_row])

# Predict and explain
prediction = pipeline.predict(X_row_df)[0]
audience = st.selectbox("Explanation audience", ["general", "policy_maker", "scientific"])
if st.button("Generate explanation"):
    explanation = explain_with_openai_for_row(explainer, pipeline, X_row_df, audience)
    st.markdown(f"### 🧠 Explanation\n{explanation}")
