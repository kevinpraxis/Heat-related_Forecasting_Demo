import streamlit as st
import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === Format prompt by audience ===
def get_prompt_by_audience(explanation_text, prediction, audience="general"):
    label_text = "a spike (1)" if prediction == 1 else "not a spike (0)"
    timeframe = "within the next 3 days"

    if audience == "general":
        return f"""
The system predicted this case as **{label_text}**, meaning an unusual increase in heat-related emergency visits is expected **{timeframe}**.

Top influencing features (feature name, value, SHAP contribution):

{explanation_text}

Please rewrite this into a clear, plain-language explanation suitable for a concerned community member. Focus on *why* this spike might happen.
"""

    elif audience == "policy_maker":
        return f"""
This predictive model supports heat-health planning by forecasting whether a spike in emergency department (ED) visits is likely **{timeframe}**.

Key contributing drivers:

{explanation_text}

Summarize the likely causes of this spike and recommend 1–2 brief, policy-relevant interpretations or responses. Language should be accessible but evidence-informed.
"""

    elif audience == "scientific":
        return f"""
Model output: **{label_text}** (probable spike in heat-related ED visits **{timeframe}**)

Top SHAP contributors:

{explanation_text}

Provide a concise scientific interpretation. Discuss mechanistic relevance (e.g., wet bulb thermoregulation, heat lag effects, cumulative exposure) with clarity and precision.
"""

    else:
        return f"Audience type '{audience}' not recognized."


# === Main explanation function ===
def explain_with_openai_for_row(explainer, model_pipeline, X_row_raw, audience="general", top_n=5):
    if isinstance(X_row_raw, dict):
        X_row_raw = pd.DataFrame([X_row_raw])
    elif isinstance(X_row_raw, pd.Series):
        X_row_raw = X_row_raw.to_frame().T

    X_sparse = model_pipeline.named_steps['preprocess'].transform(X_row_raw)
    feature_names = model_pipeline.named_steps['preprocess'].get_feature_names_out()
    X_row_transformed = pd.DataFrame(X_sparse.toarray(), columns=feature_names)

    prediction = model_pipeline.predict(X_row_raw)[0]

    shap_row = explainer(X_row_transformed)[0]
    shap_vals = shap_row.values
    feature_vals = X_row_transformed.iloc[0].values

    top_features = sorted(
        zip(feature_names, shap_vals, feature_vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    explanation_text = "\n".join([
        f"{name} = {val:.2f}, SHAP: {impact:+.2f}"
        for name, impact, val in top_features
    ])

    prompt = get_prompt_by_audience(explanation_text, prediction, audience)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content


# === Build input from template ===
def build_input_from_template(template_df, user_inputs):
    if template_df.shape[0] == 0:
        raise ValueError("Template DataFrame is empty.")

    X_row = template_df.copy()
    for key, value in user_inputs.items():
        if key not in X_row.columns:
            raise KeyError(f"Column '{key}' not found in template.")
        X_row.at[X_row.index[0], key] = value

    return X_row


