import streamlit as st
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Format prompt by audience ===
def get_prompt_by_audience(explanation_text, prediction, audience="general"):
    label_text = "a spike (1)" if prediction == 1 else "not a spike (0)"

    if audience == "general":
        return f"""
A machine learning model predicted this case as **{label_text}**, meaning there is an unusual rise in emergency visits.

Here are the top model explanations (feature name, value, contribution):

{explanation_text}

Explain this in plain language for a non-technical audience. Focus on *why* this spike might occur. Use 3–5 sentences.
"""
    elif audience == "policy_maker":
        return f"""
This model was trained to support heat-health response planning. It predicts a **{label_text}** in emergency department (ED) visits.

Key contributing features and their values:

{explanation_text}

Please summarize the likely cause of this spike and suggest **policy-relevant interpretations** in 2–3 sentences. Use clear but technical language, appropriate for government or NGO briefings.
"""
    elif audience == "scientific":
        return f"""
Model prediction: **{label_text}** (spike in heat-related ED visits)

SHAP top features:

{explanation_text}

Please explain the mechanistic interpretation in **scientific terms** (e.g., thermoregulation, wet bulb effects), but still concise. Aim for clarity and precision.
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


