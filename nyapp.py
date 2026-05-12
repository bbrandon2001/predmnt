import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Predictive Maintenance AI (Upgraded)", layout="centered")
st.title("🔧 Predictive Maintenance App (Upgraded Accuracy)")
st.markdown("Uses a tuned model + calibrated threshold to improve failure detection.")

@st.cache_resource
def load_assets():
    pipe = joblib.load("model_pipeline.pkl")
    threshold = joblib.load("threshold.pkl")
    return pipe, float(threshold)

pipe, THRESHOLD = load_assets()

st.subheader("Enter Machine Parameters")

type_input = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature (K)", value=298.1)
process_temp = st.number_input("Process Temperature (K)", value=308.6)
rpm = st.number_input("Rotational Speed (rpm)", value=1551)
torque = st.number_input("Torque (Nm)", value=42.8)
tool_wear = st.number_input("Tool Wear (min)", value=0)

# Optional: allow sensitivity tuning
st.subheader("Sensitivity")
threshold_ui = st.slider("Decision threshold (lower = catches more failures)", 0.05, 0.95, THRESHOLD, 0.01)

if st.button("Predict", type="primary"):
    X = pd.DataFrame({
        "Type": [type_input],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
    })

    prob_fail = pipe.predict_proba(X)[0][1]
    pred = 1 if prob_fail >= threshold_ui else 0

    st.metric("Failure Probability", f"{prob_fail*100:.2f}%")

    if pred == 1:
        st.error("⚠️ Predicted: FAILURE (high risk)")
    else:
        st.success("✅ Predicted: NO FAILURE (low risk)")

    # “Industrial” risk zones (helps interpretation)
    if prob_fail < 0.25:
        st.info("Zone: Safe")
    elif prob_fail < 0.50:
        st.warning("Zone: Monitor / Early warning")
    elif prob_fail < 0.75:
        st.warning("Zone: High risk")
    else:
        st.error("Zone: Critical risk")
