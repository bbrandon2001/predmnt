import streamlit as st
import pandas as pd
import joblib

# =========================
# Load trained assets
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Predictive Maintenance AI", layout="centered")

st.title(" Predictive Maintenance App")
st.markdown("Predict machine failure using AI")

# =========================
# User Inputs
# =========================
st.subheader("Enter Machine Parameters")

type_input = st.selectbox("Machine Type", ["L", "M", "H"])

air_temp = st.number_input("Air Temperature (K)", value=298.0)
process_temp = st.number_input("Process Temperature (K)", value=308.0)
rpm = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=50)

# =========================
# Prediction Button
# =========================
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Type": [type_input],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Encode categorical feature
    input_data = pd.get_dummies(input_data, drop_first=True)

    # Align columns with training data
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    st.subheader(" Results")

    if prediction == 1:
        st.error(" Failure Predicted")
    else:
        st.success(" No Failure")

    st.write(f"**Failure Probability:** {probability * 100:.2f}%")
