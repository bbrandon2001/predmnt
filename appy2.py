import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

# =========================
# App Config
# =========================
st.set_page_config(page_title="Predictive Maintenance AI (Accurate)", layout="centered")
st.title("🔧 Predictive Maintenance App (Accurate Readings)")
st.markdown("Predict machine failure using **probability + tuned threshold** for better detection.")

# =========================
# Load trained assets (cached)
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, TRAIN_COLUMNS = load_assets()

# =========================
# Helpers
# =========================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common predictive-maintenance engineered features.
    They will only matter if TRAIN_COLUMNS contains these names (we reindex later).
    """
    # Match dataset naming from predictive_maintenance.csv
    # (we keep both bracketed and non-bracketed fallbacks)
    air = "Air temperature [K]" if "Air temperature [K]" in df.columns else "Air Temperature (K)"
    proc = "Process temperature [K]" if "Process temperature [K]" in df.columns else "Process Temperature (K)"
    rpm = "Rotational speed [rpm]" if "Rotational speed [rpm]" in df.columns else "Rotational Speed (rpm)"
    tq  = "Torque [Nm]" if "Torque [Nm]" in df.columns else "Torque (Nm)"
    wear = "Tool wear [min]" if "Tool wear [min]" in df.columns else "Tool Wear (min)"

    if air in df.columns and proc in df.columns:
        df["Temp_Diff"] = df[proc] - df[air]

    if wear in df.columns and tq in df.columns:
        df["Wear_per_Torque"] = df[wear] / (df[tq] + 1.0)

    if rpm in df.columns and tq in df.columns:
        # simple load proxy
        df["Power_Proxy"] = df[rpm] * df[tq]

    return df

def preprocess_input(raw_df: pd.DataFrame) -> np.ndarray:
    """
    1) engineered features
    2) one-hot encode categoricals
    3) align columns to training columns
    4) scale with saved scaler
    """
    df = add_engineered_features(raw_df.copy())

    # One-hot encode (Type is categorical in the dataset)
    df = pd.get_dummies(df, drop_first=True)

    # Align to training columns
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    # Scale
    X_scaled = scaler.transform(df)
    return X_scaled

def predict_failure_probability(X_scaled: np.ndarray) -> float:
    """
    Returns P(failure) assuming binary classifier with predict_proba.
    """
    proba = model.predict_proba(X_scaled)[0][1]
    return float(proba)

# =========================
# User Inputs
# =========================
st.subheader("Enter Machine Parameters")

type_input = st.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature (K)", value=298.1)
process_temp = st.number_input("Process Temperature (K)", value=308.6)
rpm = st.number_input("Rotational Speed (rpm)", value=1551)
torque = st.number_input("Torque (Nm)", value=42.8)
tool_wear = st.number_input("Tool Wear (min)", value=0)

# Accuracy lever: threshold tuning
st.subheader("Decision Threshold (Accuracy Control)")
threshold = st.slider(
    "Lower threshold = catches more failures (higher recall), but may increase false alarms.",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.01
)

# Input sanity warnings (helps “accurate readings” in practice)
with st.expander("Input sanity checks (recommended)"):
    if air_temp < 250 or air_temp > 350:
        st.warning("Air temperature looks out of typical range (250–350 K).")
    if process_temp < 250 or process_temp > 400:
        st.warning("Process temperature looks out of typical range (250–400 K).")
    if rpm < 0 or rpm > 5000:
        st.warning("RPM looks out of typical range (0–5000).")
    if torque < 0 or torque > 200:
        st.warning("Torque looks out of typical range (0–200 Nm).")
    if tool_wear < 0 or tool_wear > 300:
        st.warning("Tool wear looks out of typical range (0–300 min).")

# =========================
# Predict
# =========================
if st.button("Predict", type="primary"):
    input_df = pd.DataFrame({
        # Use dataset-like names so engineered features work consistently
        "Type": [type_input],
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rpm],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
    })

    X_scaled = preprocess_input(input_df)
    p_fail = predict_failure_probability(X_scaled)

    pred_label = 1 if p_fail >= threshold else 0

    st.subheader("Prediction Result")
    st.metric("Failure Probability", f"{p_fail*100:.2f}%")

    if pred_label == 1:
        st.error("⚠️ Predicted: FAILURE (above threshold)")
    else:
        st.success("✅ Predicted: NO FAILURE (below threshold)")

    st.caption(
        "Tip: If your app keeps missing failures, LOWER the threshold (e.g., 0.35 → 0.25). "
        "If it gives too many false alarms, RAISE it."
    )

# =========================
# Optional: Evaluate on dataset to tune threshold
# =========================
st.divider()
st.subheader("Model Evaluation (Optional but best for accuracy)")

st.markdown(
    "If [predictive_maintenance.csv](https://stusouthtexascollege-my.sharepoint.com/personal/bbeltr16_stu_southtexascollege_edu/_layouts/15/Doc.aspx?sourcedoc=%7B0F420288-3B04-462C-8BBB-3EE973D938E2%7D&file=predictive_maintenance.csv&action=default&mobileredirect=true&DefaultItemOpen=1&EntityRepresentationId=28441eff-6a6b-4bab-a668-6575b16eaa3d) is in your repo, you can evaluate "
    "performance and see how threshold affects detection."
)

if st.checkbox("Run evaluation on predictive_maintenance.csv (cached)"):
    @st.cache_data
    def load_eval_data():
        df = pd.read_csv("predictive_maintenance.csv")
        return df

    try:
        eval_df = load_eval_data()

        # Mirror earlier dataset structure: it includes Target and Failure Type
        # We only need features and Target
        y_true = eval_df["Target"].astype(int)

        X_df = eval_df.drop(columns=["Target", "Failure Type"], errors="ignore")
        X_df = X_df.drop(columns=["UDI", "Product ID"], errors="ignore")

        # Ensure engineered features are based on the dataset columns
        X_scaled = preprocess_input(X_df)

        y_prob = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        st.write("Threshold:", threshold)
        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix (rows=true, cols=pred):")
        st.dataframe(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

        report = classification_report(y_true, y_pred, output_dict=True)
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        st.metric("Precision (Failure)", f"{prec:.3f}")
        st.metric("Recall (Failure)", f"{rec:.3f}")
        st.metric("F1 (Failure)", f"{f1:.3f}")

    except FileNotFoundError:
        st.error("Could not find predictive_maintenance.csv in the app directory.")
