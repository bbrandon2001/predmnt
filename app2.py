import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# -----------------------------
# Page Title
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance for CNC Machines", layout="wide")
st.title("Predictive Maintenance for CNC Machines")
st.write(
    "This app uses machine data to predict whether a failure will occur."
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("predictive_maintenance.csv")
    return data

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Data Info
# -----------------------------
st.subheader("Dataset Information")
st.write("Shape of dataset:", data.shape)
st.write("Missing values by column:")
st.write(data.isnull().sum())

# -----------------------------
# Drop unnecessary columns
# -----------------------------
data = data.drop(columns=["UDI", "Product ID"], errors="ignore")

# -----------------------------
# Define target and features
# -----------------------------
y = data["Target"]
X = data.drop(columns=["Target", "Failure Type"], errors="ignore")

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------
# Scale data
# -----------------------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train model
# -----------------------------
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# Predict and Evaluate
# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.title("Confusion Matrix")
st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance_df.head(10))

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.barh(importance_df["Feature"].head(10), importance_df["Importance"].head(10))
ax2.invert_yaxis()
ax2.set_title("Top 10 Most Important Features")
ax2.set_xlabel("Importance")
ax2.set_ylabel("Feature")
st.pyplot(fig2)

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Try a Prediction")

type_option = st.selectbox("Type", ["L", "M", "H"])
air_temp = st.number_input("Air temperature [K]", value=298.1)
process_temp = st.number_input("Process temperature [K]", value=308.6)
rot_speed = st.number_input("Rotational speed [rpm]", value=1551)
torque = st.number_input("Torque [Nm]", value=42.8)
tool_wear = st.number_input("Tool wear [min]", value=0)

input_df = pd.DataFrame({
    "Type": [type_option],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rot_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear]
})

# Match training columns after get_dummies
input_df = pd.get_dummies(input_df, drop_first=True)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_prob = model.predict_proba(input_scaled)[0][1]

if st.button("Predict Failure"):
    st.write(f"Prediction: {'Failure' if prediction == 1 else 'No Failure'}")
    st.write(f"Probability of Failure: {prediction_prob * 100:.2f}%")
