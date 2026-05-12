import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

import xgboost as xgb

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance (Improved)", layout="wide")
st.title(" Predictive Maintenance for CNC Machines (Improved)")
st.write("Enhanced model focused on detecting failures more reliably.")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_resource
def train_model(X_train_scaled, y_train):
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    return model
    return pd.read_csv("predictive_maintenance.csv")

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Basic Info
# -----------------------------
st.subheader("Dataset Information")
st.write("Shape:", data.shape)
st.write("Missing values:")
st.write(data.isnull().sum())

# -----------------------------
# Preprocessing
# -----------------------------
data = data.drop(columns=["UDI", "Product ID"], errors="ignore")

#  Feature Engineering (historical behavior proxies)
data["Temp_Diff"] = data["Process temperature [K]"] - data["Air temperature [K]"]
data["Wear_per_Torque"] = data["Tool wear [min]"] / (data["Torque [Nm]"] + 1)

# -----------------------------
# Target & Features
# -----------------------------
y = data["Target"]
X = data.drop(columns=["Target", "Failure Type"], errors="ignore")

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Handle Class Imbalance
# -----------------------------
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# -----------------------------
# Train Model
# -----------------------------
model = xgb.XGBClassifier(
    n_estimators=50,   # ↓ from default 100+
    max_depth=4,       # smaller trees
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

model = train_model(X_train_scaled, y_train)

# -----------------------------
# Predictions (with threshold tuning)
# -----------------------------
y_prob = model.predict_proba(X_test_scaled)[:, 1]

#  Lower threshold to catch more failures
THRESHOLD = 0.35
y_pred = (y_prob > THRESHOLD).astype(int)

# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

st.subheader("Classification Report (IMPORTANT)")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.title("Confusion Matrix (Threshold Adjusted)")
# st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Top Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

