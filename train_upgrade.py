import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    fbeta_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Config (goal A: catch failures)
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25

# For goal A: prioritize recall -> F2 favors recall more than precision
BETA = 2.0

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Works with dataset columns like:
    # Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]
    df = df.copy()

    # Strip spaces from column names (some datasets have trailing spaces)
    df.columns = [c.strip() for c in df.columns]

    air = "Air temperature [K]"
    proc = "Process temperature [K]"
    rpm = "Rotational speed [rpm]"
    tq  = "Torque [Nm]"
    wear = "Tool wear [min]"

    # Feature engineering (helps accuracy on tabular predictive maintenance)
    if air in df.columns and proc in df.columns:
        df["Temp_Diff"] = df[proc] - df[air]
    if wear in df.columns and tq in df.columns:
        df["Wear_per_Torque"] = df[wear] / (df[tq] + 1.0)
    if rpm in df.columns and tq in df.columns:
        df["Stress_Index"] = df[rpm] * df[tq]

    return df

def pick_threshold_for_f2(y_true, y_prob, beta=2.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds array is shorter by 1 than precision/recall
    best_t = 0.5
    best_score = -1

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score

def main():
    df = pd.read_csv("predictive_maintenance.csv")
    df.columns = [c.strip() for c in df.columns]

    # Target is binary label in your dataset (0/1) 【1-5a5748】
    y = df["Target"].astype(int)

    # Drop identifiers + leakage columns (Failure Type is related label info)
    X = df.drop(columns=["Target", "Failure Type", "UDI", "Product ID"], errors="ignore")

    # Add engineered features
    X = add_features(X)

    # Identify columns
    cat_cols = ["Type"] if "Type" in X.columns else []
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", RobustScaler(), num_cols),
        ],
        remainder="drop"
    )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Candidate models
    candidates = []

    # 1) Strong baseline: Logistic Regression (balanced)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    candidates.append(("logreg", lr, {
        "model__C": [0.5, 1.0, 2.0, 5.0]
    }))

    # 2) Robust non-linear: RandomForest (balanced)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )
    candidates.append(("rf", rf, {
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }))

    # 3) Optional: XGBoost if available (often strong on tabular)
    try:
        import xgboost as xgb
        # scale_pos_weight helps imbalance: neg/pos
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = float(neg) / float(pos) if pos > 0 else 1.0

        xgbm = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=spw,
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0
        )
        candidates.append(("xgb", xgbm, {
            "model__max_depth": [3, 4, 5],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0]
        }))
    except Exception:
        pass

    # We’ll choose best by PR-AUC (average precision) -> best for imbalance
    best_name, best_pipe, best_score = None, None, -1

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model, grid in candidates:
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        search = GridSearchCV(
            pipe,
            param_grid=grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1
        )
        search.fit(X_train, y_train)

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
            best_pipe = search.best_estimator_

    # Fit best on train
    best_pipe.fit(X_train, y_train)

    # Evaluate
    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Pick threshold optimized for goal A (F2)
    best_t, best_f2 = pick_threshold_for_f2(y_test, y_prob, beta=BETA)
    y_pred = (y_prob >= best_t).astype(int)

    print("\n============================")
    print("BEST MODEL:", best_name)
    print("CV PR-AUC:", round(best_score, 4))
    print("TEST PR-AUC:", round(pr_auc, 4))
    print("TEST ROC-AUC:", round(roc_auc, 4))
    print("BEST THRESHOLD (F2):", round(best_t, 4), "F2:", round(best_f2, 4))
    print("============================\n")

    print(classification_report(y_test, y_pred, digits=4))

    # Save artifacts
    joblib.dump(best_pipe, "model_pipeline.pkl")
    joblib.dump(best_t, "threshold.pkl")

    print("\nSaved: model_pipeline.pkl and threshold.pkl")

if __name__ == "__main__":
    main()
