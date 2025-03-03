import pandas as pd
import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier

# Paths to input and output files
DATA_PATH = "/Users/alexandra/Desktop/france/NOV24_BDS_INT_Accidents/data/preprocessed_data.csv"
OUTPUT_DIR = "/Users/alexandra/Desktop/france/NOV24_BDS_INT_Accidents/output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load preprocessed data
df = pd.read_csv(DATA_PATH)
print("Preprocessed data loaded successfully.")
print(f"Data shape: {df.shape}")

# Binary classification: Create target variable (1 = severe accidents, 0 = non-severe accidents)
df["grav"] = (df["grav"] >= 2).astype(int)

# Balance dataset: Keep 112,000 samples per class
class_0 = df[df["grav"] == 0].sample(n=112000, random_state=42, replace=False)
class_1 = df[df["grav"] == 1].sample(n=112000, random_state=42, replace=False)
df_balanced = pd.concat([class_0, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features (X) and target (y)
X = df_balanced.drop(columns=["grav"])
y = df_balanced["grav"]

# Debugging: Print dataset columns
print("\nColumns in the dataset:")
print(X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for reuse
scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to '{scaler_path}'")

# Define hyperparameter grid for tuning
xgb_params = {
    "n_estimators": [1000, 1500, 2000],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.01, 0.02, 0.05],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "scale_pos_weight": [1]  # No need for imbalance handling since dataset is balanced
}

# XGBoost Model Training with RandomizedSearchCV
xgb = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    param_distributions=xgb_params,
    n_iter=10,
    scoring="f1",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train_scaled, y_train)
best_xgb_model = xgb.best_estimator_
print("\nBest XGBoost model parameters:")
print(xgb.best_params_)

# Model evaluation on test set
y_pred = best_xgb_model.predict(X_test_scaled)
y_pred_proba = best_xgb_model.predict_proba(X_test_scaled)[:, 1]

# Display classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")

# Save the trained model
model_path = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
joblib.dump(best_xgb_model, model_path)
print(f"\nModel saved to '{model_path}'")

import json

# Save model performance metrics
metrics = {
    "roc_auc": roc_auc,
    "classification_report": classification_report(y_test, y_pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Model metrics saved to '{metrics_path}'")
