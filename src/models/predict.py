import os
import pandas as pd
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, log_loss
from xgboost import XGBClassifier

# Paths to input and output files
DATA_PATH = "/Users/alexandra/Desktop/france/NOV24_BDS_INT_Accidents/data/preprocessed_data.csv"
OUTPUT_DIR = "/Users/alexandra/Desktop/france/NOV24_BDS_INT_Accidents/output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "encoder.pkl")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models once
print("Loading models...")
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)
print("Models loaded successfully!")

# Load preprocessed data
def load_data():
    data = pd.read_csv(DATA_PATH)
    print(f"Data loaded successfully! Shape: {data.shape}")
    return data

# Preprocessing function
def preprocess_data(data):
    # Remap classes for binary classification (1 or 0)
    data["grav"] = (data["grav"] >= 2).astype(int)  # 0 for 'grav' < 2, 1 for 'grav' >= 2

    # Balance dataset: Keep 112,000 samples per class
    class_0 = data[data["grav"] == 0].sample(n=20000, random_state=42, replace=False)
    class_1 = data[data["grav"] == 1].sample(n=20000, random_state=42, replace=False)

    # Combine the balanced data and shuffle
    df_balanced = pd.concat([class_0, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate features (X) and target (y)
    X = df_balanced.drop(columns=["grav"])
    y = df_balanced["grav"]

    # Handle missing values
    X.fillna(X.median(), inplace=True)

    # Apply scaling
    X_scaled = scaler.transform(X)

    return X_scaled, y

# Main prediction function
def main():
    with threadpool_limits(limits=1, user_api="blas"):  # Prevents CPU overload
        data = load_data()
        X_scaled, y = preprocess_data(data)

        # Make predictions
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]  # Take probability for class 1

        # Evaluate model performance if ground truth is available
        if y is not None:
            print("\nClassification Report:")
            print(classification_report(y, y_pred, zero_division=1))  # Avoids precision errors

            print("\nConfusion Matrix:")
            print(confusion_matrix(y, y_pred))

            # Compute additional metrics
            roc_auc = roc_auc_score(y, y_pred_proba)
            balanced_acc = balanced_accuracy_score(y, y_pred)
            logloss = log_loss(y, y_pred_proba)

            print(f"\nROC AUC Score: {roc_auc:.4f}")
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"Log Loss: {logloss:.4f}")

if __name__ == "__main__":
    main()
