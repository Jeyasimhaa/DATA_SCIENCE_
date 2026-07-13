import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ====================================
# Load Dataset
# ====================================

df = pd.read_csv(DATASET_PATH)

print("Processed Dataset Loaded Successfully\n")

# ====================================
# Features and Target
# ====================================

X = df.drop("Accident", axis=1)
y = df["Accident"]

# ====================================
# Train-Test Split
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================
# Create XGBoost Model
# ====================================

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# ====================================
# Train Model
# ====================================

print("Training XGBoost Model...\n")

model.fit(X_train, y_train)

print("Training Completed!\n")

# ====================================
# Prediction
# ====================================

y_pred = model.predict(X_test)

# ====================================
# Accuracy
# ====================================

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy*100:.2f}%")

# ====================================
# Confusion Matrix
# ====================================

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# ====================================
# Classification Report
# ====================================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred, zero_division=0))

# ====================================
# Save Model
# ====================================

os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODELS_DIR, "xgboost.pkl"))

print("\nModel Saved Successfully!")

print("\nLocation : models/xgboost.pkl")
