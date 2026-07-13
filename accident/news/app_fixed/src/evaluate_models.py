import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
GRAPHS_DIR = os.path.join(PROJECT_ROOT, "output", "graphs")

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv(DATASET_PATH)

X = df.drop("Accident", axis=1)
y = df["Accident"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# Load Models
# =====================================

rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

results = {}

os.makedirs(GRAPHS_DIR, exist_ok=True)

# =====================================
# Evaluate Each Model
# =====================================

for name, model in models.items():

    print("=" * 60)
    print(f"Evaluating {name}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # ROC AUC needs both classes present in the test split
    if y_test.nunique() > 1:
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float("nan")

    results[name] = accuracy

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title(f"{name} Confusion Matrix")

    safe_name = name.replace(" ", "_")
    plt.savefig(os.path.join(GRAPHS_DIR, f"{safe_name}_confusion_matrix.png"))

    plt.close()

# =====================================
# Select Best Model
# =====================================

best_model_name = max(results, key=results.get)

if best_model_name == "Random Forest":
    best_model = rf_model
else:
    best_model = xgb_model

joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))

print("\n")
print("=" * 60)
print("Best Model :", best_model_name)
print("Saved as models/best_model.pkl")
print("=" * 60)
