import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..")
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "accident_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed_data.csv")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATASET_PATH)

print("Original Dataset")
print(df.head())

# -----------------------------
# Remove Missing Values
# -----------------------------
df.dropna(inplace=True)

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
df.drop_duplicates(inplace=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoders = {}

categorical_columns = [
    "Weather",
    "Road_Condition",
    "Traffic",
    "Time"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

# -----------------------------
# Save Encoders
# -----------------------------
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop("Accident", axis=1)
y = df["Accident"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Save Processed Dataset
# -----------------------------
df.to_csv(PROCESSED_PATH, index=False)

print("\nProcessed Dataset")
print(df.head())

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

print("\nPreprocessing Completed Successfully!")
