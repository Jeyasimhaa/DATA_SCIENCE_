import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "accident_data.csv")

# Load Dataset
df = pd.read_csv(DATASET_PATH)

# Encode categorical columns.
# NOTE: these must match the columns that actually exist in
# dataset/accident_data.csv: Weather, Road_Condition, Traffic, Time.
encoders = {}

categorical = [
    "Weather",
    "Road_Condition",
    "Traffic",
    "Time"
]

for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("Accident", axis=1)
y = df["Accident"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

joblib.dump(model, os.path.join(BASE_DIR, "accident_model.pkl"))
joblib.dump(encoders, os.path.join(BASE_DIR, "accident_label_encoders.pkl"))
