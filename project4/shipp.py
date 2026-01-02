import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction (Naive Bayes)")
st.write("This app uses **Naive Bayes + SelectKBest** to predict passenger survival.")

# -------------------------------------------------
# Load Dataset (Streamlit Cloud Safe)
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/titanic.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# Data Preprocessing
# -------------------------------------------------
# Handle missing values
df["age"] = df["age"].fillna(df["age"].mean())
df["fare"] = df["fare"].fillna(df["fare"].mean())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Drop unnecessary columns
df = df.drop(["passenger_id", "name", "ticket", "cabin"], axis=1)

# Encode categorical features
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["embarked"] = le_embarked.fit_transform(df["embarked"])

# Features and target
X = df.drop("survived", axis=1)
y = df["survived"]

# -------------------------------------------------
# Scaling (Required for chi2)
# -------------------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# Feature Selection
# -------------------------------------------------
selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
st.subheader("⭐ Selected Best Features")
st.write(list(selected_features))

# -------------------------------------------------
# Train Model
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.success(f"🎯 Model Accuracy: {accuracy:.2f}")

# -------------------------------------------------
# User Input
# -------------------------------------------------
st.subheader("🧍 Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings / Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents / Children aboard", 0, 10, 0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

# Encode inputs
sex_encoded = le_sex.transform([sex])[0]
embarked_encoded = le_embarked.transform([embarked])[0]

input_df = pd.DataFrame(
    [[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]],
    columns=X.columns
)

# Scale and select features
input_scaled = scaler.transform(input_df)
input_selected = selector.transform(input_scaled)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🚀 Predict Survival"):
    prediction = model.predict(input_selected)[0]

    if prediction == 1:
        st.success("✅ Passenger **Survived**")
    else:
        st.error("❌ Passenger **Did Not Survive**")
