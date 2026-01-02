import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("🚢 Titanic Survival Prediction (Naive Bayes)")
st.write("This app uses **Naive Bayes + SelectKBest** to predict survival.")

@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Handle missing values
df['age'] = df['age'].fillna(df['age'].mean())
df['fare'] = df['fare'].fillna(df['fare'].mean())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop unnecessary columns
df = df.drop(['passenger_id', 'name', 'ticket', 'cabin'], axis=1)

# Encode categorical columns
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

X = df.drop('survived', axis=1)
y = df['survived']

# Scaling (required for chi2)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X_scaled, y)

selected_features = X.columns[selector.get_support()]
st.subheader("Selected Best Features")
st.write(list(selected_features))

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.success(f"Model Accuracy: {accuracy:.2f}")

# User input
st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses aboard", 0, 10, 0)
parch = st.number_input("Parents/Children aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, value=10.0)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

sex = 1 if sex == "Male" else 0
embarked_map = {"S": 2, "C": 0, "Q": 1}
embarked = embarked_map[embarked]

input_data = pd.DataFrame(
    [[pclass, sex, age, sibsp, parch, fare, embarked]],
    columns=X.columns
)

input_scaled = scaler.transform(input_data)
input_selected = selector.transform(input_scaled)

if st.button("Predict Survival"):
    prediction = model.predict(input_selected)
    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
