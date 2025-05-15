# customer_churn_prediction.py

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# 2. Load Data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 3. Data Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop('customerID', axis=1, inplace=True)

# Encode binary columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Encode other categorical columns
df = pd.get_dummies(df, drop_first=True)

# Scale features
scaler = MinMaxScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 4. Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# 7. Save Model (optional)
import joblib
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')





# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Churn Prediction App")

# Input form
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

gender = st.selectbox("Gender", ["Female", "Male"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
phone = st.selectbox("Has Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Create input dataframe
user_input = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly],
    'TotalCharges': [total],
    'gender_Male': [1 if gender == "Male" else 0],
    'Partner': [1 if partner == "Yes" else 0],
    'Dependents': [1 if dependents == "Yes" else 0],
    'PhoneService': [1 if phone == "Yes" else 0],
    'InternetService_Fiber optic': [1 if internet == "Fiber optic" else 0],
    'InternetService_No': [1 if internet == "No" else 0],
    # Add other necessary dummy fields set to 0
}, index=[0])

# Add missing columns with 0
for col in model.get_booster().feature_names:
    if col not in user_input.columns:
        user_input[col] = 0

# Align column order
user_input = user_input[model.get_booster().feature_names]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]
    st.success(f"Prediction: {'Churn' if prediction else 'No Churn'} ({prob*100:.2f}% probability)")

