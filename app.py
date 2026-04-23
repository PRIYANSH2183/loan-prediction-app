import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("train.csv")

# Basic preprocessing
df = df.dropna()

# Convert categorical to numeric
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['Married'] = df['Married'].map({'Yes':1, 'No':0})
df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

# Features & target
X = df[['Gender','Married','Education','ApplicantIncome','LoanAmount']]
y = df['Loan_Status']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# UI
st.title("💰 Loan Prediction App")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
income = st.number_input("Applicant Income", value=5000)
loan = st.number_input("Loan Amount", value=100)

# Convert inputs
gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
education = 1 if education=="Graduate" else 0

# Predict
if st.button("Predict"):
    result = model.predict([[gender, married, education, income, loan]])
    
    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")