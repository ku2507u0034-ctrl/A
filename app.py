import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn using a trained Random Forest model.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Optional: load model columns if saved
try:
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
except:
    model_columns = None

st.header("Enter Customer Details")

# Example input fields based on common telecom churn dataset
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Manual encoding
input_dict = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": SeniorCitizen,
    "Partner": 1 if Partner == "Yes" else 0,
    "Dependents": 1 if Dependents == "Yes" else 0,
    "tenure": tenure,
    "PhoneService": 1 if PhoneService == "Yes" else 0,
    "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[Contract],
    "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[InternetService],
    "PaymentMethod": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }[PaymentMethod]
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"The customer is likely to churn. Probability: {probability:.2f}")
        else:
            st.success(f"The customer is not likely to churn. Probability: {probability:.2f}")
    except Exception as e:
        st.warning("Feature mismatch occurred. Please ensure app inputs match training columns exactly.")
        st.text(str(e))
