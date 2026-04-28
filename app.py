import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn using a trained Random Forest model.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load model columns (VERY IMPORTANT)
try:
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
except:
    model_columns = None
    st.warning("model_columns.pkl not found. Predictions may fail.")

st.header("Enter Customer Details")

# Input fields
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

# Raw input (NO manual encoding)
input_dict = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "PaperlessBilling": PaperlessBilling,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "Contract": Contract,
    "InternetService": InternetService,
    "PaymentMethod": PaymentMethod
}

input_df = pd.DataFrame([input_dict])

# Apply SAME preprocessing as training
input_df = pd.get_dummies(input_df)

# Match training columns exactly
if model_columns is not None:
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to churn.\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Customer is not likely to churn.\n\nProbability: {probability:.2f}")

    except Exception as e:
        st.error("❌ Prediction failed due to feature mismatch.")
        st.text(str(e))
