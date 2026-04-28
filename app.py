import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn using a trained ML model.")

# ------------------ LOAD MODEL ------------------ #
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Try loading training columns
model_columns = None
try:
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
except:
    st.warning("⚠️ model_columns.pkl not found. Using fallback (may cause mismatch).")

# ------------------ INPUT UI ------------------ #
st.header("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
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

# ------------------ DATA PREP ------------------ #
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

# Apply SAME encoding as training
input_df = pd.get_dummies(input_df)

# Align columns with model training
if model_columns is not None:
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

# ------------------ PREDICTION ------------------ #
if st.button("Predict Churn"):

    try:
        prediction = model.predict(input_df)[0]

        # Handle models without predict_proba
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = None

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")

        if probability is not None:
            st.info(f"Confidence: {probability:.2f}")

    except Exception as e:
        st.error("❌ Feature mismatch or model error")
        st.code(str(e))
