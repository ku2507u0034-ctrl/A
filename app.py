import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Travel Churn Prediction", layout="centered")

st.title("Customer Travel Churn Prediction")
st.write("Predict whether a customer will churn based on travel behavior.")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.header("Enter Customer Details")

# -------- INPUTS -------- #
Age = st.slider("Age", 18, 100, 30)

FrequentFlyer = st.selectbox("Frequent Flyer", ["Yes", "No"])
AnnualIncomeClass = st.selectbox("Income Class", ["Low Income", "Middle Income", "High Income"])
ServicesOpted = st.slider("Services Opted", 0, 6, 2)
AccountSyncedToSocialMedia = st.selectbox("Account Synced to Social Media", ["Yes", "No"])
BookedHotelOrNot = st.selectbox("Booked Hotel", ["Yes", "No"])

# -------- ENCODING (IMPORTANT) -------- #
try:
    input_dict = {
        "Age": Age,
        "FrequentFlyer_Encoded": encoders['FrequentFlyer'].transform([FrequentFlyer])[0],
        "AnnualIncomeClass_Encoded": encoders['AnnualIncomeClass'].transform([AnnualIncomeClass])[0],
        "ServicesOpted": ServicesOpted,
        "AccountSyncedToSocialMedia_Encoded": encoders['AccountSyncedToSocialMedia'].transform([AccountSyncedToSocialMedia])[0],
        "BookedHotelOrNot_Encoded": encoders['BookedHotelOrNot'].transform([BookedHotelOrNot])[0],
    }

    input_df = pd.DataFrame([input_dict])

except Exception as e:
    st.error("Encoding error. Make sure input values match training categories.")
    st.stop()

# -------- PREDICTION -------- #
if st.button("Predict Churn"):

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Customer is likely to churn\n\nProbability: {probability:.2f}")
        else:
            st.success(f"✅ Customer is not likely to churn\n\nProbability: {probability:.2f}")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.code(str(e))
