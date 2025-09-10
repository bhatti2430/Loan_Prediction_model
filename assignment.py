import streamlit as st
import pickle
import numpy as np
import pandas as pd


with open("Loan_Prediction_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
scaler = data["scaler"]
encoders = data["encoders"]
features = data["features"]


st.title("Loan Approval Prediction")
st.write("Enter applicant details to predict **Loan Status**")


def user_input():
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    ApplicantIncome = st.number_input("Applicant Income", min_value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
    LoanAmount = st.number_input("Loan Amount", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (days)", min_value=0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

   
    data = {
        "Gender": Gender,
        "Married": Married,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area": Property_Area,
    }

    return pd.DataFrame([data])


input_df = user_input()


for col, le in encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

input_df = input_df.reindex(columns=features)


input_scaled = scaler.transform(input_df)


if st.button("Predict Loan Status"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:  
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
