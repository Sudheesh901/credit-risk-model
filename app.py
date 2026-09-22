# 1 is Good(Lower Risk), 0 is Bad(Higher risk)
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgboost_credit_model.pkl")
columns=["Sex","Housing","Saving accounts","Checking account"]
encoder = {col:joblib.load(f"{col}_encoder.pkl") for col in columns}

st.title("Credit risk Prediction App")
st.write("Enter the appicants information to predict if the credit risk is good or bad")

age=st.number_input("Age",min_value=18,max_value=80,value=30)
sex=st.selectbox("Sex",["male","female"])
job=st.number_input("Job (0-3)",min_value=0,max_value=3,value=0)
housing=st.selctbox("Housing",["own","rent","free"])
saving_accounts=st.selectbox("Saving accounts",["little","moderate","rich","quite rich"])
checking_accounts=st.selctbox("Checking accounts"),["little","moderate","rich"]
credit_amount=st.number_input("Credit Amount",min_value=0,value=100)
duration=st.number_input("Duration(months)",min_value=1,value=12)

input_df=pd.DataFrame(
    {
        "Age":[age],
        "Sex":[encoder["Sex"].transform([sex])[0]],
        "Job":[job],
        "Housing":[encoder["Housing"].transform([housing])[0]],
        "Saving accounts":[encoder["Saving accounts"].transform([saving_accounts])[0]],
        "Checking accounts":[encoder["Checking accounts"].transform([checking_accounts])[0]],
        "Credit amount":[credit_amount],
        "Duration":[duration]
    }
)
if st.button("Predict Risk"):
    predict=model.prdict(input_df)[0]

    if predict == 1:
        st.success("The predicted credit risk is: **GOOD**")