# 1 is Good(Lower Risk), 0 is Bad(Higher risk)
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

base_dir = Path(__file__).resolve().parent
notebooks_dir = base_dir / "Notebooks"

model = joblib.load(notebooks_dir / "xgboost_credit_model.pkl")
feature_columns = list(getattr(model, "feature_names_in_", [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
]))

encoders = {}
for column in feature_columns:
    if column in {"Age", "Credit amount", "Duration"}:
        continue
    encoder_path = notebooks_dir / f"{column}_encoder.pkl"
    if encoder_path.exists():
        encoders[column] = joblib.load(encoder_path)


def encode_value(column_name, value):
    encoder = encoders[column_name]
    return int(encoder.transform([value])[0])


st.title("Credit risk Prediction App")
st.write("Enter the applicant information to predict if the credit risk is good or bad.")

# Raw categorical values for the user, encoded internally before prediction
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", list(encoders["Sex"].classes_))
job = st.selectbox("Job", list(encoders["Job"].classes_))
housing = st.selectbox("Housing", list(encoders["Housing"].classes_))
saving_accounts = st.selectbox("Saving accounts", list(encoders["Saving accounts"].classes_))
checking_account = st.selectbox("Checking account", list(encoders["Checking account"].classes_))
credit_amount = st.number_input("Credit Amount", min_value=0, value=100)
duration = st.number_input("Duration (months)", min_value=1, value=12)

user_inputs = {
    "Age": age,
    "Sex": sex,
    "Job": job,
    "Housing": housing,
    "Saving accounts": saving_accounts,
    "Checking account": checking_account,
    "Credit amount": credit_amount,
    "Duration": duration,
}

input_df = pd.DataFrame([
    {
        column: encode_value(column, user_inputs[column]) if column in encoders else user_inputs[column]
        for column in feature_columns
    }
])

if st.button("Predict Risk"):
    prediction = int(model.predict(input_df)[0])

    if prediction == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")