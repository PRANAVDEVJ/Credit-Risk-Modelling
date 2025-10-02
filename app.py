# app.py

import streamlit as st     # Streamlit: used to build interactive web apps for data and ML projects
import pandas as pd        # pandas: for handling and formatting data
import joblib              # joblib: for loading the saved model and encoders

# ---------------------------------
# 1️⃣ Load Model & Encoders
# ---------------------------------
model = joblib.load("extra_trees_credit_model.pkl")
# Load the trained Extra Trees model that was saved earlier using joblib

encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}
# Load all categorical feature encoders (LabelEncoders) used during training
# These encoders convert user inputs (text) into the same numeric format the model expects

# ---------------------------------
# 2️⃣ Streamlit UI
# ---------------------------------
st.title("💳 Credit Risk Prediction App")
# Set the title of the web application

st.write("Enter applicant information below to predict if the credit risk is **Good** or **Bad**.")
# Display a short description explaining what the app does

# Collect input values from the user using Streamlit widgets
age = st.number_input("Age", min_value=18, max_value=80, value=30)
# Numeric input box for applicant's age (between 18–80)

sex = st.selectbox("Sex", ["male", "female"])
# Dropdown menu to select gender

job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)
# Numeric input box for job category (0–3, based on dataset)

housing = st.selectbox("Housing", ["own", "rent", "free"])
# Dropdown menu for type of housing

saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
# Dropdown menu for saving account category

checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
# Dropdown menu for checking account category

credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
# Numeric input box for credit amount (loan size)

duration = st.number_input("Duration (months)", min_value=1, value=12)
# Numeric input box for loan duration in months

# ---------------------------------
# 3️⃣ Encode Input Data
# ---------------------------------
try:
    # Convert all input values into a pandas DataFrame in the same column order as training data
    input_df = pd.DataFrame({
        "Age": [age],
        # Transform categorical features using the previously saved encoders
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
        "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
        "Credit amount": [credit_amount],
        "Duration": [duration]
    })
    # The encoders ensure consistent mapping between the input text values and numeric labels
except Exception as e:
    # If any encoder or input value fails to transform, show an error message
    st.error(f"⚠️ Encoding error: {e}")
    st.stop()  # Stop the app execution safely

# ---------------------------------
# 4️⃣ Predict
# ---------------------------------
if st.button("Predict Credit Risk"):
    # When the user clicks the "Predict Credit Risk" button:
    pred = model.predict(input_df)[0]
    # Use the trained model to predict based on the encoded input

    target_encoder = joblib.load("target_encoder.pkl")
    # Load the LabelEncoder used for the target (Risk) column to decode numeric prediction

    risk_label = target_encoder.inverse_transform([pred])[0]
    # Convert the numeric prediction (0 or 1) back to the original label ("good" or "bad")

    # Display the prediction result in the UI
    if risk_label.lower() == "good":
        st.success("✅ The predicted credit risk is: **GOOD**")
        # Show a green success box if model predicts "good" credit risk
    else:
        st.error("⚠️ The predicted credit risk is: **BAD**")
        # Show a red warning box if model predicts "bad" credit risk
