import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and expected feature names
model = joblib.load("models/attrition_model.pkl")
feature_names = joblib.load("models/features.pkl")

st.title("Employee Attrition Prediction App")

# --- Simple input form ---
st.sidebar.header("Employee Info")

age = st.sidebar.slider("Age", 18, 60, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
job_role = st.sidebar.selectbox("Job Role", [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
    'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
])
department = st.sidebar.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])

# --- Prepare the input data ---
user_data = {
    'Age': age,
    'Gender': gender,
    'MonthlyIncome': monthly_income,
    'JobRole': job_role,
    'Department': department
}
user_df = pd.DataFrame([user_data])

# --- Perform the same preprocessing as during training ---
# One-hot encode categorical features to match training data
user_encoded = pd.get_dummies(user_df)

# Create full input row with all expected features
full_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Fill in only the matching columns
for col in user_encoded.columns:
    if col in full_input.columns:
        full_input[col] = user_encoded[col]

# --- Predict ---
prediction = model.predict(full_input)[0]
probability = model.predict_proba(full_input)[0][1]

# --- Display result ---
st.subheader("Prediction Result")
st.write(f"Prediction: {'Attrition' if prediction == 1 else 'No Attrition'}")
st.write(f"Probability of Attrition: {probability:.2%}")
