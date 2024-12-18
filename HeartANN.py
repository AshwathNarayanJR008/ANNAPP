import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the saved model and scaler
model = load_model('heart_disease_model_tanh_all_layers.h5')
scaler = joblib.load('scaler.pkl')

# App title
st.title("Heart Disease Prediction App")

# User inputs
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
sex = st.radio("Sex", options=["Male", "Female"])
sex = 1 if sex == "Male" else 0

chest_pain_type = st.radio(
    "Chest Pain Type",
    options=["TA (Typical Angina)", "ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)"],
)
chest_pain_type = ["TA", "ATA", "NAP", "ASY"].index(chest_pain_type)

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120, step=1)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=800, value=200, step=1)

fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL?", options=["Yes", "No"])
fasting_bs = 1 if fasting_bs == "Yes" else 0

resting_ecg = st.radio(
    "Resting ECG Results",
    options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
)
resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)

max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150, step=1)

exercise_angina = st.radio("Exercise Induced Angina?", options=["Yes", "No"])
exercise_angina = 1 if exercise_angina == "Yes" else 0

oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

st_slope = st.radio("ST Slope", options=["Upsloping", "Flat", "Downsloping"])
st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

# Collect inputs
input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

# Prediction button
if st.button("Predict"):
    try:
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        prediction = (prediction > 0.5).astype("int32")

        # Display result
        if prediction == 1:
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")
    except Exception as e:
        st.error(f"Error: {str(e)}")