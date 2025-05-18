
import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load("heart_disease_model.pkl")

st.title("Heart Disease Prediction App")

st.markdown("### Please enter the clinical values below:")

input_labels = [
    "Age (years)",
    "Sex (1 = male, 0 = female)",
    "Chest Pain Type (0–3)",
    "Resting Blood Pressure (mm Hg)",
    "Cholesterol (mg/dl)",
    "Fasting Blood Sugar > 120 mg/dl (1 = true, 0 = false)",
    "Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)",
    "Maximum Heart Rate Achieved",
    "Exercise-Induced Angina (1 = yes, 0 = no)",
    "ST Depression (Oldpeak)",
    "Slope of ST Segment (0–2)",
    "Number of Major Vessels (0–3)",
    "Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)"
]


user_inputs = []
for label in input_labels:
    val = st.number_input(label, format="%.6f", value=0.0)
    user_inputs.append(val)

if st.button("Predict"):
    prediction = model.predict([user_inputs])[0]
    if prediction == 1:
        st.error("Prediction: High Risk of Heart Disease")
    else:
        st.success("Prediction: Low Risk / No Heart Disease")
