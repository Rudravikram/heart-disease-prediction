import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaling data
model = joblib.load("heart_disease_model.pkl")
min_vals = np.load("min_values.npy")
max_vals = np.load("max_values.npy")

st.title("Heart Disease Prediction App")
st.markdown("### Please enter the clinical values below:")

# Input labels and default values (example values; adjust as needed)
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

default_values = [55, 1, 0, 130, 250, 0, 1, 150, 0, 1.0, 1, 0, 2]

user_inputs = []
for label, default in zip(input_labels, default_values):
    val = st.number_input(label, value=float(default))
    user_inputs.append(val)

if st.button("Predict"):
    # Normalize inputs using min-max scaling
    normalized_input = [
        (x - min_v) / (max_v - min_v) if max_v != min_v else 0.0
        for x, min_v, max_v in zip(user_inputs, min_vals, max_vals)
    ]

    prediction = model.predict([normalized_input])[0]
    if prediction == 1:
        st.error("🩺 Prediction: High Risk of Heart Disease")
    else:
        st.success("💖 Prediction: Low Risk / No Heart Disease")
