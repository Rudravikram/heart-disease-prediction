import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

# Load the saved min and max values
min_values = np.load("min_values.npy")
max_values = np.load("max_values.npy")

# Function to scale real input values using Min-Max scaling
def scale_input(user_input, min_vals, max_vals):
    return [
        (val - minv) / (maxv - minv) if maxv != minv else 0
        for val, minv, maxv in zip(user_input, min_vals, max_vals)
    ]

st.title("Heart Disease Prediction App")
st.markdown("### Please enter the clinical values below:")

# Define input labels and get user input
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
# Accept values from user
for label in input_labels:
    val = st.number_input(label, format="%.6f", value=0.0)
    user_inputs.append(val)

if st.button("Predict"):
    # Apply Min-Max Scaling before prediction
    scaled_input = scale_input(user_inputs, min_values, max_values)
    
    prediction = model.predict([scaled_input])[0]
    
    if prediction == 1:
        st.error("Prediction: High Risk of Heart Disease")
    else:
        st.success("Prediction: Low Risk / No Heart Disease")
