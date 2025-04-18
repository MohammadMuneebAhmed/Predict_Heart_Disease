import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("model.joblib")

st.title("❤️ Heart Disease Prediction")

st.write("Enter patient details:")

# Collect user input
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST depression", value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)", [0, 1, 2])

# Convert input to model format
input_data = np.array([[
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease!")
