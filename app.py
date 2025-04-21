import streamlit as st
import pickle
import numpy as np


# Load model
with open("heart_disease_best_model.saved", "rb") as file:
    model = pickle.load(file)

# Page Title
st.title("Heart Disease Risk Predictor")

st.markdown("### Input your health information below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
chest_pain = st.selectbox("Chest Pain Type", options=["TA (Typical Angina)", "ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
fasting_bs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")
resting_ecg = st.selectbox("Resting ECG Result", options=["Normal", "ST", "LVH"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.checkbox("Exercise-Induced Angina")
st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

# Map categorical inputs to numerical values
cp_map = {"TA (Typical Angina)": 0, "ATA (Atypical Angina)": 1, "NAP (Non-Anginal Pain)": 2, "ASY (Asymptomatic)": 3}
ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
slope_map = {"Up": 0, "Flat": 1, "Down": 2}

input_data = np.array([[
    age,
    cp_map[chest_pain],
    resting_bp,
    int(fasting_bs),
    ecg_map[resting_ecg],
    max_hr,
    int(exercise_angina),
    slope_map[st_slope]
]])

# Predict button
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error("❗ The person is likely to have heart disease.")
    else:
        st.success("✅ The person is unlikely to have heart disease.")
    
    st.write(f"**Prediction Confidence:** {probability:.2%}")

# Add footer
st.markdown("---")
st.markdown("**Model Used:** Logistic Regression")
st.markdown("**Note:** This model was trained on a simplified dataset and should not be used for medical diagnosis.")
