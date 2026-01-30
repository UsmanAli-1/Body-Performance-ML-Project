import streamlit as st
import joblib
import pandas as pd

# Load saved model & scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# streamlit UI 

st.set_page_config(
    page_title="Fitness Checker",
    page_icon="🏋️",
    layout="centered"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center; margin-top: 0;">
        <h1>Body Fitness Checker 🏋️</h1>
        <p style="font-size:18px; margin-bottom: 0;">
            Predict your <b>fitness category</b> (A, B, C, D) using body performance metrics
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# User inputs



# Case A 
age = st.number_input("Age", min_value=15, max_value=70, value=27)
body_fat = st.number_input("Body Fat %", min_value=5.0, max_value=50.0, value=21.3)
diastolic = st.number_input("Diastolic (mmHg)", min_value=50, max_value=120, value=80)
systolic = st.number_input("Systolic (mmHg)", min_value=90, max_value=200, value=130)
grip = st.number_input("Grip Force (kg)", min_value=10, max_value=70, value=54)
flex = st.number_input("Sit & Bend Forward (cm)", min_value=-5, max_value=30, value=18)
situps = st.number_input("Sit-ups Count", min_value=0, max_value=80, value=60)
jump = st.number_input("Broad Jump (cm)", min_value=100, max_value=300, value=217)
bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=26.8)
gender = st.radio("Gender", ["Male", "Female"], index=0)  # male


# Case C 
# age = st.number_input("Age", min_value=15, max_value=70, value=22)
# body_fat = st.number_input("Body Fat %", min_value=5.0, max_value=50.0, value=15.0)
# diastolic = st.number_input("Diastolic (mmHg)", min_value=50, max_value=120, value=70)
# systolic = st.number_input("Systolic (mmHg)", min_value=90, max_value=200, value=110)
# grip = st.number_input("Grip Force (kg)", min_value=10, max_value=70, value=50)
# flex = st.number_input("Sit & Bend Forward (cm)", min_value=-5, max_value=30, value=10)
# situps = st.number_input("Sit-ups Count", min_value=0, max_value=80, value=50)
# jump = st.number_input("Broad Jump (cm)", min_value=100, max_value=300, value=200)
# bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.0)
# gender = st.radio("Gender", ["Male", "Female"], index=0)  # Male


# Case D 
# age = st.number_input("Age", min_value=15, max_value=70, value=25)
# body_fat = st.number_input("Body Fat %", min_value=5.0, max_value=50.0, value=18.0)
# diastolic = st.number_input("Diastolic (mmHg)", min_value=50, max_value=120, value=70)
# systolic = st.number_input("Systolic (mmHg)", min_value=90, max_value=200, value=120)
# grip = st.number_input("Grip Force (kg)", min_value=10, max_value=70, value=40)
# flex = st.number_input("Sit & Bend Forward (cm)", min_value=-5, max_value=30, value=5)
# situps = st.number_input("Sit-ups Count", min_value=0, max_value=80, value=40)
# jump = st.number_input("Broad Jump (cm)", min_value=100, max_value=300, value=180)
# bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=22.0)
# gender = st.radio("Gender", ["Male", "Female"])




gender_M = 1 if gender == "Male" else 0

# Prepare input as DataFrame
new_data = pd.DataFrame([{
    "age": age,
    "body fat_%": body_fat,
    "diastolic": diastolic,
    "systolic": systolic,
    "gripForce": grip,
    "sit and bend forward_cm": flex,
    "sit-ups counts": situps,
    "broad jump_cm": jump,
    "BMI": bmi,
    "gender_M": gender_M
}])

# Predict button
if st.button("Predict Fitness Class"):
    # Scale input
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    prediction = model.predict(new_data_scaled)

    # Map to labels
    class_map = {0: "A (Elite)", 1: "B (Good)", 2: "C (Average)", 3: "D (Below Average)"}
    result = class_map[prediction[0]]

    st.success(f"🏆 Your Predicted Fitness Class: **{result}**")
