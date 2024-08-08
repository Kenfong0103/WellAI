import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model_diabetes.joblib")

# Create the Streamlit app
st.title("Diabetes Prediction App")

# Define mappings for categorical variables
glucose_mapping = {
    'Normal': 0,
    'Above Normal': 1,
    'Well Above Normal': 2
}

smoking_status_mapping = {
    'Never smoked': 0,
    'Formerly smoked': 1,
    'Smokes': 2
}

binary_mapping = {
    'Yes': 1,
    'No': 0
}

# Define the input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 10, 70, 0)
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
smoking_status = st.selectbox('Smoking Status', list(smoking_status_mapping.keys()))
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0)
HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, value=0.0)
blood_glucose_level = st.selectbox('Blood Glucose Level Category', list(glucose_mapping.keys()))

# Preprocess the input data
def preprocess_input(gender, age, hypertension, heart_disease, smoking_status, bmi, HbA1c_level, blood_glucose_level):
    # Encode categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    hypertension_encoded = binary_mapping[hypertension]
    heart_disease_encoded = binary_mapping[heart_disease]
    smoking_status_encoded = smoking_status_mapping[smoking_status]
    glucose_encoded = glucose_mapping[blood_glucose_level]

    return pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Hypertension': [hypertension_encoded],
        'Heart_disease': [heart_disease_encoded],
        'Smoking_Status': [smoking_status_encoded],
        'BMI': [bmi],
        'HbA1c_level': [HbA1c_level],
        'Glucose': [glucose_encoded]
    })

# Get the preprocessed input data
input_data = preprocess_input(gender, age, hypertension, heart_disease, smoking_status, bmi, HbA1c_level, blood_glucose_level)

# Make predictions
predicted_prob = model.predict_proba(input_data)[:, 1]
predicted_prob_percent = predicted_prob * 100

# Display the result
st.write(f"Predicted probability of diabetes: {predicted_prob_percent[0]:.2f}%")
