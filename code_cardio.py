import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model_cardio.joblib")

# Create the Streamlit app
st.title("Cardio Prediction App")

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

cholesterol_mapping = {
    'Normal': 0,
    'Above Normal': 1,
    'Well Above Normal': 2
}

binary_mapping = {
    'Yes': 1,
    'No': 0
}

# Define the input fields
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 29, 64, 50)
cholesterol = st.selectbox('Cholesterol', list(cholesterol_mapping.keys()))
blood_glucose_level = st.selectbox('Blood Glucose Level Category', list(glucose_mapping.keys()))
smoking_status = st.selectbox('Smoking Status', list(smoking_status_mapping.keys()))
alcohol_intake = st.selectbox('Alcohol Intake', ['Yes', 'No'])
physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])

# Preprocess the input data
def preprocess_input(gender, age, cholesterol, blood_glucose_level, smoking_status, alcohol_intake, physical_activity):
    # Encode categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    cholesterol_encoded = cholesterol_mapping[cholesterol]
    glucose_encoded = glucose_mapping[blood_glucose_level]
    smoking_status_encoded = smoking_status_mapping[smoking_status]
    alcohol_intake_encoded = binary_mapping[alcohol_intake]
    physical_activity_encoded = binary_mapping[physical_activity]

    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Cholesterol': [cholesterol_encoded],
        'Glucose': [glucose_encoded],
        'Smoking_Status': [smoking_status_encoded],
        'Alcohol Intake': [alcohol_intake_encoded],
        'Physical Activity': [physical_activity_encoded]
    })

# Get the preprocessed input data
input_data = preprocess_input(gender, age, cholesterol, blood_glucose_level, smoking_status, alcohol_intake, physical_activity)

# Make predictions
predicted_prob = model.predict_proba(input_data)[:, 1]
predicted_prob_percent = predicted_prob * 100

# Display the result
st.write(f"Predicted probability of cardio issues: {predicted_prob_percent[0]:.2f}%")
