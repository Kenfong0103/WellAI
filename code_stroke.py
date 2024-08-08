import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model_stroke.joblib")

# Create the Streamlit app
st.title("Stroke Prediction App")

# Define mappings for categorical variables
work_type_mapping = {
    'Private': 0,
    'Self-employed': 1,
    'Govt_job': 2,
    'children': 3,
    'Never_worked': 4
}

residence_type_mapping = {
    'Urban': 0,
    'Rural': 1
}

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
age = st.slider('Age', 0, 100, 0)
hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', list(work_type_mapping.keys()))
residence_type = st.selectbox('Residence Type', list(residence_type_mapping.keys()))
glucose = st.selectbox('Blood Glucose Level Category', list(glucose_mapping.keys()))
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0)
smoking_status = st.selectbox('Smoking Status', list(smoking_status_mapping.keys()))

# Preprocess the input data
def preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, glucose, bmi, smoking_status):
    # Encode categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    hypertension_encoded = binary_mapping[hypertension]
    heart_disease_encoded = binary_mapping[heart_disease]
    ever_married_encoded = binary_mapping[ever_married]
    work_type_encoded = work_type_mapping[work_type]
    residence_type_encoded = residence_type_mapping[residence_type]
    glucose_encoded = glucose_mapping[glucose]
    smoking_status_encoded = smoking_status_mapping[smoking_status]

    # Return a DataFrame with the feature names in the correct order
    return pd.DataFrame({
        'Gender': [gender_encoded],
        'Age': [age],
        'Hypertension': [hypertension_encoded],
        'Heart_disease': [heart_disease_encoded],
        'Ever_Married': [ever_married_encoded],
        'Work_Type': [work_type_encoded],
        'Residence_Type': [residence_type_encoded],
        'Glucose': [glucose_encoded],
        'BMI': [bmi],
        'Smoking_Status': [smoking_status_encoded]
    })

# Get the preprocessed input data
input_data = preprocess_input(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, glucose, bmi, smoking_status)

# Make predictions
predicted_prob = model.predict_proba(input_data)[:, 1]
predicted_prob_percent = predicted_prob * 100

# Display the result
st.write(f"Predicted probability of stroke: {predicted_prob_percent[0]:.2f}%")
