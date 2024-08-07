import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import joblib

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Load the trained models
cardio_model = joblib.load("cardio_model.joblib")
stroke_model = joblib.load("stroke_model.joblib")
diabetes_model = joblib.load("diabetes_model.joblib")

# Create the Streamlit app
st.title("Health Prediction App")

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

binary_mapping = {
    'Yes': 1,
    'No': 0
}

# Define the input fields
with st.form(key="user_form"):
    # New fields
    name = st.text_input(label="Your Name*")
    address = st.text_area(label="Your Address")
    contact_number = st.text_input(label="Contact Number (Example: 010-1234567)*", max_chars=12)

    # Fill with leading zeros if necessary
    contact_number = contact_number.zfill(11)

    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 0, 100, 0)
    hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
    heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    work_type = st.selectbox('Work Type', list(work_type_mapping.keys()))
    residence_type = st.selectbox('Residence Type', list(residence_type_mapping.keys()))
    blood_glucose_level = st.selectbox('Blood Glucose Level Category', list(glucose_mapping.keys()))
    cholesterol = st.selectbox('Cholesterol', list(cholesterol_mapping.keys()))
    smoking_status = st.selectbox('Smoking Status', list(smoking_status_mapping.keys()))
    alcohol_intake = st.selectbox('Alcohol Intake', ['Yes', 'No'])
    physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])
    bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0)
    HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, value=0.0)

    submit_button = st.form_submit_button(label="Submit")

# Preprocess the input data
def preprocess_input_for_cardio(gender, age, cholesterol, blood_glucose_level, smoking_status, alcohol_intake, physical_activity):
    gender_encoded = 1 if gender == 'Male' else 0
    cholesterol_encoded = cholesterol_mapping[cholesterol]
    glucose_encoded = glucose_mapping[blood_glucose_level]
    smoking_status_encoded = smoking_status_mapping[smoking_status]
    alcohol_intake_encoded = binary_mapping[alcohol_intake]
    physical_activity_encoded = binary_mapping[physical_activity]

    return pd.DataFrame({
        'age': [age],
        'gender': [gender_encoded],
        'cholesterol': [cholesterol_encoded],
        'gluc': [glucose_encoded],
        'smoke': [smoking_status_encoded],
        'alco': [alcohol_intake_encoded],
        'active': [physical_activity_encoded]
    })

def preprocess_input_for_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, blood_glucose_level, bmi, smoking_status):
    gender_encoded = 1 if gender == 'Male' else 0
    hypertension_encoded = binary_mapping[hypertension]
    heart_disease_encoded = binary_mapping[heart_disease]
    ever_married_encoded = binary_mapping[ever_married]
    work_type_encoded = work_type_mapping[work_type]
    residence_type_encoded = residence_type_mapping[residence_type]
    glucose_encoded = glucose_mapping[blood_glucose_level]
    smoking_status_encoded = smoking_status_mapping[smoking_status]

    return pd.DataFrame({
        'gender': [gender_encoded],
        'age': [age],
        'hypertension': [hypertension_encoded],
        'heart_disease': [heart_disease_encoded],
        'ever_married': [ever_married_encoded],
        'work_type': [work_type_encoded],
        'Residence_type': [residence_type_encoded],
        'gluc': [glucose_encoded],
        'bmi': [bmi],
        'smoking_status': [smoking_status_encoded]
    })

def preprocess_input_for_diabetes(gender, age, hypertension, heart_disease, smoking_status, bmi, HbA1c_level, blood_glucose_level):
    gender_encoded = 1 if gender == 'Male' else 0
    hypertension_encoded = binary_mapping[hypertension]
    heart_disease_encoded = binary_mapping[heart_disease]
    smoking_status_encoded = smoking_status_mapping[smoking_status]
    glucose_encoded = glucose_mapping[blood_glucose_level]

    return pd.DataFrame({
        'gender': [gender_encoded],
        'age': [age],
        'hypertension': [hypertension_encoded],
        'heart_disease': [heart_disease_encoded],
        'smoking_history': [smoking_status_encoded],
        'bmi': [bmi],
        'HbA1c_level': [HbA1c_level],
        'gluc': [glucose_encoded]
    })

# Prediction function
def make_predictions(model, input_data, condition_name):
    predicted_prob = model.predict_proba(input_data)[:, 1]
    predicted_prob_percent = predicted_prob * 100
    st.write(f"Predicted probability of {condition_name}: {predicted_prob_percent[0]:.2f}%")
    return predicted_prob_percent[0]

# If the submit button is pressed
if submit_button:
    # Check if all mandatory fields are filled
    if not name or not contact_number:
        st.warning("Ensure all mandatory fields are filled.")
        st.stop()
    else:
        # Cardio prediction
        cardio_input_data = preprocess_input_for_cardio(gender, age, cholesterol, blood_glucose_level, smoking_status, alcohol_intake, physical_activity)
        cardio_prob = make_predictions(cardio_model, cardio_input_data, "cardio")

        # Stroke prediction
        stroke_input_data = preprocess_input_for_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, blood_glucose_level, bmi, smoking_status)
        stroke_prob = make_predictions(stroke_model, stroke_input_data, "stroke")

        # Diabetes prediction
        diabetes_input_data = preprocess_input_for_diabetes(gender, age, hypertension, heart_disease, smoking_status, bmi, HbA1c_level, blood_glucose_level)
        diabetes_prob = make_predictions(diabetes_model, diabetes_input_data, "diabetes")

        # Append the predictions to the user data
        new_user_data = pd.DataFrame({
            "Name": [name],
            "Address": [address],
            "ContactNumber": [contact_number],
            "Gender": [gender],
            "Age": [age],
            "Hypertension": [hypertension],
            "HeartDisease": [heart_disease],
            "EverMarried": [ever_married],
            "WorkType": [work_type],
            "ResidenceType": [residence_type],
            "BMI": [bmi],
            "SmokingStatus": [smoking_status],
            "HbA1cLevel": [HbA1c_level],
            "Cholesterol": [cholesterol],
            "Glucose": [blood_glucose_level],
            "AlcoholIntake": [alcohol_intake],
            "PhysicalActivity": [physical_activity],
            "Cardio_Probability": [cardio_prob],
            "Stroke_Probability": [stroke_prob],
            "Diabetes_Probability": [diabetes_prob]
        })

        # Read existing data from the Google Sheets
        existing_data = conn.read(worksheet="WellAI")

        st.write("Existing data:")
        st.write(existing_data)

        if existing_data.empty:
            # If the sheet is empty, initialize with headers
            st.write("Sheet is empty. Initializing with headers.")
            existing_data = pd.DataFrame(columns=new_user_data.columns)
        
        # Append the new user data to the existing data
        updated_data = pd.concat([existing_data, new_user_data], ignore_index=True)

        st.write("Updated data:")
        st.write(updated_data)

        # Update Google Sheets with the combined data
        conn.update(worksheet="WellAI", data=updated_data)

        st.success("Your details and predictions have been successfully submitted!")
