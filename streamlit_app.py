import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import joblib

# Set page layout to wide
st.set_page_config(layout="wide")

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Load the trained models
stroke_model = joblib.load("model_stroke1.joblib")
cardio_model = joblib.load("model_cardio1.joblib")
diabetes_model = joblib.load("model_diabetes1.joblib")

# Function to fetch existing user data from Google Sheets
def fetch_existing_data():
    existing_data = conn.read(worksheet="WellAI", ttl=5)
    existing_data = existing_data.dropna(how="all")
    return existing_data

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

work_type_mapping = {
    'Currently No Working': 0,
    'Currently Working': 1
}

residence_type_mapping = {
    'Urban': 0,
    'Rural': 1
}

binary_mapping = {
    'Yes': 1,
    'No': 0
}

# Fetch existing user data
existing_data = fetch_existing_data()

# Define the input fields using two columns layout
with st.form(key="user_form"):
    col1, col2, col3 = st.columns(3)

    # Column 1 - Left
    with col1:
        name = st.text_input(label="Your Name*")
        address = st.text_area(label="Your Address")
        contact_number = st.text_input(label="Contact Number (Example: 010-1234567)*", max_chars=12)
        
        # Fill with leading zeros if necessary
        contact_number = contact_number.zfill(11)
        
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=0, max_value=100, value=0)  # Changed from slider to number input

    # Column 2 - Right
    with col2:
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
        work_type = st.selectbox('Work Status', list(work_type_mapping.keys()))
        residence_type = st.selectbox('Residence Type', list(residence_type_mapping.keys()))
        smoking_status = st.selectbox('Smoking Status', list(smoking_status_mapping.keys()))
        alcohol_intake = st.selectbox('Alcohol Intake', ['Yes', 'No'])
        physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])

    # Column 3 - Right
    with col3:
        hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
        heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
        blood_glucose_level = st.selectbox('Blood Glucose Level Category', list(glucose_mapping.keys()))
        bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0)
        HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, value=0.0)
        submit_button = st.form_submit_button(label="Submit")

# Preprocess the input data (functions remain the same)
def preprocess_input_for_stroke(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                                blood_glucose_level, bmi, smoking_status):
    gender_encoded = 1 if gender == 'Male' else 0
    hypertension_encoded = binary_mapping[hypertension]
    heart_disease_encoded = binary_mapping[heart_disease]
    ever_married_encoded = binary_mapping[ever_married]
    work_type_encoded = work_type_mapping[work_type]
    residence_type_encoded = residence_type_mapping[residence_type]
    glucose_encoded = glucose_mapping[blood_glucose_level]
    smoking_status_encoded = smoking_status_mapping[smoking_status]

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

def preprocess_input_for_cardio(gender, age, blood_glucose_level, smoking_status, alcohol_intake,
                                physical_activity):
    gender_encoded = 1 if gender == 'Male' else 0
    glucose_encoded = glucose_mapping[blood_glucose_level]
    smoking_status_encoded = smoking_status_mapping[smoking_status]
    alcohol_intake_encoded = binary_mapping[alcohol_intake]
    physical_activity_encoded = binary_mapping[physical_activity]

    return pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Glucose': [glucose_encoded],
        'Smoking_Status': [smoking_status_encoded],
        'Alcohol Intake': [alcohol_intake_encoded],
        'Physical Activity': [physical_activity_encoded]
    })

def preprocess_input_for_diabetes(gender, age, hypertension, heart_disease, smoking_status, bmi, HbA1c_level,
                                  blood_glucose_level):
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
        # Stroke prediction
        stroke_input_data = preprocess_input_for_stroke(gender, age, hypertension, heart_disease, ever_married, work_type,
                                                        residence_type, blood_glucose_level, bmi, smoking_status)
        stroke_prob = make_predictions(stroke_model, stroke_input_data, "stroke")
        
        # Cardio prediction
        cardio_input_data = preprocess_input_for_cardio(gender, age, blood_glucose_level, smoking_status,
                                                        alcohol_intake, physical_activity)
        cardio_prob = make_predictions(cardio_model, cardio_input_data, "cardio")

        # Diabetes prediction
        diabetes_input_data = preprocess_input_for_diabetes(gender, age, hypertension, heart_disease, smoking_status, bmi,
                                                            HbA1c_level, blood_glucose_level)
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
            "Glucose": [blood_glucose_level],
            "AlcoholIntake": [alcohol_intake],
            "PhysicalActivity": [physical_activity],
            "Stroke_Probability": [stroke_prob],
            "Cardio_Probability": [cardio_prob],
            "Diabetes_Probability": [diabetes_prob]
        })

        # Append the new user data to the existing data
        updated_data = existing_data.append(new_user_data, ignore_index=True)

        # Select only the necessary columns for updating in Google Sheets
        columns_to_update = ['Name', 'Address', 'ContactNumber', 'Gender', 'Age', 'Hypertension', 'HeartDisease',
                            'EverMarried', 'WorkType', 'ResidenceType', 'BMI', 'SmokingStatus', 'HbA1cLevel',
                            'Cholesterol', 'Glucose', 'AlcoholIntake', 'PhysicalActivity',
                            'Stroke_Probability', 'Cardio_Probability', 'Diabetes_Probability']
        
        # Update Google Sheets with the new user data
        conn.update(worksheet="WellAI", data=updated_data[columns_to_update])

        # Display success message
        st.success("Your details and predictions have been successfully submitted!")
