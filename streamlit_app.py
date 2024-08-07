import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import joblib

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Load the trained models
cardio = joblib.load("cardio_model.joblib")
stroke = joblib.load("stroke_model.joblib")
diabetes = joblib.load("diabetes_model.joblib")

# Function to fetch existing user data from Google Sheets
def fetch_existing_data():
    existing_data = conn.read(worksheet="WellAI", ttl=5)
    existing_data = existing_data.dropna(how="all")
    return existing_data

# Define mappings for categorical variables
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Hypertension': {'Yes': 1, 'No': 0},
    'Heart Disease': {'Yes': 1, 'No': 0},
    'Ever Married': {'Yes': 1, 'No': 0},
    'Work Type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3, 'Never_worked': 4},
    'Residence Type': {'Urban': 1, 'Rural': 0},
    'Smoking Status': {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3},
    'Cholesterol': {'Normal': 0, 'Above normal': 1, 'Well above normal': 2},
    'Glucose': {'Normal': 0, 'Above normal': 1, 'Well above normal': 2},
    'Alcohol Intake': {'Yes': 1, 'No': 0},
    'Physical Activity': {'Yes': 1, 'No': 0}
}

# Function to encode categorical values
def encode(value, mapping):
    return mapping.get(value, None)

# Function to make predictions
def predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, bmi, smoking_status, HbA1c_level, cholesterol, glucose, alcohol_intake, physical_activity):
    # Preprocess the input data
    input_data = {
        'Gender': encode(gender, mappings['Gender']),
        'Age': age,
        'Hypertension': encode(hypertension, mappings['Hypertension']),
        'Heart_disease': encode(heart_disease, mappings['Heart Disease']),
        'Ever_Married': encode(ever_married, mappings['Ever Married']),
        'Work_Type': encode(work_type, mappings['Work Type']),
        'Residence_Type': encode(residence_type, mappings['Residence Type']),
        'BMI': bmi,
        'Smoking_Status': encode(smoking_status, mappings['Smoking Status']),
        'HbA1c_level': HbA1c_level,
        'Cholesterol': encode(cholesterol, mappings['Cholesterol']),
        'Glucose': encode(glucose, mappings['Glucose']),
        'Alcohol Intake': encode(alcohol_intake, mappings['Alcohol Intake']),
        'Physical Activity': encode(physical_activity, mappings['Physical Activity'])
    }
    
    # Make predictions
    cardio_pred = cardio.predict(pd.DataFrame([input_data]))[0] * 100
    stroke_pred = stroke.predict(pd.DataFrame([input_data]))[0] * 100
    diabetes_pred = diabetes.predict(pd.DataFrame([input_data]))[0] * 100

    predictions = {
        'Cardio': cardio_pred,
        'Stroke': stroke_pred,
        'Diabetes': diabetes_pred
    }

    return predictions

# Main function
def main():
    st.title("Health Prediction App")
    st.markdown("Enter your details below.")

    # Fetch existing user data
    existing_data = fetch_existing_data()

    # Onboarding New User Form
    with st.form(key="user_form"):
        # User details inputs
        name = st.text_input(label="Your Name*")
        address = st.text_area(label="Your Address")
        contact_number = st.text_input(label="Contact Number (Example: 010-1234567)*", max_chars=12)
        contact_number = contact_number.zfill(11)

        # Health prediction inputs
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.slider('Age', 0, 100, 0)
        hypertension = st.selectbox('Hypertension', ['Yes', 'No'])
        heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'])
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
        work_type = st.selectbox('Work Type', list(mappings['Work Type'].keys()))
        residence_type = st.selectbox('Residence Type', list(mappings['Residence Type'].keys()))
        smoking_status = st.selectbox('Smoking Status', list(mappings['Smoking Status'].keys()))
        cholesterol = st.selectbox('Cholesterol', list(mappings['Cholesterol'].keys()))
        glucose = st.selectbox('Glucose', list(mappings['Glucose'].keys()))
        alcohol_intake = st.selectbox('Alcohol Intake', ['Yes', 'No'])
        physical_activity = st.selectbox('Physical Activity', ['Yes', 'No'])
        bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0)
        HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, max_value=20.0, value=0.0)

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        if not name or not contact_number:
            st.warning("Ensure all mandatory fields are filled.")
            st.stop()
        else:
            predictions = predict(gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, bmi, smoking_status, HbA1c_level, cholesterol, glucose, alcohol_intake, physical_activity)

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
                "Glucose": [glucose],
                "AlcoholIntake": [alcohol_intake],
                "PhysicalActivity": [physical_activity]
            })

            for model, prob in predictions.items():
                new_user_data[model + '_Probability'] = prob

            updated_data = existing_data.append(new_user_data, ignore_index=True)

            conn.update(worksheet="WellAI", data=updated_data)

            st.subheader("Prediction Results:")
            for model, prob in predictions.items():
                st.write(f"{model} Probability: {prob:.2f}%")

            st.success("Your details and predictions have been successfully submitted!")

if __name__ == "__main__":
    main()
