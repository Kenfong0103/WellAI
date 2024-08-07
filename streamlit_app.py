import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import joblib
import numpy

# Establishing a Google Sheets connection
conn = st.connection("gsheets", type=GSheetsConnection)

# Load the trained models
cardio = joblib.load("cardio_model.joblib")
diabetes = joblib.load("diabetes_model.joblib")
stroke = joblib.load("stroke_model.joblib")

# Function to fetch existing user data from Google Sheets
def fetch_existing_data():
    existing_data = conn.read(worksheet="WellAI", ttl=5)
    existing_data = existing_data.dropna(how="all")
    return existing_data

# Function to encode categorical values
def code(original_value):
    mapping = {
        'Male': 1,
        'Female': 0,
        'Yes': 1,
        'No': 0,
        'Govt_job': 2,
        'Private': 0,
        'Self-employed': 1,
        'Children': 3,
        'Never_worked': 4,
        'Urban': 1,
        'Rural': 0,
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2,
        'Unknown': 3,
        'Normal': 1,
        'Above normal': 2,
        'Well above normal': 3
    }

    encoded_value = mapping.get(original_value, None)
    return encoded_value

# Function to make predictions
def predict(gender, age, hypertension, heart_disease, ever_married,
            work_type, Residence_type, bmi, smoking_status, HbA1c_level,
            cholesterol, gluc_encoded, alco, active):
    # Check if HbA1c_level is below 3 or above 7
    if HbA1c_level < 3:
        diabetes_pred_prob = 0.0
    elif HbA1c_level >= 6.5:
        diabetes_pred_prob = 1.0
    else:
        # Create DataFrame for input features
        input_diabetes = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_status],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'gluc': [gluc_encoded]
        })

        # Make prediction using diabetes model
        diabetes_pred_prob = diabetes.predict(input_diabetes)[0]

        # Clip the probability to range [0, 1]
        diabetes_pred_prob = max(0.0, min(1.0, diabetes_pred_prob))

    # Make predictions using stroke and cardio models
    input_stroke = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [Residence_type],
        'bmi': [bmi],
        'smoking_status': [smoking_status],
        'gluc': [gluc_encoded]
    })

    input_cardio = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'cholesterol': [cholesterol],
        'gluc': [gluc_encoded],
        'smoke': [smoking_status],
        'alco': [alco],
        'active': [active],
        'hypertension': [hypertension],
        'bmi': [bmi]
    })

    stroke_pred_prob = stroke.predict(input_stroke)
    cardio_pred_prob = cardio.predict(input_cardio)

    if stroke_pred_prob[0] < 0:
        stroke_pred_prob[0] = 0.00
    elif stroke_pred_prob[0] > 100:
        stroke_pred_prob[0] = 1.00

    if cardio_pred_prob[0] < 0:
        cardio_pred_prob[0] = 0.00
    elif cardio_pred_prob[0] > 100:
        cardio_pred_prob[0] = 1.00

    # Format predictions
    predictions = {
        'Stroke': stroke_pred_prob[0] * 100,
        'Cardio': cardio_pred_prob[0] * 100,
        'Diabetes': diabetes_pred_prob * 100
    }

    return predictions

# Main function
def main():
    # Display Title and Description
    st.title("Health Prediction")
    st.markdown("Enter your details below.")

    # Fetch existing user data
    existing_data = fetch_existing_data()

    # Onboarding New User Form
    with st.form(key="user_form"):
        # User details inputs
        name = st.text_input(label="Your Name*")
        address = st.text_area(label="Your Address")
        # Modify the contact number input section
        contact_number = st.text_input(label="Contact Number (Example: 010-1234567)*", max_chars=12)

        # Fill with leading zeros if necessary
        contact_number = contact_number.zfill(11)

        # Health prediction inputs
        gender = st.selectbox('Gender', ('Male', 'Female'))
        age = st.number_input('Age', step=1, min_value=0)
        hypertension = st.selectbox('Hypertension', ('Yes', 'No'))
        heart_disease = st.selectbox('Heart Disease', ('Yes', 'No'))
        ever_married = st.selectbox('Ever Married', ('Yes', 'No'))
        work_type = st.selectbox('Work Type', ('Children', 'Govt_job', 'Never_worked', 'Private', 'Self-employed'))
        Residence_type = st.selectbox('Residence Type', ('Rural', 'Urban'))
        bmi = st.number_input('BMI', step=0.1, value=0.0)
        smoking_status = st.selectbox('Smoking Status', ('never smoked', 'formerly smoked', 'smokes', 'Unknown'))
        HbA1c_level = st.number_input('HbA1c level', step=0.1, value=0.0)
        cholesterol = st.selectbox('Cholesterol', ('Normal', 'Above normal', 'Well above normal'))
        gluc = st.selectbox('Glucose', ('Normal', 'Above normal', 'Well above normal'))
        alco = st.selectbox('Alcohol intake', ('Yes', 'No'))
        active = st.selectbox('Physical activity', ('Yes', 'No'))

        # Mark mandatory fields
        st.markdown("**required*")

        submit_button = st.form_submit_button(label="Submit Details")

        # If the submit button is pressed
        if submit_button:
            # Check if all mandatory fields are filled
            if not name or not contact_number:
                st.warning("Ensure all mandatory fields are filled.")
                st.stop()
            else:
                # Make predictions
                predictions = predict(code(gender), age, code(hypertension), code(heart_disease),
                                      code(ever_married), code(work_type), code(Residence_type), bmi,
                                      code(smoking_status), HbA1c_level, code(cholesterol), code(gluc), code(alco),
                                      code(active))

                # Create a new row of user data
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
                    "ResidenceType": [Residence_type],
                    "BMI": [bmi],
                    "SmokingStatus": [smoking_status],
                    "HbA1cLevel": [HbA1c_level],
                    "Cholesterol": [cholesterol],
                    "Glucose": [gluc],
                    "AlcoholIntake": [alco],
                    "PhysicalActivity": [active]
                })

                # Append the predictions to the user data
                for model, prob in predictions.items():
                    new_user_data[model + '_Probability'] = prob

                # Append the new user data to the existing data
                updated_data = existing_data.append(new_user_data, ignore_index=True)

                # Select only the necessary columns for updating in Google Sheets
                columns_to_update = ['Name', 'Address', 'ContactNumber', 'Gender', 'Age', 'Hypertension', 'HeartDisease',
                                     'EverMarried', 'WorkType', 'ResidenceType', 'BMI', 'SmokingStatus', 'HbA1cLevel',
                                     'Cholesterol', 'Glucose', 'AlcoholIntake', 'PhysicalActivity',
                                     'Stroke_Probability', 'Cardio_Probability', 'Diabetes_Probability']

                # Update Google Sheets with the selected columns
                conn.update(worksheet="WellAI", data=updated_data[columns_to_update])

                # Display the prediction results
                st.subheader("Prediction Results:")
                for model, prob in predictions.items():
                    st.write(f"{model} Probability: {prob:.2f}%")

                # Display success message
                st.success("Your details and predictions have been successfully submitted!")

# Run the main function
if __name__ == "__main__":
    main()
