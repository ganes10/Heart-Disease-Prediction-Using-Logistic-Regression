import streamlit as st
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Streamlit app title
st.title("Heart Disease Prediction")

# Input fields for each feature with validation and instructions
age = st.text_input("Age", placeholder="Enter your age (e.g., 18-150)")
if age:
    try:
        age = int(age)
        if age < 18 or age > 150:
            st.error("Age must be between 18 and 150.")
    except ValueError:
        st.error("Please enter a valid number for age.")

sex = st.selectbox("Sex", ["Select", "Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Select", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_BP = st.text_input("Resting Blood Pressure", placeholder="Enter resting BP (e.g., 50-250 mm Hg)")
if resting_BP:
    try:
        resting_BP = int(resting_BP)
        if resting_BP < 50 or resting_BP > 250:
            st.error("Resting Blood Pressure must be between 50 and 250 mm Hg.")
    except ValueError:
        st.error("Please enter a valid number for Resting Blood Pressure.")

cholesterol = st.text_input("Serum Cholesterol", placeholder="Enter cholesterol level (e.g., 80-600 mg/dl)")
if cholesterol:
    try:
        cholesterol = int(cholesterol)
        if cholesterol < 80 or cholesterol > 600:
            st.error("Cholesterol must be between 80 and 600 mg/dl.")
    except ValueError:
        st.error("Please enter a valid number for Cholesterol.")

fasting_bs = st.selectbox("Fasting Blood Sugar", ["Select", "120 or Under", "Over 120"])
resting_ECG = st.selectbox("Resting ECG Results", ["Select", "Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
MaxHR = st.text_input("Maximum Heart Rate Achieved", placeholder="Enter MaxHR (e.g., 50-250)")
if MaxHR:
    try:
        MaxHR = int(MaxHR)
        if MaxHR < 50 or MaxHR > 250:
            st.error("Maximum Heart Rate must be between 50 and 250.")
    except ValueError:
        st.error("Please enter a valid number for Maximum Heart Rate.")

ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["Select", "No", "Yes"])
oldpeak = st.text_input("ST Depression (Oldpeak)", placeholder="Enter Oldpeak (e.g., 0.0-10.0)")
if oldpeak:
    try:
        oldpeak = float(oldpeak)
        if oldpeak < 0.0 or oldpeak > 10.0:
            st.error("ST Depression (Oldpeak) must be between 0.0 and 10.0.")
    except ValueError:
        st.error("Please enter a valid number for ST Depression (Oldpeak).")
ST_Slope = st.selectbox("ST Segment Slope", ["Select", "Up", "Flat", "Down"])

# Load the trained model
try:
    with open('logistic_regressor1.pkl', 'rb') as model_file:
        pipeline = pickle.load(model_file)  # Load the entire pipeline (preprocessor + model)
except FileNotFoundError:
    st.error("Model file not found! Please check the path or retrain the model.")
    st.stop()

# Prepare the input data
def prepare_input_data():
    return pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_BP],
        "Cholesterol": [cholesterol],
        "FastingBS": [1 if fasting_bs == "Over 120" else 0],
        "RestingECG": [resting_ECG],
        "MaxHR": [MaxHR],
        "ExerciseAngina": [ExerciseAngina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [ST_Slope]
    })

# Predict using the loaded pipeline
def predict():
    try:
        # Prepare the input data
        input_data = prepare_input_data()
        
        # Make predictions
        prediction = pipeline.predict(input_data)
        probability_positive = pipeline.predict_proba(input_data)[0][1]
        
        # Display results
        if prediction[0] == 1:
            st.success(f"It is predicted that the patient has heart disease.")
        else:
            st.success(f"It is predicted that the patient does not have heart disease.")
        st.info(f"Chance of being heart disease positive: {100 * probability_positive:.2f}%")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Add a button to trigger prediction
if st.button("Predict"):
    # Add a progress bar for better user experience
    progress_bar = st.progress(0)
    for i in range(101):
        progress_bar.progress(i)
        time.sleep(0.01)
    predict()

# Footer
st.subheader("Developed By: Ganesh Basnet (00020111)")