import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Streamlit app title
st.title("Heart Disease Prediction")

# Input fields for each feature
age = st.number_input("Age", min_value=18, max_value=150, step=1, value=18)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", step=1, value=120)
cholesterol = st.number_input("Serum Cholesterol (mm/dl)", step=1, min_value=80, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar", ["120 or Under", "Over 120"])
resting_ECG = st.selectbox("Resting ECG Results", ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"])
MaxHR = st.number_input("Maximum Heart Rate Achieved", step=1, min_value=50, value=140)
ExerciseAngina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.number_input("ST Depression", value=0.0)
ST_Slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

# Model selection
model_options = ["Logistic Regression"]
selected_model = st.selectbox("Classification Model", model_options)

def convert_categorical_variables(sex_, chest_pain_, fasting_bs_, resting_ECG_, ExerciseAngina_, ST_Slope_):
    sex_conversion = {"Male": "M", "Female": "F"}
    chest_pain_conversion = {"Typical Angina": "TA", "Atypical Angina": "ATA", "Non-Anginal Pain": "NAP", "Asymptomatic": "ASY"}
    fasting_bs_conversion = {"Over 120": 1, "120 or Under": 0}
    resting_ECG_conversion = {"ST-T wave abnormality": "ST", "Normal": "Normal", "Left Ventricular Hypertrophy": "LVH"}
    ExerciseAngina_conversion = {"Yes": "Y", "No": "N"}
    ST_Slope_conversion = {"Up": "Up", "Flat": "Flat", "Down": "Down"}
    return sex_conversion[sex_], chest_pain_conversion[chest_pain_], fasting_bs_conversion[fasting_bs_], resting_ECG_conversion[resting_ECG_], ExerciseAngina_conversion[ExerciseAngina_], ST_Slope_conversion[ST_Slope_]

sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope = convert_categorical_variables(sex, chest_pain, fasting_bs, resting_ECG, ExerciseAngina, ST_Slope)

def preprocess_new_data():
    new_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_BP],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ECG],
        "MaxHR": [MaxHR],
        "ExerciseAngina": [ExerciseAngina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [ST_Slope]
    })
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]),
        ('cat', OneHotEncoder(drop='first'), ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"])
    ])
    
    return preprocessor.fit_transform(new_data)

to_predict = preprocess_new_data()

# Load the trained model
try:
    log_reg_model = pickle.load(open('logistic_regressor1.pkl', 'rb'))
except FileNotFoundError:
    st.error("🚨 Model file not found! Please check the path or retrain the model.")
    st.stop()

def predict():
    prediction = log_reg_model.predict(to_predict)
    probability_positive = log_reg_model.predict_proba(to_predict)[0][1]
    
    if prediction[0] == 1:
        st.success(f"It is predicted that the patient has heart disease.")
    else:
        st.success(f"It is predicted that the patient does not have heart disease.")
    st.success(f"Chance of being heart disease positive: {100*probability_positive:.2f}%")

if st.button("Predict"):
    progress_bar = st.progress(0)
    for i in range(101):
        progress_bar.progress(i)
        time.sleep(0.002)
    predict()

st.subheader("Developed By: Ganesh Basnet (00020111)")
