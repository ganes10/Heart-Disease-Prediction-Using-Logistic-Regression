import streamlit as st
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configure page
st.set_page_config(page_title="Heart Disease Prediction")

# App title with description
st.title("Heart Disease Prediction Using Logistic Regression") 
st.markdown("""
This clinical tool estimates your heart disease risk using validated medical parameters.
**Please provide accurate information for a reliable assessment.**
""")
st.caption("All fields are required")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Personal Information Section
    st.subheader("Personal Information")
    age = st.number_input("Age (years)", min_value=18, max_value=150, step=1, 
                         value=None, placeholder="Enter value 18-150",
                         help="Your current age in whole years (18-150)")
    
    sex = st.selectbox("Sex", ["Male", "Female"], 
                      help="Your biological sex at birth")

    # Heart Health Section
    st.subheader("Heart Health Indicators")
    chest_pain = st.selectbox(
        "Chest Pain Type", 
        ["ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", 
         "ASY (Asymptomatic)", "TA (Typical Angina)"],
        help="""Type of chest discomfort:
        - ATA: Chest pain not caused by heart blockage
        - NAP: Non-heart-related chest discomfort
        - ASY: No chest pain symptoms
        - TA: Classic heart-related chest pain"""
    )
    
    resting_BP = st.number_input("Resting Blood Pressure (mm Hg)", 
                               min_value=50, max_value=250, step=1,
                               value=None, placeholder="Enter value 50-250",
                               help="Your blood pressure measurement at rest (50-250 mm Hg)")

with col2:
    # Blood Work Section
    st.subheader("Blood Test Results")
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 
                                min_value=80, max_value=600, step=1,
                                value=None, placeholder="Enter value 80-600",
                                help="Cholesterol level from blood test (80-600 mg/dl)")
    
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?", 
        ["No", "Yes"],
        help="Blood sugar level after fasting overnight (Indicates potential diabetes)"
    )

    # ECG Section
    st.subheader("ECG Results")
    resting_ECG = st.selectbox(
        "Resting ECG Results", 
        ["Normal (Normal ECG)", "ST (ST-T Wave Abnormality)", 
         "LVH (Left Ventricular Hypertrophy)"],
        help="""Electrocardiogram results:
        - Normal: No abnormalities detected
        - ST: Irregular heart electrical activity
        - LVH: Thickening of heart muscle"""
    )
    
    MaxHR = st.number_input("Maximum Heart Rate Achieved", 
                          min_value=50, max_value=250, step=1,
                          value=None, placeholder="Enter value 50-250",
                          help="Highest heart rate during exercise (50-250 beats per minute)")

# Bottom section for remaining inputs
st.subheader("Exercise Test Results")
ExerciseAngina = st.selectbox(
    "Exercise-Induced Angina", 
    ["No", "Yes"],
    help="Chest pain during physical activity (Indicates possible heart disease)"
)

oldpeak = st.number_input("ST Depression (Oldpeak)", 
                         min_value=0.0, max_value=10.0, step=0.1,
                         value=None, placeholder="Enter value 0.0-10.0",
                         help="ECG change during exercise indicating reduced blood flow (0.0-10.0)")

ST_Slope = st.selectbox(
    "ST Segment Slope", 
    ["Up (Upsloping)", "Flat (Horizontal)", "Down (Downsloping)"],
    help="""Pattern of ECG readings during peak exercise:
    - Up: Normal pattern
    - Flat: Possible heart issue
    - Down: High probability of heart disease"""
)

# Load the trained model with better error handling
@st.cache_resource
def load_model():
    try:
        with open('logistic_regressor1.pkl', 'rb') as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'logistic_regressor1.pkl' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

pipeline = load_model()

# Categorical mappings
categorical_mappings = {
    "Sex": {"Male": "M", "Female": "F"},
    "ChestPainType": {
        "ATA (Atypical Angina)": "ATA",
        "NAP (Non-Anginal Pain)": "NAP",
        "ASY (Asymptomatic)": "ASY",
        "TA (Typical Angina)": "TA"
    },
    "RestingECG": {
        "Normal (Normal ECG)": "Normal",
        "ST (ST-T Wave Abnormality)": "ST",
        "LVH (Left Ventricular Hypertrophy)": "LVH"
    },
    "ExerciseAngina": {"No": "N", "Yes": "Y"},
    "ST_Slope": {
        "Up (Upsloping)": "Up",
        "Flat (Horizontal)": "Flat",
        "Down (Downsloping)": "Down"
    }
}

# Prepare input data
def prepare_input_data():
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [categorical_mappings["Sex"][sex]],
        "ChestPainType": [categorical_mappings["ChestPainType"][chest_pain]],
        "RestingBP": [resting_BP],
        "Cholesterol": [cholesterol],
        "FastingBS": [1 if fasting_bs == "Yes" else 0],
        "RestingECG": [categorical_mappings["RestingECG"][resting_ECG]],
        "MaxHR": [MaxHR],
        "ExerciseAngina": [categorical_mappings["ExerciseAngina"][ExerciseAngina]],
        "Oldpeak": [oldpeak],
        "ST_Slope": [categorical_mappings["ST_Slope"][ST_Slope]]
    })
    return input_data

# Enhanced prediction function
def predict():
    try:
        input_data = prepare_input_data()
        
        # Show raw input data in expander
        with st.expander("Review Your Input Data"):
            st.dataframe(input_data)
        
        # Make prediction with progress
        with st.spinner('Analyzing your heart health...'):
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
            
            prediction = pipeline.predict(input_data)
            probability = pipeline.predict_proba(input_data)[0][1] * 100
            
            # Clear progress bar
            progress_bar.empty()
            
            # Display results with better formatting
            st.write("---")
            if prediction[0] == 1:
                st.error(f"## High Risk Detected ({probability:.1f}%)")
                st.warning("""
                **Clinical Interpretation:**  
                This suggests an elevated probability of heart disease. 
                
                **Recommendations:**
                - Consult a cardiologist
                - Monitor your cardiovascular health
                - Follow up with diagnostic tests
                """)
            else:
                st.success(f"## Low Risk Detected ({probability:.1f}%)")
                st.info("""
                **Clinical Interpretation:**  
                This suggests a lower probability of heart disease.  
                
                **Recommendations:**
                - Maintain heart-healthy habits
                - Get regular checkups
                - Monitor risk factors
                """)
            
            # Add interpretation guide
            with st.expander("Understanding Your Results"):
                st.markdown("""
                **Risk Classification:**
                - **<30%**: Very low risk
                - **30-60%**: Moderate risk
                - **>60%**: High risk
                
                **About This Assessment:**
                - Based on 11 clinical parameters
                - Uses logistic regression algorithm
                - Trained on clinical datasets
                
                *Note: This tool provides risk assessment only and is not a diagnosis.  
                Always consult healthcare professionals for medical advice.*
                """)
                
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.error("Please verify all inputs and try again. Contact support if issues persist.")

# Validate inputs before prediction
def validate_inputs():
    errors = []
    if age is None: errors.append(" Age:Must be between 18-150 years")
    if resting_BP is None: errors.append("Resting BP: Must be between 50-250 mm Hg")
    if cholesterol is None: errors.append("Cholesterol: Must be between 80-600 mg/dl")
    if MaxHR is None: errors.append("Max HR: Must be between 50-250 bpm")
    if oldpeak is None: errors.append("ST Depression: Must be between 0.0-10.0")
    return errors

# Prediction button with validation
if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    validation_errors = validate_inputs()
    if validation_errors:
        st.error("Please correct the following issues:")
        for error in validation_errors:
            st.error(error)
    else:
        predict()

# Footer
st.write("---")
st.markdown("""
<div style="text-align: center;">
    <p>Developed by <strong>Ganesh Basnet (00020111)</strong></p>
    <p><em>For educational purposes only - Not a medical diagnosis</em></p>
</div>
""", unsafe_allow_html=True)