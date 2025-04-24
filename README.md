# Heart Disease Prediction System

## Overview
This project is a web-based application for predicting the likelihood of heart disease in individuals. It uses a pre-trained logistic regression model combined with a user-friendly interface built with Streamlit. The system allows users to input various health metrics and receive predictions in real-time, assisting in early diagnosis and prevention.

## Features
- Real-time heart disease prediction
- User-friendly web interface built with Streamlit
- Input validation to ensure accurate data entry
- Pre-trained logistic regression model for predictions
- Probability score for the likelihood of heart disease

## Requirements
- Python 3.x
- Libraries:
    - streamlit
    - pandas
    - numpy
    - scikit-learn

## Installation
1. Clone the repository:
     ```bash
     git clone https://github.com/ganes10/Heart-Disease-Prediction-System.git
     ```
2. Navigate to the project directory:
     ```bash
     cd Heart-Disease-Prediction-System
     ```
3. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

## Usage
1. Ensure the pre-trained model file (`logistic_regressor1.pkl`) is in the project directory.
2. Run the Streamlit application:
     ```bash
     streamlit run app.py
     ```
3. Open the application in your browser (usually at `http://localhost:8501`).
4. Enter the required health metrics (e.g., age, cholesterol, blood pressure) and receive predictions.

## Input Features
The system requires the following inputs:
- **Age**: Enter a value between 18 and 150.
- **Sex**: Select either "Male" or "Female."
- **Chest Pain Type**: Choose from "Typical Angina," "Atypical Angina," "Non-Anginal Pain," or "Asymptomatic."
- **Resting Blood Pressure**: Enter a value between 50 and 250 mm Hg.
- **Serum Cholesterol**: Enter a value between 80 and 600 mg/dl.
- **Fasting Blood Sugar**: Select "120 or Under" or "Over 120."
- **Resting ECG Results**: Choose from "Normal," "ST-T wave abnormality," or "Left Ventricular Hypertrophy."
- **Maximum Heart Rate Achieved**: Enter a value between 50 and 250.
- **Exercise-Induced Angina**: Select "Yes" or "No."
- **ST Depression (Oldpeak)**: Enter a value between 0.0 and 10.0.
- **ST Segment Slope**: Choose from "Up," "Flat," or "Down."

## Dataset
The model was trained on a publicly available heart disease dataset, which includes features such as age, cholesterol levels, blood pressure, and other relevant health metrics.

## Results
The logistic regression model provides accurate predictions for heart disease. The application also displays the probability of heart disease for better interpretability.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the system.

## License
Ganesh Basnet

## Acknowledgments
- [Streamlit documentation](https://docs.streamlit.io/)
- [scikit-learn documentation](https://scikit-learn.org/)
- Publicly available heart disease datasets
- Inspiration from healthcare analytics projects
