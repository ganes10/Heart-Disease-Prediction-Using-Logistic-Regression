import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load your dataset (replace 'heart_disease_data.csv' with your actual dataset file)
data = pd.read_csv('/Users/praswishbasnet/Desktop/Heart-Disease-Prediction/data/heart_disease_data.csv')

# Define features and target
X = data.drop(columns=['HeartDisease'])  # Replace 'HeartDisease' with the actual target column name
y = data['HeartDisease']

# Preprocessing
numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# Create a pipeline with preprocessing and the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Save the trained pipeline (preprocessor + model)
with open('logistic_regressor1.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model trained and saved as 'logistic_regressor1.pkl'")