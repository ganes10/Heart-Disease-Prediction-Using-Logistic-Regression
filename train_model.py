import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import set_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enable feature name preservation
set_config(transform_output="pandas")

# Load the dataset
data = pd.read_csv('/Users/praswishbasnet/Documents/GitHub/Heart-Disease-Prediction-Using-Logistic-Regression/data/HeartDisease_Dataset.csv')

# Define features and target
X = data.drop(columns=['HeartDisease'])
y = data['HeartDisease']

# Preprocessing
numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# FIX: Add sparse_output=False to OneHotEncoder
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Critical fix
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split the data
logging.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log feature details
logging.info(f"Original features: {X.columns.tolist()}")
logging.info(f"Categorical features: {categorical_features}")

# Train the model
logging.info("Training the model...")
pipeline.fit(X_train, y_train)

# Log transformed feature names
try:
    transformed_features = pipeline.named_steps['preprocessor'].get_feature_names_out()
    logging.info(f"Transformed features: {transformed_features.tolist()}")
except Exception as e:
    logging.warning(f"Feature name logging failed: {str(e)}")

# Evaluate
logging.info("Evaluating model...")
y_pred = pipeline.predict(X_test)

logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Save model
logging.info("Saving model...")
with open('logistic_regressor1.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

logging.info("Training completed successfully")