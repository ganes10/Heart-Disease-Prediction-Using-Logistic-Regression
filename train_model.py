import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample dataset (replace this with your actual dataset)
np.random.seed(42)
X = np.random.rand(100, 5) * 100  # 100 samples, 5 features
y = np.random.choice([0, 1], size=100)  # Binary target (0: No Disease, 1: Disease)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model
model_filename = "logistic_regressor1.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"✅ Model saved as {model_filename}")
