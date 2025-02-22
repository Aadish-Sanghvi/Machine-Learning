import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained models
linear_model = joblib.load(r"C:\Users\abhin\Desktop\machine_learning\linear_regression\trainded_linear_regression.pkl")
ridge_model = joblib.load(r"C:\Users\abhin\Desktop\machine_learning\linear_regression\trainded_ridge_regression.pkl")
lasso_model = joblib.load(r"C:\Users\abhin\Desktop\machine_learning\linear_regression\trainded_lasso_regression.pkl")

# Load feature scaler (you must save it in your training script)
scaler = StandardScaler()

# Define feature names
feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude"
]

def get_user_input():
    """Function to take user input for prediction"""
    print("Enter values for the following features:")
    input_data = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        input_data.append(value)

    return np.array(input_data).reshape(1, -1)

# Take user input
user_input = get_user_input()

# Standardize the input (must match the training scale)
scaled_input = scaler.fit_transform(user_input)

# Make predictions using the trained models
linear_pred = linear_model.predict(scaled_input)
ridge_pred = ridge_model.predict(scaled_input)
lasso_pred = lasso_model.predict(scaled_input)

# Display predictions
print("\nPredictions:")
print(f"Linear Regression Prediction: {linear_pred[0][0]:.4f}")
print(f"Ridge Regression Prediction: {ridge_pred[0]:.4f}")
print(f"Lasso Regression Prediction: {lasso_pred[0]:.4f}")
