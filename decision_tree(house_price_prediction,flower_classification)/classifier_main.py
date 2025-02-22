import numpy as np
import joblib

# Load the trained Decision Tree model
model = joblib.load(r"C:\Users\abhin\Desktop\machine_learning\decision_tree\trained_decissiontree_classifier.pkl")

# Define feature names (from the Iris dataset)
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Get user input for each feature
print("Enter values for the following features:")
user_input = []
for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Convert to NumPy array and reshape for prediction
user_input = np.array(user_input).reshape(1, -1)

# Make prediction
prediction = model.predict(user_input)

# Class mapping (Iris dataset species)
class_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Print the prediction result
print(f"Predicted Species: {class_mapping[prediction[0]]}")
