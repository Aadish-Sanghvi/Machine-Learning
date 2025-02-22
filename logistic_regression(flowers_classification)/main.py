import joblib
import numpy as np

# Load the trained model
model = joblib.load("trained_iris_model.pkl")
print("Model loaded successfully!")

# Take manual input for prediction
sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

# Convert input into a NumPy array
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Predict species
prediction = model.predict(user_input)

# Mapping prediction to species name
species = {0: "Versicolor", 1: "Virginica"}
print("\nPredicted Species:", species[prediction[0]])
