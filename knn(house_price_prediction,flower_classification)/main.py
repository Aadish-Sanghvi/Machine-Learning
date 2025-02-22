import numpy as np
import joblib

# Load trained models
classifier = joblib.load(r"C:\Users\abhin\Desktop\machine_learning\knn\trained_knn_classifier.pkl")
regressor = joblib.load(r"C:\Users\abhin\Desktop\machine_learning\knn\trained_knn_regressor.pkl")

# ========================= USER INPUT FOR CLASSIFICATION ========================= #
def classify_flower():
    print("\nEnter flower features for classification (Iris dataset):")
    
    try:
        sepal_length = float(input("Sepal Length: "))
        sepal_width = float(input("Sepal Width: "))
        petal_length = float(input("Petal Length: "))
        petal_width = float(input("Petal Width: "))

        # Prepare input data
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict species
        prediction = classifier.predict(user_input)[0]
        species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        
        print("\nPredicted Flower Species:", species[prediction])

    except ValueError:
        print("Invalid input! Please enter numeric values.")

# ========================= USER INPUT FOR REGRESSION ========================= #
def predict_house_price():
    print("\nEnter housing features for regression (California Housing dataset):")
    
    features = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", 
        "AveOccup", "Latitude", "Longitude"
    ]
    
    user_input = []
    
    try:
        for feature in features:
            value = float(input(f"{feature}: "))
            user_input.append(value)
        
        # Convert to NumPy array
        user_input = np.array([user_input])

        # Predict house price
        predicted_price = regressor.predict(user_input)[0]
        print("\nPredicted House Price:", round(predicted_price, 2), "USD (approx)")

    except ValueError:
        print("Invalid input! Please enter numeric values.")

# ========================= MENU TO CHOOSE FUNCTION ========================= #
while True:
    print("\nChoose an option:")
    print("1. Classify a flower (Iris Dataset)")
    print("2. Predict house price (California Housing Dataset)")
    print("3. Exit")

    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        classify_flower()
    elif choice == "2":
        predict_house_price()
    elif choice == "3":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Please try again.")
