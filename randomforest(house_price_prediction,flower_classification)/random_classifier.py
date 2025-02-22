import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load dataset (Replace 'your_data.csv' with your actual file)
df = sns.load_dataset('iris')

# Assuming the last column is the target variable (Classification)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Splitting into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestClassifier()

# Hyperparameter tuning grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 10, 20],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Min samples to split
    'min_samples_leaf': [1, 2, 4],  # Min samples per leaf
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features per split
    'bootstrap': [True, False],  # Whether to use bootstrapping
}

# GridSearchCV for tuning
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train with best parameters
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Final model with best params
best_rf = grid_search.best_estimator_

# Predictions
y_pred = best_rf.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
