import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
# ========================= KNN CLASSIFIER ========================= #

# Load Iris dataset (for classification)
iris = load_iris()
X_cls, y_cls = iris.data, iris.target

# Train-Test Split
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_cls = scaler.fit_transform(X_train_cls)
X_test_cls = scaler.transform(X_test_cls)

# Hyperparameter tuning using GridSearchCV for Classification
param_grid_cls = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search_cls = GridSearchCV(KNeighborsClassifier(), param_grid_cls, cv=5, scoring='accuracy')
grid_search_cls.fit(X_train_cls, y_train_cls)

# Best parameters and model
print("Best Parameters for Classification:", grid_search_cls.best_params_)
print(grid_search_cls.best_score_)

# Predictions and Accuracy
y_pred_cls = grid_search_cls.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print("Best Model Accuracy (Classifier):", accuracy)

# ========================= KNN REGRESSOR ========================= #

# Load California Housing dataset (for regression)
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target

# Train-Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Feature Scaling
X_train_reg = scaler.fit_transform(X_train_reg)
X_test_reg = scaler.transform(X_test_reg)

# Hyperparameter tuning using GridSearchCV for Regression
param_grid_reg = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search_reg = GridSearchCV(KNeighborsRegressor(), param_grid_reg, cv=5, scoring='neg_mean_absolute_error')
grid_search_reg.fit(X_train_reg, y_train_reg)

# Best parameters and model
print("Best Parameters for Regression:", grid_search_reg.best_params_)
best_knn_reg = grid_search_reg.best_estimator_

# Predictions and MAE
y_pred_reg = best_knn_reg.predict(X_test_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print("Best Model MAE (Regressor):", mae)



#save trained models
joblib.dump(grid_search_cls, "trained_knn_classifier.pkl")
joblib.dump(grid_search_reg , "trained_knn_regressor.pkl")
