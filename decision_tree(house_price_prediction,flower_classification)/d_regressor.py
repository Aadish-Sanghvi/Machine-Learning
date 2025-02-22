import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1️ Load California Housing Data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # Target: Median House Value

# 2️ Split Data into Train & Test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️ Hyperparameter Tuning with GridSearchCV
param_grid = {
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse'],  # Regression Criteria
    'max_depth': [5, 10, 15, None],  # Depth of Tree
    'min_samples_split': [2, 5, 10],  # Min Samples to Split a Node
    'min_samples_leaf': [1, 2, 5],  # Min Samples at Leaf
}

#hypertunning of parameters
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='r2',n_jobs=-1)
grid_search.fit(X_train, y_train)

# 4 Best Parameters Found
print("Best Parameters:", grid_search.best_params_)

# 5 no need of this step it get removed in updation

# 6 Predictions on Test Set
y_pred = grid_search.predict(X_test)

# 7️ Model Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(r2, mse)
# 8️ Plot Actual vs. Predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Decision Tree Regression: Actual vs Predicted")
plt.show()

#save the trained model
import joblib 
joblib.dump(grid_search,"trained_decissiontree_regressor.pkl")