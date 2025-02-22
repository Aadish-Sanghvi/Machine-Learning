from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.25, random_state=42)

# Define parameter grid for tuning (including 'criterion')
param_grid = {
    'n_estimators': [50],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'absolute_error']  # Added criterion
}

# Perform Grid Search
rf_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_reg, param_grid, cv=5 , verbose=2,n_jobs=-5)
grid_search.fit(X_train, y_train)

# Best Parameters
# print("Best Parameters:", grid_search.best_params_)

# Predict on test set
# y_pred = grid_search.best_estimator_.predict(X_test)
z_pred=grid_search.predict(X_test)
# print(y_pred)
print(z_pred)

# Evaluate model
mse = mean_squared_error(y_test, z_pred)
# print("Test Mean Squared Error:", mse)

#save the trained model
import joblib 
joblib.dump(grid_search,"trainde_random_regressor.pkl")
