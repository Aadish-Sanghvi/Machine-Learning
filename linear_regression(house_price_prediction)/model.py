import pandas as pd
import numpy as np 
import sklearn
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import joblib

df=fetch_california_housing()
# print(df)
x=pd.DataFrame(df.data, columns=df.feature_names)
y=pd.DataFrame(df.target, columns=['Target'])
# print(x)
# print(y)

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42)

# print(X_train)

#standardizing the dataset
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
# print(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

regression=LinearRegression()
regression.fit(X_train,y_train)

mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
print(np.mean(mse))
# print(mse)
reg_pred=regression.predict(X_test)
print(reg_pred)

import seaborn as sb
# sb.displot(reg_pred-y_test,kind='kde')
# plt.show()

from sklearn.metrics import r2_score
# train_score = r2_score(y_train, regression.predict(X_train))
# print("Training R² Score:", train_score*100)

# test_score=r2_score(y_test,reg_pred)
# print(test_score*100)


                                #RIDGE REGRESSION

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge_regressor=Ridge()

parameters={'alpha':[1,2,3,4,5,6,7,8,9,10,20,30,40]}
ridgecv=GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_squared_error',cv=5)
ridgecv.fit(X_train,y_train)
print(ridgecv.best_params_)
print(ridgecv.best_score_)
ridge_pred=ridgecv.predict(X_test)
print(ridge_pred)

sb.displot(ridge_pred - y_test.values.flatten(), kind='kde')
plt.show()


                       #lasso l1 regularization


from sklearn.linear_model import Lasso
lasso=Lasso()
parameters={'alpha':[1,2,3,4,5,6,7,8,9,10,20,30,40]}
lassocv=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lassocv.fit(X_train,y_train)
print(lassocv.best_params_)
print(lassocv.best_score_)
lasso_pred=lassocv.predict(X_test)
print(lasso_pred)

sb.displot(lasso_pred - y_test.values.flatten(), kind='kde')
plt.show()

#save the above trained models
joblib.dump(regression,"trainded_linear_regression.pkl")
joblib.dump(ridgecv,"trainded_ridge_regression.pkl")
joblib.dump(lassocv,"trainded_lasso_regression.pkl")