import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df=sns.load_dataset('iris')
print(df.head())

print(df['species'].unique())
df=df[df['species']!='setosa']
print(df.head())

df['species']=df['species'].map({'versicolor':0,'virginica':1})
print(df.head())
x=df.iloc[:,:-1]
y=df.iloc[:,-1] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.26, random_state=42)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

from sklearn.model_selection import GridSearchCV

parameters={'penalty':["l1","l2","elasticnet"],'C':[1,2,3,4,5,6,7,10,20,30,40],'max_iter':[100,200]}
classifier_grid=GridSearchCV(classifier,param_grid=parameters,scoring='accuracy',cv=5)
classifier_grid.fit(X_train,y_train)
# print(classifier_grid.best_params_)
# print(classifier_grid.best_score_)

# y_pred=classifier_grid.predict(X_test)
# print(y_pred)

from sklearn.metrics import accuracy_score,classification_report
# score=accuracy_score(y_pred,y_test)
# print(score)

# print(classification_report(y_pred,y_test))

# sns.pairplot(df,hue='species')
# plt.show()

joblib.dump(classifier_grid, "trained_iris_model(logistic_regression).pkl")
