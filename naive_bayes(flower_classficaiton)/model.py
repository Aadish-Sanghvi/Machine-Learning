import seaborn as sns
import pandas as pd
import numpy as np
data=sns.load_dataset('iris')

# print(data.head())

from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['species']=encoder.fit_transform(data['species'])
# print(data.head())

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split,GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
parameter={
    'var_smoothing':np.logspace(0,-9,num=100)
}
nb_grid=GridSearchCV(nb,parameter,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
nb_grid.fit(X_train,y_train)

print(nb_grid.best_params_)
print(nb_grid.best_score_)

nb_pred=nb_grid.predict(X_test)
print(nb_pred)
print(y_test)
from sklearn.metrics import classification_report,accuracy_score
print(accuracy_score(y_test,nb_pred))
print(classification_report(y_test,nb_pred))

#save the trained model 
import joblib
joblib.dump(nb_grid,"trained_nave.pkl")
