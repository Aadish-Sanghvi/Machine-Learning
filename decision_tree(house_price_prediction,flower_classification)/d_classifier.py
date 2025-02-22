import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
data=sns.load_dataset('iris')
# print(data.head())
data['species']=LabelEncoder().fit_transform(data['species'])
print(data.head())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42)

tree=DecisionTreeClassifier()
parameter={
    'criterion':["gini","entropy", "log_loss"],
    'splitter': ['best', 'random'],
    'max_depth':[1,2,3,4,None],
    'max_features':['auto', 'sqrt', 'log2']
}
gred_dec=GridSearchCV(tree,param_grid=parameter,cv=5,scoring='accuracy')
gred_dec.fit(X_train,y_train)

print(gred_dec.best_params_)
print(gred_dec.best_score_)

tree_pred=gred_dec.predict(X_test)
print(tree_pred)
print(y_test)

from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(tree_pred,y_test)
print(score)

print(classification_report(tree_pred,y_test))

#save the trained model
import joblib
joblib.dump(gred_dec,"trained_decissiontree_classifier.pkl")