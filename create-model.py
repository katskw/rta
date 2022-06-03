import numpy as np
import pandas as pd

dane = {'zm1':[3,4,np.NaN,2,6,1], 
        'zm2':[54,np.NaN, 56,342,643,123],
        'zm3':['niebieski','czerwony','czerwony',
               'niebieski','zielony','zielony'],
        'zm4':['male','male','female','male','female','female'],
        'target':[0,1,1,0,0,1]}

df = pd.DataFrame(dane)
df

X = df.drop(columns=['target'])
y = df['target']

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_features = ["zm1", "zm2"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_features = ["zm3", "zm4"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num_trans", numeric_transformer, numeric_features),
    ("cat_trans", categorical_transformer, categorical_features)
])

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("model", LogisticRegression())
])

from sklearn import set_config
set_config(display='diagram')
pipeline

from sklearn.model_selection import train_test_split
X_tr, X_test, y_tr, y_test = train_test_split(X,y,
test_size=0.2, random_state=42)

pipeline.fit(X_tr, y_tr)

score = pipeline.score(X_test, y_test)
print(score)

import joblib
joblib.dump(pipeline, 'my_model.pkl')

##################

param_grid = [
              {"preproc__num_trans__imputer__strategy":
              ["mean","median"],
               "model__n_estimators":[2,5,10,100,500],
               "model__min_samples_leaf": [1, 0.1],
               "model":[RandomForestClassifier()]},
              {"preproc__num_trans__imputer__strategy":
                ["mean","median"],
               "model__C":[0.1,1.0,10.0,100.0,1000],
                "model":[LogisticRegression()]}
]

from sklearn.model_selection import GridSearchCV


grid_search = GridSearchCV(pipeline, param_grid,
cv=2, verbose=1, n_jobs=-1)


grid_search.fit(X_tr, y_tr)

grid_search.best_params_


from sklearn.base import BaseEstimator, TransformerMixin

class DelOneValueFeature(BaseEstimator, TransformerMixin):
    """Transformacja usuwająca zmienne, które posiadają
    tylko jedną wartość w całej kolumnie. Takie kolumny
    nie nadają się do modelowania. Metoda fit() wyszuka
    wszystkie takie kolumny. Natomiast metoda transform()
    usunie je ze zbioru danych.
"""
    def __init__(self):
        self.one_value_features = []
    def fit(self, X, y=None):
        for feature in X.columns:
            unikalne = X[feature].unique()
            if len(unikalne)==1:
                self.one_value_features.append(feature)
        return self
    def transform(self, X, y=None):
        if not self.one_value_features:
            return X
        return X.drop(axis='columns',
        columns=self.one_value_features)

pipeline2 = Pipeline([
    ("moja_transformacja",DelOneValueFeature()),
    ("preprocesser", preprocessor),
    ("classifier", LogisticRegression())])
    
pipeline2.fit(X_tr, y_tr)
score2 = pipeline2.score(X_test, y_test)


# własny model 

# implementacja 
class Perceptron():
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(xi, target)
                update = self.eta*(target-self.predict(xi))
                #print(update)
                self.w_[1:] += update*xi
                self.w_[0] += update
                #print(self.w_)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)

X=np.array([[1],[2],[3],[4]])
y=[-1,-1,1,1]

# AND
# X = np.array([0,0],[0,1],[1,0],[1,1])
# y = [-1,-1,-1,1]
# XOR


d = Perceptron()
d.fit(X,y)

print(d.errors_)
print(d.w_)

d.predict(np.array([-3]))
np.array([1,2,3,4]).reshape(-1,1)

d.predict(np.array([1,2,3,4,6,-1,23,-3]).reshape(-1,1))