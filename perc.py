from sysconfig import get_python_version
from keras import Sequential
from flask import request
from IPython import get_ipython

class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)
    

perc = Perceptron()

import numpy as np
import pandas as pd



from sklearn.datasets import load_iris
iris=load_iris()


df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])
df['species'] = pd.Categorical.from_codes(iris.target,
iris.target_names)


df.head()
dic = {0:'setosa',1:'versicolor'}

df2=df.iloc[:100,[0,2,4]]
df2.head()
df2.tail()

from sklearn.model_selection import train_test_split

X=df2[["sepal length (cm)","petal length (cm)"]]
y=df2["target"]


X_train, X_test, y_train, y_test = train_test_split(X.values,y.values, test_size=0.2, random_state=44)


perc = Perceptron(eta=0.1, n_iter=10)
perc.fit(X_train,y_train)
pred = np.array(perc.predict(X_test))

jak_źle = np.sum((y_test-pred)**2)/np.size(pred)
jak_źle



import pickle


pickle.dump(perc,open("perc.pkl","wb"))



def zamien(predict):
    sl=int(input("Sepal length "))
    pl=int(input("Petal length "))
    dic = {0:'setosa',1:'versicolor'}
    return dic[predict[0]]


def zamien(predict):
    dic = {0:'setosa',1:'versicolor'}
    return dic[predict[0]]
zamien(perc.predict([[2,2]]))


pickled_model = pickle.load(open('perc.pkl', 'rb'))




import subprocess



p = subprocess.Popen(["python", "siup.py"])
response = request.get("http://127.0.0.1:5000/api/v1.0/predict", params={"x1": 2.5, "x2": 3.4})
response.json()



p.kill()