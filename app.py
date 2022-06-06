from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd



app = Flask(__name__)

@app.route("/api/v1.0/predict", methods=['GET'])
def make_prediction():   
    x1 = request.args.get("x1", 0, type=float)   
    x2 = request.args.get("x2", 0, type=float) 
    
    arr = np.array([[x1,x2]])
    pickled_model = pickle.load(open('perc.pkl', 'rb'))
    result = pickled_model.predict(arr)    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="localhost", port=80)