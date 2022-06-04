import pickle
from sklearn.model_selection import train_test_split

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd

# Create a flask
app = Flask(__name__)

# Create an API end point
@app.route("/api/v1.0/predict", methods=['GET'])
def make_prediction():
    a1 = request.args.get("a1", 0, type=float)
    a2 = request.args.get("a2", 0, type=float)
    a3 = request.args.get("a3", 0, type=float)
    a4 = request.args.get("a4", 0, type=float)
    
    arr = np.array([[a1,a2,a3,a4]])
    pickled_model = pickle.load(open('perc.pkl', 'rb'))
    result = pickled_model.predict(arr) 
    
    return jsonify(result)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)