#!flask/bin/python
from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict/tx', methods=['GET'])
def create_task():
    tx_data = request.json
    model = joblib.load('house.model')
    res = model.predict([[1177.698,2,7,2,2,2]])
    return jsonify(res[0]), 201

if __name__ == '__main__':
    app.run(debug=True)
