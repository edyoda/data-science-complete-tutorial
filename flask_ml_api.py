#!flask/bin/python
from flask import Flask, jsonify, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/predict/tx', methods=['GET'])
def create_task():
    tx_data = request.json
    #model = joblib.load('model0.pipeline')
    model.predict([[]])
    #tx_data['P'] = model.best_estimator_.predict(df)[0]
    #cwd.write_json_table(tx_data)
    #tx_data['P'] = str(tx_data['P'])
    return jsonify("Hello World"), 201

if __name__ == '__main__':
    app.run(debug=True)
