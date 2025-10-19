import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd


app=Flask(__name__)

# Load model
regmodel= pickle.load(open('regmodel.pkl', 'rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html', prediction=None)

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     print(data)
#     print(np.array(list(data.values())).reshape(1, -1))
#     new_data=scalar.transform(np.array(list(data.values())).reshape(1, -1))
#     output=regmodel.predict(new_data)
#     print(output[0])
#     return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scalar.transform(np.array([features]))  # scale inputs
    prediction = regmodel.predict(final_features)[0]
    prediction = round(prediction, 2)
    return render_template('home.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)









