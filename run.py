import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("ML_linear_iris.pkl", "rb"))

@app.route("/", methods = ["POST", "GET"])
def home():
    return render_template("index.html", Predict = 0)

@app.route("/predict", methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        feature = [float(feature) for feature in request.form.values()]
        species = {0: "Iris-setosa",1: "Iris-versicolor",2: "Iris-virginica"}
        prediction_output = model.predict([np.array(feature)])
        return render_template("index.html", Predict = species[prediction_output[0]])

if __name__ == "__main__":
    app.run(debug= True)