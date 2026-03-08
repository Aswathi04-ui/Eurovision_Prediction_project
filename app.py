from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("eurovision_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    english = int(request.form["english"])
    group = int(request.form["group"])
    danceability = float(request.form["danceability"])
    energy = float(request.form["energy"])

    features = np.array([[english,group,danceability,energy]])

    prediction = model.predict(features)[0]

    return render_template("index.html",
        prediction_text=f"Predicted Points: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)