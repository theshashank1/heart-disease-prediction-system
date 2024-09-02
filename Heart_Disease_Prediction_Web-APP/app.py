from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib
import os

# Set the path to the model file
location = r"E:\FinalProjects\pythonProject\Heart_Disease_Prediction_System\Heart_Disease_Prediction_Web-APP"
fullpath = os.path.join(location, 'hdp_model.pkl')

app = Flask(__name__)

# Configuring the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'  # Creates 'data.db' file in your project directory
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initializing SQLAlchemy
db = SQLAlchemy(app)

# Load the trained model
try:
    model = joblib.load(fullpath)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file not found at path: {fullpath}")
    model = None
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detail", methods=["POST"])
def submit():
    if request.method == "POST":
        name = request.form["Username"]
        return render_template("details.html", n=name)
    return render_template("index.html")  # Redirect to home if not POST


@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST" and model:
        try:
            # Collecting form data
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            # Prepare the data for prediction
            values = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = model.predict(values)

            return render_template('prediction.html', prediction=prediction[0])

        except ValueError as ve:
            return f"Error in input data: {ve}", 400  # Bad Request

        except Exception as e:
            return f"An error occurred during prediction: {e}", 500  # Internal Server Error

    return render_template("index.html")  # Redirect to home if not POST or model not loaded


if __name__ == "__main__":
    app.run(debug=True)
