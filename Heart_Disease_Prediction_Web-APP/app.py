# from flask import Flask, render_template, request
# from flask_sqlalchemy import SQLAlchemy
# import numpy as np
# import joblib
# import os
#
# # Set the path to the model file
# location = r"E:\FinalProjects\pythonProject\Heart_Disease_Prediction_System\Heart_Disease_Prediction_Web-APP"
# fullpath = os.path.join(location, 'hdp_model.pkl')
#
# app = Flask(__name__)
#
# # Configuring the SQLite database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'  # Creates 'data.db' file in your project directory
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#
# # Initializing SQLAlchemy
# db = SQLAlchemy(app)
#
# # Load the trained model
# try:
#     model = joblib.load(fullpath)
#     print("Model loaded successfully!")
# except FileNotFoundError:
#     print(f"Model file not found at path: {fullpath}")
#     model = None
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")
#     model = None
#
#
#
# @app.route("/register")
# def register():
#     return render_template("register.html")
#
# @app.route("/")
# def home():
#     return render_template("login.html")
#
#
# @app.route("/detail", methods=["POST"])
# def submit():
#     if request.method == "POST":
#         name = request.form["Username"]
#         return render_template("details.html", n=name)
#     return render_template("index.html")  # Redirect to home if not POST
#
#
# @app.route('/predict', methods=["POST"])
# def predict():
#     if request.method == "POST" and model:
#         try:
#             # Collecting form data
#             age = int(request.form['age'])
#             sex = int(request.form['sex'])
#             cp = int(request.form['cp'])
#             trestbps = int(request.form['trestbps'])
#             chol = int(request.form['chol'])
#             fbs = int(request.form['fbs'])
#             restecg = int(request.form['restecg'])
#             thalach = int(request.form['thalach'])
#             exang = int(request.form['exang'])
#             oldpeak = float(request.form['oldpeak'])
#             slope = int(request.form['slope'])
#             ca = int(request.form['ca'])
#             thal = int(request.form['thal'])
#
#             # Prepare the data for prediction
#             values = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
#             prediction = model.predict(values)
#
#             return render_template('prediction.html', prediction=prediction[0])
#
#         except ValueError as ve:
#             return f"Error in input data: {ve}", 400  # Bad Request
#
#         except Exception as e:
#             return f"An error occurred during prediction: {e}", 500  # Internal Server Error
#
#     return render_template("index.html")  # Redirect to home if not POST or model not loaded
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from models import User, Prediction, db  # Importing models and database setup

# Set the path to the model file
location = r"E:\FinalProjects\pythonProject\Heart_Disease_Prediction_System\Heart_Disease_Prediction_Web-APP"
fullpath = os.path.join(location, 'hdp_model.pkl')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Configuring the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db.init_app(app)

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

# Create tables in the database
with app.app_context():
    db.create_all()


def get_user_from_cookie(request):
    user_id = request.cookies.get('user_id')
    if user_id:
        return User.query.get(int(user_id))
    return None


@app.route("/", methods=["GET", "POST"])
def home():
    user = get_user_from_cookie(request)
    if user:
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

                # Save prediction to the database
                new_prediction = Prediction(
                    user_id=user.id,
                    prediction_result=bool(prediction[0]),
                    age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol, fbs=fbs, restecg=restecg,
                    thalach=thalach, exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal
                )
                db.session.add(new_prediction)
                db.session.commit()

                return render_template('prediction.html', prediction=prediction[0], n=user.username)

            except ValueError as ve:
                return f"Error in input data: {ve}", 400  # Bad Request

            except Exception as e:
                return f"An error occurred during prediction: {e}", 500  # Internal Server Error

        return render_template("details.html", n=user.username)
    else:
        return redirect(url_for('login'))  # Redirect to login route


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["Username"]
        email = request.form["email"]
        password = request.form["password"]

        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already taken", 400

        # Save data to the database
        new_user = User(username, email, password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["Username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            response = make_response(redirect(url_for("home")))
            response.set_cookie('user_id', str(user.id), max_age=timedelta(days=30))
            return response
        else:
            return "Invalid username or password", 401

    return render_template("login.html")



def prediction_to_dict(prediction):
    return {
        'id': prediction.id,
        'user_id': prediction.user_id,
        'prediction_result': prediction.prediction_result,
        'prediction_date': prediction.prediction_date,
        'age': prediction.age,
        'sex': prediction.sex,
        'cp': prediction.cp,
        'trestbps': prediction.trestbps,
        'chol': prediction.chol,
        'fbs': prediction.fbs,
        'restecg': prediction.restecg,
        'thalach': prediction.thalach,
        'exang': prediction.exang,
        'oldpeak': prediction.oldpeak,
        'slope': prediction.slope,
        'ca': prediction.ca,
        'thal': prediction.thal
    }

# @app.route("/dashboard")
# def dashboard():
#     user = get_user_from_cookie(request)
#     if user:
#         predictions_query = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.prediction_date.desc()).limit(10).all()
#         predictions = [prediction_to_dict(p) for p in predictions_query]
#         return render_template("dashboard.html", user=user, predictions=predictions)
#     else:
#         return redirect(url_for('login'))

@app.route("/dashboard")
def dashboard():
    user = get_user_from_cookie(request)
    if user:
        predictions_query = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.prediction_date.desc()).limit(10).all()
        predictions = [prediction_to_dict(p) for p in predictions_query]
        return render_template("dashboard.html", user=user, predictions=predictions)
    else:
        return redirect(url_for('login'))


@app.route("/logout")
def logout():
    response = make_response(redirect(url_for("login")))
    response.delete_cookie('user_id')
    return response


if __name__ == "__main__":
    app.run(debug=True)
