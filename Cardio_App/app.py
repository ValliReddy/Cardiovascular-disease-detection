import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
from keras.models import load_model

class RandomForestCustom:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y.iloc[sample_indices].values
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.zeros((len(X), len(self.trees)))
        for i, tree in enumerate(self.trees):
            tree_preds[:, i] = tree.predict(X)
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=tree_preds)
        return majority_votes
# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define User model for database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Load models for biometric prediction and ECG model
MODEL_PATH = "models/models/ecg_model.keras"
LABEL_ENCODER_PATH = "models/models/label_encoder.pkl"
BIOMETRIC_MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Ensure models exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError("ECG model or label encoder not found!")

# Load ECG model and label encoder
ecg_model = load_model(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Load Biometric model and scaler
biometric_model = joblib.load(BIOMETRIC_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Image size for ECG model
TARGET_SIZE = (224, 224)

# Preprocess ECG image for prediction
def preprocess_image(image_path):
    """Preprocess an ECG image for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    image = cv2.resize(image, TARGET_SIZE)
    image = image / 255.0  # Normalize
    return np.reshape(image, (1, 224, 224, 3))  # Reshape for model input

# Predict ECG class
def predict_ecg(image_path):
    """Predict class of ECG image."""
    image = preprocess_image(image_path)
    prediction = ecg_model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for("register"))

        # Check if user already exists
        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already registered!", "error")
            return redirect(url_for("register"))

        # Hash the password and save the user to the database
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(first_name=first_name, last_name=last_name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please sign in.", "success")
        return redirect(url_for("signin"))

    return render_template("register.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Verify user credentials
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            flash(f"Welcome back, {user.first_name}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for("signin"))

    return render_template("signin.html")

@app.route("/predict")
def predict():
    if 'user_id' not in session:
        flash("You need to log in to access this page.", "error")
        return redirect(url_for("signin"))
    return render_template("biometrics.html")

@app.route("/predict_two")
def predict_two():
    if 'user_id' not in session:
        flash("You need to log in to access this page.", "error")
        return redirect(url_for("signin"))
    return render_template("ecg.html")

@app.route("/biometrics", methods=["GET", "POST"])
def biometrics():
    result = None

    if request.method == "POST":
        try:
            # Retrieving input data from the form
            age = float(request.form["age"])
            sex = int(request.form["sex"])
            chest_pain = int(request.form["chest_pain"])
            resting_bp = float(request.form["resting_bp"])
            cholesterol = float(request.form["cholesterol"])
            fasting_blood_sugar = int(request.form["fasting_blood_sugar"])
            resting_ecg = int(request.form["resting_ecg"])
            max_heart_rate = int(request.form["max_heart_rate"])
            exercise_angina = int(request.form["exercise_angina"])
            oldpeak = float(request.form["oldpeak"])
            st_slope = int(request.form["st_slope"])


            # Prepare input data as a numpy array and reshape it to match the model's input shape
            input_data = np.array([age, sex, chest_pain, resting_bp, cholesterol, fasting_blood_sugar,
                                   resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope]).reshape(1, -1)

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make a prediction using the loaded model
            prediction = biometric_model.predict(input_data_scaled)

            # Interpret the result
            if prediction == 1:
                result = "Risk of Cardiovascular Disease"
            else:
                result = "No Risk of Cardiovascular Disease"
        except ValueError as e:
            # Flash the error message if there is an issue with the input
            flash(f"Error: {str(e)}", "error")

    # Render the result in the biometrics.html template
    return render_template("biometrics.html", result=result)
# ECG prediction route
@app.route("/ecg", methods=["GET", "POST"])
def ecg():
    result = None
    uploaded_image_path = None

    if request.method == "POST":
        if "ecg_image" not in request.files or not request.files["ecg_image"]:
            return render_template("ecg.html", result="No file uploaded.")

        ecg_image = request.files["ecg_image"]
        if ecg_image and ecg_image.filename:
            os.makedirs("static/images", exist_ok=True)
            file_path = os.path.join("static/images", ecg_image.filename)
            ecg_image.save(file_path)
            uploaded_image_path = file_path

            # Get prediction
            result = predict_ecg(file_path)

    return render_template("ecg.html", result=result, uploaded_image=uploaded_image_path)

# Run the app
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)
