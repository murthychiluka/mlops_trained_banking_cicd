# ----------------------------
# app.py
# ----------------------------

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

# ----------------------------
# Load Model and Encoders
# ----------------------------
try:
    model = joblib.load("models/model_full.pkl")
    le_gender = joblib.load("models/encoder_gender.pkl")
    le_married = joblib.load("models/encoder_married.pkl")
    le_approved = joblib.load("models/encoder_approved.pkl")
    model_status = "Model loaded successfully"

    # Store feature list for validation
    model.expected_features = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'gender', 'married']

except Exception as e:
    model = None
    le_gender = None
    le_married = None
    le_approved = None
    model_status = f"Error loading model: {str(e)}"


# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template("form.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not trained or unavailable"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # ----------------------------
        # Validate Feature Names
        # ----------------------------
        expected_fields = getattr(model, "expected_features", None)
        received_fields = list(data.keys())
        extra_fields = [f for f in received_fields if f not in expected_fields]
        missing_fields = [f for f in expected_fields if f not in received_fields]

        if extra_fields or missing_fields:
            return jsonify({
                "error": "Invalid fields",
                "missing_fields": missing_fields,
                "extra_fields": extra_fields
            }), 400

        # ----------------------------
        # Extract + Encode Inputs
        # ----------------------------
        try:
            age = float(data.get('age'))
            income = float(data.get('income'))
            loan_amount = float(data.get('loan_amount'))
            loan_term = float(data.get('loan_term'))
            credit_score = float(data.get('credit_score'))
        except ValueError:
            return jsonify({"error": "Numeric fields must be numbers"}), 400

        gender = data.get('gender').lower()
        married = data.get('married').lower()

        if gender not in le_gender.classes_:
            return jsonify({"error": f"Invalid gender: {gender}. Allowed: {list(le_gender.classes_)}"}), 400
        if married not in le_married.classes_:
            return jsonify({"error": f"Invalid married status: {married}. Allowed: {list(le_married.classes_)}"}), 400

        gender_encoded = le_gender.transform([gender])[0]
        married_encoded = le_married.transform([married])[0]

        input_features = np.array([[age, income, loan_amount, loan_term, credit_score, gender_encoded, married_encoded]])

        # ----------------------------
        # Make Prediction
        # ----------------------------
        prediction = model.predict(input_features)
        result = le_approved.inverse_transform(prediction)[0]

        return jsonify({"prediction": result, "status": "success"})

    except Exception as e:
        print("Backend error:", str(e))
        return jsonify({
            "error": "Model not trained or invalid input. Please check fields."
        }), 400


@app.route('/health', methods=['GET'])
def health():
    if "successfully" in model_status:
        return jsonify({
            "status": "ok",
            "message": model_status
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": model_status
        }), 500


# ----------------------------
# Run Server
# ----------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
