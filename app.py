# app.py
# Flask API — send network traffic features, get threat prediction back

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model   = joblib.load("models/threat_model.pkl")
scaler  = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Cybersecurity Threat Detection API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with 41 numeric feature values.
    Returns whether the traffic is Normal or an Attack.
    """
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            "prediction": "Attack" if prediction == 1 else "Normal",
            "attack_probability": round(float(probability), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)