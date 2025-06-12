import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

# Hardcoded label encoders (same as during training)
label_encoders = {
    'fueltype': {'gas': 1, 'diesel': 0},
    'aspiration': {'std': 1, 'turbo': 0},
    'doornumber': {'two': 1, 'four': 0},
    'carbody': {'sedan': 3, 'hatchback': 2, 'wagon': 4, 'hardtop': 1, 'convertible': 0},
    'drivewheel': {'fwd': 1, 'rwd': 2, '4wd': 0},
    'enginelocation': {'front': 1, 'rear': 0},
    'enginetype': {'ohc': 5, 'dohc': 1, 'ohcf': 6, 'rotor': 7, 'ohcv': 4, 'l': 0, 'dohcv': 2},
    'cylindernumber': {'four': 4, 'six': 6, 'five': 5, 'eight': 8, 'two': 2, 'three': 3, 'twelve': 12},
    'fuelsystem': {'mpfi': 4, '2bbl': 0, '1bbl': 1, '4bbl': 3, 'idi': 2, 'mfi': 5, 'spdi': 6, 'spfi': 7}
}
categorical_cols = list(label_encoders.keys())

@app.route('/')
def home():
    return render_template('home.html')  # Optional: can remove if no frontend

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Get input JSON from request
        data = request.json['data']
        print("Input data:", data)

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical features
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].map(label_encoders[col])

        # Check for any unmapped or invalid categories
        if df[categorical_cols].isnull().any().any():
            return jsonify({"error": "Invalid categorical value detected."}), 400

        # Apply scaler to numeric + encoded features
        scaled_data = scaler.transform(df)

        # Predict with the model
        prediction = model.predict(scaled_data)

        return jsonify({'prediction': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
