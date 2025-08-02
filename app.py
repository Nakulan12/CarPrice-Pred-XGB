import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Try to load the model and scaler with error handling
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaling.pkl', 'rb'))
    print("✅ Model and scaler loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Running in demo mode - using mock predictions")
    model = None
    scaler = None
    MODEL_LOADED = False

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

def mock_prediction(form_data):
    """Generate a realistic mock prediction when model is not available"""
    try:
        # Base price calculation using some key features
        base_price = 15000
        
        # Engine size impact
        engine_size = float(form_data.get('enginesize', 150))
        base_price += engine_size * 50
        
        # Horsepower impact
        horsepower = float(form_data.get('horsepower', 100))
        base_price += horsepower * 80
        
        # Car body type impact
        body_multipliers = {
            'sedan': 1.0, 'hatchback': 0.9, 'wagon': 1.1, 
            'convertible': 1.4, 'hardtop': 1.2
        }
        body_type = form_data.get('carbody', 'sedan')
        base_price *= body_multipliers.get(body_type, 1.0)
        
        # Fuel type impact
        if form_data.get('fueltype') == 'diesel':
            base_price *= 1.1
            
        # Add some randomness for realism
        import random
        random.seed(hash(str(form_data)) % 1000)
        variation = random.uniform(0.85, 1.15)
        final_price = base_price * variation
        
        return max(8000, min(80000, final_price))  # Reasonable price range
    except:
        return 25000  # Fallback price

@app.route('/')
def home():
  return render_template('index.html', model_status=MODEL_LOADED)

@app.route('/predict')
def predict_page():
  return render_template('predict.html', model_status=MODEL_LOADED)

@app.route('/about')
def about():
  return render_template('about.html', model_status=MODEL_LOADED)

@app.route('/result')
def result():
  return render_template('result.html', model_status=MODEL_LOADED)

@app.route('/predict_api', methods=['POST'])
def predict_api():
  try:
      data = request.json['data']
      
      if MODEL_LOADED:
          # Real prediction with loaded model
          df = pd.DataFrame([data])
          
          # Encode categorical features
          for col in categorical_cols:
              if col in df.columns:
                  df[col] = df[col].map(label_encoders[col])
          
          if df[categorical_cols].isnull().any().any():
              return jsonify({"error": "Invalid categorical value detected."}), 400
          
          scaled_data = scaler.transform(df)
          prediction = model.predict(scaled_data)
          price = float(prediction[0])
      else:
          # Mock prediction
          price = mock_prediction(data)
      
      return jsonify({
          'prediction': price,
          'formatted_price': f"${price:,.2f}",
          'model_status': 'real' if MODEL_LOADED else 'demo'
      })
      
  except Exception as e:
      return jsonify({'error': str(e)}), 500

@app.route('/predict_form', methods=['POST'])
def predict_form():
  try:
      form_data = request.form.to_dict()
      
      if MODEL_LOADED:
          # Real prediction with loaded model
          for col in categorical_cols:
              if col in form_data:
                  val = form_data[col]
                  if val in label_encoders[col]:
                      form_data[col] = label_encoders[col][val]
                  else:
                      return render_template('predict.html', 
                                           error_message=f"Invalid value for {col}: {val}",
                                           model_status=MODEL_LOADED)
          
          final_features = [float(val) for val in form_data.values()]
          scaled_input = scaler.transform(np.array(final_features).reshape(1, -1))
          output = model.predict(scaled_input)[0]
          price = float(output)
          model_type = "XGBoost ML Model"
      else:
          # Mock prediction
          price = mock_prediction(form_data)
          model_type = "Demo Mode"
      
      return render_template('result.html', 
                           prediction_price=f"${price:,.2f}",
                           model_type=model_type,
                           model_status=MODEL_LOADED,
                           car_specs=form_data)
  
  except Exception as e:
      return render_template('predict.html', 
                           error_message=f"Error occurred: {e}",
                           model_status=MODEL_LOADED)

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

if __name__ == "__main__":
    print("🚗 Starting Car Price Prediction Server...")
    print(f"📊 Model Status: {'Loaded' if MODEL_LOADED else 'Demo Mode'}")
    app.run(debug=True, host='0.0.0.0', port=5000)
