"""
Flask Web Application for Ice Cream Sales Prediction
This app loads a pre-trained Linear Regression model and provides a web interface
to predict ice cream sales based on temperature input.
"""

import pickle
import os
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = 'model.pkl'
model = None

def load_model():
    """Load the trained model from pickle file."""
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
        return True
    else:
        print(f"✗ Model file not found: {MODEL_PATH}")
        print("Please run 'python train_model.py' first to train and save the model.")
        return False

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to make predictions.
    Expects JSON with 'temperature' field.
    """
    try:
        # Get temperature from request
        data = request.get_json()
        temperature = float(data.get('temperature'))
        
        # Validate temperature input
        if temperature < -50 or temperature > 150:
            return jsonify({
                'success': False,
                'error': 'Temperature must be between -50°F and 150°F'
            }), 400
        
        # Make prediction
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please restart the application.'
            }), 500
        
        prediction = model.predict(np.array([[temperature]]))[0]
        
        # Ensure non-negative prediction
        prediction = max(0, round(prediction, 2))
        
        return jsonify({
            'success': True,
            'temperature': temperature,
            'predicted_sales': prediction,
            'message': f'At {temperature}°F, estimated ice cream sales: {prediction} units'
        })
    
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid input. Please enter a valid temperature number.'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """API endpoint to get model information."""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'algorithm': 'Linear Regression',
        'feature': 'Temperature (°F)',
        'target': 'Ice Creams Sold',
        'coefficient': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'equation': f'Sales = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Temperature'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load the model before starting the app
    if load_model():
        port = int(os.environ.get("PORT", 10000))
        print("\n✓ Flask app is ready to serve predictions!")
        print(f"✓ Visit http://localhost:{port} in your browser")
        app.run(host="0.0.0.0", port=port)
    else:
        print("\n✗ Cannot start app without a trained model.")
        print("Please run 'python train_model.py' first.")
        exit(1)
