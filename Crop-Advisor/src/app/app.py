from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = 'models/trained_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    model = None  # Handle missing model case
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Trained model not found!'})

    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract input values
        N = float(data['nitrogen'])
        P = float(data['phosphorus'])
        K = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Prepare input for prediction
        features = [[N, P, K, temperature, humidity, ph, rainfall]]

        # Make prediction
        predicted_crop = model.predict(features)[0]

        return jsonify({'predicted_crop': predicted_crop})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

# Handle favicon request
@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

if __name__ == '__main__':
    app.run(debug=True)
