import pandas as pd
import pickle

# Step 1: Load the trained model
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Step 2: Function to make predictions
def predict_crop(features):
    # Predict the crop using the trained model
    prediction = model.predict([features])
    return prediction[0]

# Step 3: Example: Predicting based on new input (e.g., N, P, K, temperature, humidity, ph, rainfall)
new_input = [90, 40, 40, 25, 80, 6.5, 200]  # Example input (N, P, K, temperature, humidity, pH, rainfall)
predicted_crop = predict_crop(new_input)

# Step 4: Output the prediction
print(f"Predicted Crop: {predicted_crop}")
