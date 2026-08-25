# src/prediction_api.py
import joblib                     # To load the pre-trained model
from flask import Flask, request, jsonify # Flask for the web server and JSON handling
import os                         # For managing file paths

# --- Configuration and Model Loading ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_knn_model.pkl')
TARGET_NAMES_PATH = os.path.join(MODEL_DIR, 'iris_target_names.pkl')

model = None
target_names = None

try:
    # Load the trained model and species names into memory
    model = joblib.load(MODEL_PATH)
    target_names = joblib.load(TARGET_NAMES_PATH)
    print("Model and target names loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model files not found. Run model_trainer.py first and check paths!")

app = Flask(__name__)

# --- API Endpoint Definition ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives flower measurements via POST request and returns the predicted species.
    """
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
        
    try:
        data = request.get_json()
        features = data['features']
        
        # Prepare input for the model
        prediction_input = [features]
        
        # Make prediction
        prediction = model.predict(prediction_input)
        
        # Convert index (0, 1, 2) to species name
        predicted_species_index = prediction[0]
        predicted_species = target_names[predicted_species_index]
        
        # Return the result
        return jsonify({
            'prediction_index': int(predicted_species_index), 
            'species': predicted_species
        })

    except Exception as e:
        # Handle invalid input or processing errors
        return jsonify({'error': f'Invalid input or processing error: {str(e)}'}), 400

if __name__ == '__main__':
    # Start the Flask server
    print("\n--- Starting Flask API Server ---")
    app.run(debug=True, port=5000)