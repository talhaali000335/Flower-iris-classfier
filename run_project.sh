#!/bin/bash

# --- Iris Classifier Project Startup Script ---

echo "Starting Iris Classifier Setup and Deployment..."

# 1. Install Dependencies
echo -e "\n1. Installing dependencies..."
pip install -r requirements.txt || { echo "Installation failed."; exit 1; }

# 2. Train and Save Model
echo -e "\n2. Running model training script..."
python src/model_trainer.py || { echo "Model training failed."; exit 1; }

# 3. Start Flask API Server (in background)
echo -e "\n3. Starting Flask API Server on port 5000..."
nohup python src/prediction_api.py > api_log.out 2>&1 &
FLASK_PID=$!
echo "Flask API started with PID: $FLASK_PID"
sleep 3 # Give the server a moment to start

# 4. Start Streamlit Frontend
echo -e "\n4. Starting Streamlit Web App (opens in browser)..."
streamlit run src/web_app.py

# 5. Clean up (this part runs after you close the Streamlit app)
echo -e "\nWeb app closed. Stopping Flask API (PID: $FLASK_PID)..."
kill $FLASK_PID
echo "Cleanup complete."