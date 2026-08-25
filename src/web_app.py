# src/web_app.py
import streamlit as st  # Main library for building the web UI
import requests         # To send HTTP requests to the Flask API

# --- Configuration ---
# Must match the port used in prediction_api.py
API_URL = "http://127.0.0.1:5000/predict" 

st.set_page_config(page_title="Iris Classifier", page_icon="🌺")

# --- UI Layout and Title ---
st.title("Iris Flower Species Classifier")
st.markdown("---")
st.markdown("## Input Flower Measurements (cm)")

# --- User Input Sliders ---
SLIDER_RANGES = {
    "Sepal Length": (4.0, 8.0, 5.8, 0.1),  
    "Sepal Width": (2.0, 4.5, 3.0, 0.1),
    "Petal Length": (1.0, 7.0, 4.3, 0.1),
    "Petal Width": (0.1, 2.5, 1.3, 0.1),
}

inputs = {}
cols = st.columns(2)
for i, (name, (min_val, max_val, default_val, step)) in enumerate(SLIDER_RANGES.items()):
    with cols[i % 2]:
        inputs[name] = st.slider(
            name, 
            min_value=min_val, 
            max_value=max_val, 
            value=default_val, 
            step=step
        )

# Order the features as required by the model: [SL, SW, PL, PW]
feature_list = [
    inputs["Sepal Length"],
    inputs["Sepal Width"],
    inputs["Petal Length"],
    inputs["Petal Width"],
]

# --- Prediction Logic ---
st.markdown("---")
if st.button("Predict Species", type="primary", use_container_width=True):
    
    # 1. Package the collected data into the format the Flask API expects
    data_to_send = {"features": feature_list}
    
    # 2. Send a POST request to the API
    try:
        response = requests.post(API_URL, json=data_to_send)
        
        # 3. Check for successful response
        if response.status_code == 200:
            result = response.json()
            species = result.get('species', 'Unknown')
            
            # Display the result
            st.success(f"**Predicted Species:** {species.capitalize()}")
            st.info(f"Input features used: {feature_list} cm")
            
        else:
            # Display API error
            st.error(f"API Error: Status Code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        # Display connection error if API is down
        st.error("Connection Error: The prediction server is not running.")
        st.warning("Please ensure you have run `python src/prediction_api.py`.")