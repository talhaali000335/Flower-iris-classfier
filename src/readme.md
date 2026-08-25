# 🌸 Iris Flower Species Classifier

An end-to-end Machine Learning project that classifies iris flowers into three species using a production-style architecture.  
This project demonstrates an **MLOps-lite pipeline**, integrating model training, API deployment, and frontend interaction.

---

## 📌 Project Overview

This application:

- Trains a Machine Learning model using the Iris dataset
- Exposes the trained model through a REST API
- Provides an interactive web interface for real-time predictions
- Demonstrates separation of ML logic, backend API, and frontend UI

---

## 🛠️ Technology Stack

- Scikit-learn – Machine Learning
- Joblib – Model serialization
- Flask – Backend API
- Streamlit – Frontend UI
- Python 3.x

---

## 📂 Project Structure

```bash
iris-classifier/
├── src/
│   ├── __init__.py
│   ├── model_trainer.py
│   ├── prediction_api.py
│   └── web_app.py
├── models/
│   ├── iris_knn_model.pkl
│   └── iris_target_names.pkl
├── README.md
├── requirements.txt
└── run_project.sh
```

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

Make sure you have:

- Python 3.x
- pip installed

---

### 2️⃣ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/iris-classifier.git
cd iris-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the models directory:

```bash
mkdir models
```

---

## ▶️ Running the Application

You will need **three terminals**.

---

### 🔹 Step 1: Train the Model

```bash
python src/model_trainer.py
```

This will:
- Train the KNN model
- Save model artifacts in the `models/` directory

---

### 🔹 Step 2: Start the Backend API

```bash
python src/prediction_api.py
```

The Flask server will start on:

```
http://127.0.0.1:5000
```

Keep this terminal running.

---

### 🔹 Step 3: Launch the Frontend

```bash
streamlit run src/web_app.py
```

The Streamlit app will open in your browser and connect to the Flask API.

---

## 🤖 Model Information

### Dataset

- 150 samples
- 3 species:
  - Setosa
  - Versicolor
  - Virginica

### Algorithm

- K-Nearest Neighbors (K=3)

### Input Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 🧪 Example API Request (Optional)

You can test the API using curl:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features":[5.1,3.5,1.4,0.2]}'
```

---

## 🛑 Stopping the Application

- Close the Streamlit browser tab
- Press `Ctrl + C` in the Flask terminal

---

## 🎯 Project Purpose

This project demonstrates:

- Clean ML pipeline structure
- API deployment principles
- Frontend-backend communication
- Real-time model inference
- Production-style folder organization

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

Vaishnavi Prashant Nayak 
GitHub: https://github.com/VaishnaviNayak2023
