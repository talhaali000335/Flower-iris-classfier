# src/model_trainer.py
import joblib                                      # Used to save and load the trained model (serialization)
from sklearn.datasets import load_iris             # Used to easily load the built-in Iris dataset
from sklearn.neighbors import KNeighborsClassifier # Our chosen classification algorithm (KNN)
import os                                          # Used for handling file paths across different operating systems

# --- Configuration for File Saving ---

# Define the directory where the trained model files will be saved. 
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True) # Create the 'models' directory if it doesn't exist

# Define the full paths for the saved files
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_knn_model.pkl')
TARGET_NAMES_PATH = os.path.join(MODEL_DIR, 'iris_target_names.pkl')

def train_and_save_model():
    """
    Main function to handle data loading, model training, and saving.
    """
    
    print("Starting model training...")

    # 1. Load Data
    iris = load_iris()
    X, y = iris.data, iris.target # X holds the features (measurements), y holds the target (species labels)

    # 2. Train Model 
    # Initialize the KNN classifier with n_neighbors=3
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y) # Train the model using all the available data

    # 3. Save the Model and Target Names
    joblib.dump(model, MODEL_PATH) # Save the trained model object to a file
    joblib.dump(iris.target_names, TARGET_NAMES_PATH) # Save the list of species names

    print(f"Training complete and model saved.")

if __name__ == "__main__":
    train_and_save_model()