import os
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH, SUBMISSION_FILE
from src.data_preprocessing import DataPreprocessor
from src.model import PodcastListeningTimePredictor
from src.predict import generate_submission

def train_and_save_model():
    """Entraîne et sauve le modèle"""
    model = PodcastListeningTimePredictor()
    X_train, y_train, X_test = model.load_test_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    model.train(X_train, y_train)
    model.save_model(MODEL_SAVE_PATH)

def generate_submission_file():
    """Génère le fichier de soumission à partir du modèle entraîné"""
    model = PodcastListeningTimePredictor()
    model.load_model(MODEL_SAVE_PATH)
    generate_submission(MODEL_SAVE_PATH, TEST_DATA_PATH, SUBMISSION_FILE)

if __name__ == "__main__":
    # Création du répertoire de sauvegarde du modèle s'il n'existe pas
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Entraînement et sauvegarde du modèle
    train_and_save_model()

    # Génération du fichier de soumission
    generate_submission_file()