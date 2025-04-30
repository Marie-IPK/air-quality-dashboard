import pandas as pd
from src.config import *
from src.model import PodcastListeningTimePredictor
from src.data_preprocessing import load_data

def generate_submission(model_path, test_data_path, submission_file):
    """
    Génère le fichier de soumission à partir du modèle entraîné et des données de test.
    
    Args:
        model_path (str): Chemin vers le modèle entraîné.
        test_data_path (str): Chemin vers les données de test.
        submission_file (str): Chemin vers le fichier de soumission.
    """
    # Charger les données de test
    test_df = load_data(test_data_path)
    
    # Charger le modèle entraîné
    model = PodcastListeningTimePredictor()
    model.load_model(model_path)
    
    # Effectuer les prédictions sur les données de test
    X_test = model.preprocessor.transform(test_df)
    y_pred = model.predict(X_test)
    
    # Créer le fichier de soumission
    submission_df = pd.DataFrame({'id': test_df['id'], 'Listening_Time_minutes': y_pred})
    submission_df.to_csv(submission_file, index=False)
    print(f"Fichier de soumission généré : {submission_file}")