import unittest
from tests.test_utils import TestPodcastListeningTimePredictor
from src.model import PodcastListeningTimePredictor
from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_PATH


def run_tests():
    """Exécute les tests unitaires"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPodcastListeningTimePredictor)
    unittest.TextTestRunner(verbosity=2).run(suite)

def train_and_save_model():
    """Entraîne et sauvegarde le modèle"""
    model = PodcastListeningTimePredictor(model_type='xgboost')
    X_train, y_train = model.load_test_data(TRAIN_DATA_PATH)
    model.train(X_train, y_train)
    model.save_model(MODEL_SAVE_PATH)

def generate_submission():
    """Génère le fichier de soumission"""
    from src.predict import generate_submission
    generate_submission(
        model_path=MODEL_SAVE_PATH,
        test_data_path=TEST_DATA_PATH,
        submission_file=SUBMISSION_FILE.format(model_name='mon_modele')
    )

if __name__ == "__main__":
    # Exécuter les tests unitaires
    run_tests()
    
    # Entraîner et sauvegarder le modèle
    train_and_save_model()
    
    # Générer le fichier de soumission
    generate_submission()