import unittest
import numpy as np
from src.model import PodcastListeningTimePredictor

class TestPodcastListeningTimePredictor(unittest.TestCase):
    def setUp(self):
        self.model = PodcastListeningTimePredictor(model_type='xgboost')
        
        # Données de test
        self.X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y_train = np.array([10, 20, 30])
        
    def test_train_and_predict(self):
        self.model.train(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_train)
        
        # Vérifier que les prédictions ne sont pas négatives
        self.assertTrue(np.all(y_pred >= 0))
        
    def test_evaluate(self):
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_train, self.y_train)
        
        # Vérifier que les métriques ont des valeurs raisonnables
        self.assertLess(metrics['rmse'], 10)
        self.assertLess(metrics['mae'], 5)
        self.assertGreater(metrics['r2'], 0.5)