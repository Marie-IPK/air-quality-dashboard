import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from joblib import dump, load
import os
import optuna
from src.config import *
from src.data_preprocessing import load_data, DataPreprocessor

class PodcastListeningTimePredictor:

    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.preprocessor = DataPreprocessor()

    def train(self, X_train, y_train):
        """Entraîne le modèle sur les données d'entraînement"""
        if self.model is None:
            # Recherche des hyperparamètres optimaux avec Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=100)
            self.model = self.initialize_model(study.best_params)

        # Corriger les valeurs négatives dans y_train
        y_train = np.maximum(0, y_train)

        self.model.fit(X_train, y_train)
        return self.model

    def load_test_data(self, train_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH):
        """
        Charge et prétraite les données d'entraînement et de test.
        
        Args:
            train_data_path (str): Chemin vers les données d'entraînement.
            test_data_path (str): Chemin vers les données de test.
        
        Returns:
            tuple: X_train, y_train, X_test
        """
        # Charger les données d'entraînement
        train_df = load_data(train_data_path)
        X_train, y_train = self.preprocessor.fit_transform(train_df)

        # Charger les données de test
        test_df = load_data(test_data_path)
        X_test = self.preprocessor.transform(test_df)

        return X_train, y_train, X_test
    
    def _objective(self, trial, X, y):
        """Fonction objective pour Optuna"""
        if self.model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }
        elif self.model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 15, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42
            }
        
        model = self.initialize_model(params)
        cv_scores = self.cross_validate(X, y, cv=5)
        return cv_scores['mean_rmse']
    
    def initialize_model(self, params=None):
        """Initialise le modèle selon le type spécifié"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**params)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**params)
        # Ajoutez d'autres modèles selon vos besoins
        return self.model
    
    def predict(self, X):
        """Prédit le temps d'écoute sur de nouvelles données"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez train() d'abord.")
        
        # Prédire et s'assurer que les temps d'écoute ne sont pas négatifs
        predictions = self.model.predict(X)
        return np.maximum(0, predictions)
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle sur les données de test"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez train() d'abord.")
        
        y_pred = self.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Effectue une validation croisée sur les données"""
        if self.model is None:
            self.initialize_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        return {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'all_scores': rmse_scores
        }
    
    def save_model(self, filepath):
        """Sauvegarde le modèle entraîné dans un fichier"""
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné. Utilisez train() d'abord.")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        dump(self.model, filepath)
        print(f"Modèle sauvegardé dans {filepath}")
    
    def load_model(self, filepath):
        """Charge un modèle préalablement entraîné"""
        self.model = load(filepath)
        print(f"Modèle chargé depuis {filepath}")
        return self.model