import os

# # Chemins des données
# DATA_DIR = os.path.join('..', 'data')
# TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
# TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')
TRAIN_DATA_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/test.csv'

# Chemins de sauvegarde des résultats
RESULTS_DIR = 'results'
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'podcast_model.joblib')
SUBMISSION_FILE = os.path.join(RESULTS_DIR, 'submitplayground_{model_name}.csv')

# Paramètres par défaut des modèles
DEFAULT_MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'regression',
        'random_state': 42
    }
}