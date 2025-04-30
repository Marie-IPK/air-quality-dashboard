import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.config import *

# Vous pouvez maintenant utiliser les variables de configuration
train_data_path = TRAIN_DATA_PATH
test_data_path = TEST_DATA_PATH

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.numerical_features = []
        self.categorical_features = []
    
    def identify_feature_types(self, df):
        """Identifie les types de caractéristiques dans le DataFrame"""
        # Exclure 'id' et la variable cible 'Listening_Time_minutes' s'ils existent
        exclude_cols = ['id']
        if 'Listening_Time_minutes' in df.columns:
            exclude_cols.append('Listening_Time_minutes')
        
        self.numerical_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                                 if col not in exclude_cols]
        self.categorical_features = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                                   if col not in exclude_cols]
        
        return self.numerical_features, self.categorical_features
    
    def handle_missing_values(self, df):
        """Gère les valeurs manquantes de façon basique"""
        # Remplir les valeurs numériques manquantes avec la moyenne
        for col in self.numerical_features:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        
        # Remplir les valeurs catégorielles manquantes avec le mode
        for col in self.categorical_features:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def build_preprocessor(self):
        """Construit un pipeline de prétraitement avec sklearn"""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Utiliser la moyenne pour les valeurs manquantes
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Ignorer les colonnes non spécifiées
        )
        
        return self.preprocessor
    
    def fit_transform(self, df):
        """Applique le prétraitement au DataFrame et retourne les données transformées et la variable cible"""
        if not self.numerical_features and not self.categorical_features:
            self.identify_feature_types(df)

        if self.preprocessor is None:
            self.build_preprocessor()

        # Conserver les colonnes originales pour référence
        X = df.copy()
        y = df['Listening_Time_minutes'].values

        # Appliquer le préprocesseur
        X_transformed = self.preprocessor.fit_transform(X)

        return X_transformed, y
    
    def transform(self, df):
        """Transforme de nouvelles données avec le préprocesseur déjà entraîné"""
        if self.preprocessor is None:
            raise ValueError("Le préprocesseur n'a pas encore été entraîné. Utilisez fit_transform d'abord.")
        
        return self.preprocessor.transform(df)

def load_data(file_path):
    """Charge les données depuis un fichier CSV ou autre"""
    return pd.read_csv(file_path)

def feature_engineering(df):
    """Ajoute des caractéristiques supplémentaires pertinentes pour le temps d'écoute d'un podcast"""
    df_copy = df.copy()
    
    # Traitement des heures de publication (Time Features)
    if 'Publication_Time' in df_copy.columns:
        # Convertir en heure du jour (0-23)
        df_copy['Publication_Hour'] = df_copy['Publication_Time'].apply(
            lambda x: int(x.split(':')[0]) if isinstance(x, str) and ':' in x else np.nan
        )
        
        # Catégoriser l'heure de publication
        def categorize_time(hour):
            if pd.isna(hour):
                return np.nan
            elif 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        df_copy['Time_of_Day'] = df_copy['Publication_Hour'].apply(categorize_time)
    
    # Traitement du jour de publication
    if 'Publication_Day' in df_copy.columns:
        # Ordre des jours pour l'encodage numérique
        day_order = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        
        # Encodage numérique du jour
        df_copy['Day_Number'] = df_copy['Publication_Day'].map(day_order)
        
        # Catégoriser en jour de semaine vs weekend
        df_copy['Is_Weekend'] = df_copy['Publication_Day'].apply(
            lambda x: 1 if x in ['Saturday', 'Sunday'] else 0
        )
    
    # Caractéristiques pour la popularité
    if 'Host_Popularity_percentage' in df_copy.columns:
        # Catégoriser la popularité de l'hôte
        def categorize_popularity(pop):
            if pd.isna(pop):
                return np.nan
            elif pop < 25:
                return 'Low'
            elif pop < 50:
                return 'Medium-Low'
            elif pop < 75:
                return 'Medium-High'
            else:
                return 'High'
        
        df_copy['Host_Popularity_Category'] = df_copy['Host_Popularity_percentage'].apply(categorize_popularity)
    
    if 'Guest_Popularity_percentage' in df_copy.columns:
        df_copy['Guest_Popularity_Category'] = df_copy['Guest_Popularity_percentage'].apply(categorize_popularity)
        
        # Différence entre popularité de l'hôte et de l'invité
        df_copy['Host_Guest_Popularity_Diff'] = df_copy['Host_Popularity_percentage'] - df_copy['Guest_Popularity_percentage'].fillna(0)
    
    # Ratios intéressants
    if 'Episode_Length_minutes' in df_copy.columns and 'Number_of_Ads' in df_copy.columns:
        # Densité des publicités (nombre de pubs par minute)
        df_copy['Ad_Density'] = df_copy.apply(
            lambda row: row['Number_of_Ads'] / row['Episode_Length_minutes'] if not pd.isna(row['Episode_Length_minutes']) and row['Episode_Length_minutes'] > 0 else np.nan,
            axis=1
        )
    
    # Traitement du sentiment des épisodes
    if 'Episode_Sentiment' in df_copy.columns:
        # Convertir sentiment en valeur numérique
        sentiment_map = {
            'Very Negative': -2,
            'Negative': -1,
            'Neutral': 0,
            'Positive': 1,
            'Very Positive': 2
        }
        df_copy['Sentiment_Score'] = df_copy['Episode_Sentiment'].map(sentiment_map)
        
        # Binaire: positif ou non
        df_copy['Is_Positive'] = df_copy['Episode_Sentiment'].apply(
            lambda x: 1 if x in ['Positive', 'Very Positive'] else 0
        )
    
    # Features basées sur le titre de l'épisode
    if 'Episode_Title' in df_copy.columns:
        # Longueur du titre
        df_copy['Title_Length'] = df_copy['Episode_Title'].apply(lambda x: len(str(x)))
        
        # Présence de mots clés accrocheurs dans le titre
        clickbait_words = ['exclusive', 'reveal', 'secret', 'shocking', 'amazing', 'incredible', 
                           'must', 'never', 'best', 'worst', 'top', 'ultimate']
        
        df_copy['Has_Clickbait'] = df_copy['Episode_Title'].apply(
            lambda x: 1 if any(word in str(x).lower() for word in clickbait_words) else 0
        )
    
    # Interactions entre caractéristiques
    # Popularité combinée (hôte + invité)
    if 'Host_Popularity_percentage' in df_copy.columns and 'Guest_Popularity_percentage' in df_copy.columns:
        df_copy['Combined_Popularity'] = df_copy['Host_Popularity_percentage'] + df_copy['Guest_Popularity_percentage'].fillna(0)
    
    return df_copy