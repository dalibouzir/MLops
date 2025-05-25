# model/predictor.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

class PlayerPredictor:
    def __init__(self):
        self.model = None
        
    def train_model(self, players):
        """Train prediction model on player data"""
        df = pd.DataFrame(players)
        X = df[['now_cost', 'form', 'minutes', 'goals_scored', 'assists', 
                'yellow_cards', 'team_name', 'next_opponent']]
        y = df['total_points']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), 
                 ['now_cost', 'form', 'minutes', 'goals_scored', 'assists', 'yellow_cards']),
                ('cat', OneHotEncoder(), ['team_name', 'next_opponent'])
            ]
        )
        
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100))
        ])
        
        self.model.fit(X, y)
        return self.model
    
    def predict_performance(self, players):
        """Predict performance for list of players"""
        if not self.model:
            raise Exception("Model not trained or loaded!")
        df = pd.DataFrame(players)
        X = df[['now_cost', 'form', 'minutes', 'goals_scored', 
                'assists', 'yellow_cards', 'team_name', 'next_opponent']]
        
        predicted_points = self.model.predict(X)
        for idx, player in enumerate(players):
            player['predicted_points'] = float(predicted_points[idx])
        return players
    
    def save_model(self, filepath):
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        return self.model 
