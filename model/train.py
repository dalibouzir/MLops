import requests
from predictor import PlayerPredictor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_players_from_api():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    players = []
    for player in data['elements']:
        players.append({
            "now_cost": player['now_cost'],
            "form": float(player['form']),
            "minutes": player['minutes'],
            "goals_scored": player['goals_scored'],
            "assists": player['assists'],
            "yellow_cards": player['yellow_cards'],
            "team_name": str(player['team_code']),
            "next_opponent": str(player.get('next_opponent', 0)),  # Adjust if next_opponent data is available
            "total_points": player['total_points']
        })
    return players

def train_and_save_model(model_path="model/fantasy_model.pkl"):
    players = fetch_players_from_api()
    logger.info(f"Fetched {len(players)} players from FPL API.")
    
    predictor = PlayerPredictor()
    predictor.train_model(players)
    predictor.save_model(model_path)
    logger.info(f"Model trained and saved to {model_path}.")

if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)
    train_and_save_model()
