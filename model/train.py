import requests
import mlflow
import mlflow.sklearn
from predictor import PlayerPredictor
import numpy as np
input_example = np.array([[55, 5.6, 1800, 5, 7, 1, "10", "14"]])
mlflow.sklearn.log_model(
    predictor.model,
    "model",
    input_example=input_example
)

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
            "next_opponent": str(player.get('next_opponent', 0)), # Use real field if available!
            "total_points": player['total_points']
        })
    return players

def train_and_log_model(players, model_name="fantasy_model"):
    predictor = PlayerPredictor()
    model = predictor.train_model(players)
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("train_size", len(players))
        mlflow.sklearn.log_model(predictor.model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)
        print(f"Model registered as '{model_name}' in MLflow!")
    predictor.save_model("model/fantasy_model.pkl")

if __name__ == "__main__":
    players = fetch_players_from_api()
    print(f"Fetched {len(players)} players from FPL API.")
    train_and_log_model(players)
