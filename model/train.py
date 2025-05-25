import requests
from predictor import PlayerPredictor
import os
import logging
import mlflow
import mlflow.sklearn
import argparse

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
            "next_opponent": str(player.get('next_opponent', 0)),
            "total_points": player['total_points']
        })
    return players

def train_and_save_model(model_path="model/fantasy_model.pkl", tracking_uri=None):
    """
    Train model, log to MLflow, register it, and save locally.
    tracking_uri: MLflow tracking URI (e.g., 'http://mlflow:5000' for Docker,
                  or 'http://localhost:5001' for local).
    """

    if tracking_uri is None:
        # Check ENV or default
        if os.environ.get("IN_DOCKER", "0") == "1":
            tracking_uri = "http://mlflow:5000"
        else:
            tracking_uri = "http://localhost:5001"
    mlflow.set_tracking_uri(tracking_uri)

    os.makedirs("model", exist_ok=True)
    players = fetch_players_from_api()
    logger.info(f"Fetched {len(players)} players from FPL API.")

    predictor = PlayerPredictor()
    predictor.train_model(players)

    input_example = {
        "now_cost": 55,
        "form": 5.6,
        "minutes": 1800,
        "goals_scored": 5,
        "assists": 7,
        "yellow_cards": 1,
        "team_name": "10",
        "next_opponent": "14"
    }

    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("train_size", len(players))
        mlflow.sklearn.log_model(
            predictor.model,
            artifact_path="model",
            input_example=[input_example]
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "fantasy_model")
        logger.info(f"Model registered as 'fantasy_model' in MLflow.")

    predictor.save_model(model_path)
    logger.info(f"Model trained and saved to {model_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true", help="Run with Docker MLflow settings")
    parser.add_argument("--tracking-uri", type=str, default=None, help="Override MLflow tracking URI")
    args = parser.parse_args()

    # Priority: CLI arg > env IN_DOCKER > default localhost
    if args.tracking_uri:
        uri = args.tracking_uri
    elif args.docker or os.environ.get("IN_DOCKER", "0") == "1":
        uri = "http://mlflow:5000"
    else:
        uri = "http://localhost:5001"

    train_and_save_model(tracking_uri=uri)
