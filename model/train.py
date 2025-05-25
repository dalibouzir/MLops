# import requests
# from predictor import PlayerPredictor
# import os
# import logging
# import mlflow
# import mlflow.sklearn
# import argparse
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from datetime import datetime

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def fetch_players_from_api():
#     url = "https://fantasy.premierleague.com/api/bootstrap-static/"
#     response = requests.get(url)
#     data = response.json()
#     players = []
#     for player in data['elements']:
#         players.append({
#             "now_cost": player['now_cost'],
#             "form": float(player['form']),
#             "minutes": player['minutes'],
#             "goals_scored": player['goals_scored'],
#             "assists": player['assists'],
#             "yellow_cards": player['yellow_cards'],
#             "team_name": str(player['team_code']),
#             "next_opponent": str(player.get('next_opponent', 0)),
#             "total_points": player['total_points']
#         })
#     return players

# def train_and_save_model(model_path="model/fantasy_model.pkl", tracking_uri=None):
#     """
#     Train model, log to MLflow, register it, and save locally.
#     tracking_uri: MLflow tracking URI (e.g., 'http://mlflow:5000' for Docker,
#                   or 'http://localhost:5001' for local).
#     """
#     # Import visualization libraries
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     if tracking_uri is None:
#         # Check ENV or default
#         if os.environ.get("IN_DOCKER", "0") == "1":
#             tracking_uri = "http://mlflow:5000"
#         else:
#             tracking_uri = "http://localhost:5001"
#     mlflow.set_tracking_uri(tracking_uri)

#     os.makedirs("model", exist_ok=True)
#     players = fetch_players_from_api()
#     logger.info(f"Fetched {len(players)} players from FPL API.")

#     predictor = PlayerPredictor()
#     predictor.train_model(players)

#     input_example = {
#         "now_cost": 55,
#         "form": 5.6,
#         "minutes": 1800,
#         "goals_scored": 5,
#         "assists": 7,
#         "yellow_cards": 1,
#         "team_name": "10",
#         "next_opponent": "14"
#     }

#     with mlflow.start_run() as run:
#         # Set custom tags for the run
#         mlflow.set_tag("project", "Fantasy Premier League Assistant")
#         mlflow.set_tag("data_source", "FPL API")
#         mlflow.set_tag("data_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#         mlflow.set_tag("model_purpose", "Player point prediction")
        
#         # Log model parameters
#         mlflow.log_param("model_type", "RandomForestRegressor")
#         mlflow.log_param("n_estimators", predictor.model.named_steps['regressor'].n_estimators)
#         mlflow.log_param("random_state", predictor.model.named_steps['regressor'].random_state)
        
#         # Log preprocessing parameters
#         mlflow.log_param("numeric_features", "now_cost, form, minutes, goals_scored, assists, yellow_cards")
#         mlflow.log_param("categorical_features", "team_name, next_opponent")
#         mlflow.log_param("imputation_strategy", "mean")
        
#         # Define feature groups
#         numeric_features = ['now_cost', 'form', 'minutes', 'goals_scored', 'assists', 'yellow_cards']
#         categorical_features = ['team_name', 'next_opponent']
#         all_features = numeric_features + categorical_features
        
#         # Calculate and log metrics
#         X = pd.DataFrame(players)[all_features]
#         y = pd.DataFrame(players)['total_points']
        
#         # Split data for validation
#         from sklearn.model_selection import train_test_split
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Log training metrics
#         train_score = predictor.model.score(X_train, y_train)
#         mlflow.log_metric("train_score", train_score)
        
#         # Log validation metrics
#         val_score = predictor.model.score(X_val, y_val)
#         mlflow.log_metric("validation_score", val_score)
        
#         # Calculate predictions for validation set
#         y_pred = predictor.model.predict(X_val)
        
#         # Log advanced evaluation metrics
#         mlflow.log_metric("mae", mean_absolute_error(y_val, y_pred))
#         mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_val, y_pred)))
#         mlflow.log_metric("r2", r2_score(y_val, y_pred))
        
#         # Create and log correlation heatmap for numeric features
#         import matplotlib.pyplot as plt
#         import seaborn as sns
        
#         numeric_X = X_train[numeric_features]
#         plt.figure(figsize=(10, 8))
#         corr_matrix = numeric_X.corr()
#         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#         plt.title('Feature Correlation Matrix')
#         plt.tight_layout()
#         plt.savefig('correlation_matrix.png')
#         mlflow.log_artifact('correlation_matrix.png')
        
#         # Log distribution of predictions vs actual values
#         plt.figure(figsize=(10, 6))
#         plt.hist(y_val, alpha=0.5, label='Actual Points', bins=20)
#         plt.hist(y_pred, alpha=0.5, label='Predicted Points', bins=20)
#         plt.legend()
#         plt.title('Distribution of Actual vs Predicted Points')
#         plt.xlabel('Points')
#         plt.ylabel('Frequency')
#         plt.tight_layout()
#         plt.savefig('prediction_distribution.png')
#         mlflow.log_artifact('prediction_distribution.png')
        
#         # Log scatter plot of predicted vs actual values
#         plt.figure(figsize=(10, 6))
#         plt.scatter(y_val, y_pred, alpha=0.5)
#         plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
#         plt.xlabel('Actual Points')
#         plt.ylabel('Predicted Points')
#         plt.title('Actual vs Predicted Points')
#         plt.tight_layout()
#         plt.savefig('actual_vs_predicted.png')
#         mlflow.log_artifact('actual_vs_predicted.png')
        
#         # Residual analysis
#         residuals = y_val - y_pred
#         plt.figure(figsize=(10, 6))
#         plt.scatter(y_pred, residuals, alpha=0.5)
#         plt.axhline(y=0, color='r', linestyle='-')
#         plt.xlabel('Predicted Points')
#         plt.ylabel('Residuals (Actual - Predicted)')
#         plt.title('Residual Analysis')
#         plt.tight_layout()
#         plt.savefig('residual_analysis.png')
#         mlflow.log_artifact('residual_analysis.png')
        
#         # Log basic performance metrics summary as a table
#         metrics_summary = {
#             'train_score': train_score,
#             'validation_score': val_score,
#             'mae': mean_absolute_error(y_val, y_pred),
#             'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
#             'r2': r2_score(y_val, y_pred),
#             'training_size': len(X_train),
#             'validation_size': len(X_val)
#         }
#         mlflow.log_dict(metrics_summary, "metrics_summary.json")
        
#         # Log feature importance
#         importances = predictor.model.named_steps['regressor'].feature_importances_
        
#         # Get column names from preprocessing pipeline
#         # This is complex because the preprocessor combines both numeric and categorical features
        
#         # Get one-hot encoded column names for categorical features
#         # The OneHotEncoder is directly used in the categorical transformer
#         ohe = predictor.model.named_steps['preprocessor'].transformers_[1][1]
#         categorical_cols = ohe.get_feature_names_out(categorical_features)
        
#         # Combined feature names in the correct order
#         all_feature_names = numeric_features + list(categorical_cols)
        
#         # Create feature importance dict (limit to actual number of features used)
#         feature_importance = dict(zip(all_feature_names[:len(importances)], importances))
#         mlflow.log_dict(feature_importance, "feature_importance.json")
        
#         # Log model
#         mlflow.sklearn.log_model(
#             predictor.model,
#             artifact_path="model",
#             input_example=[input_example]
#         )
        
#         # Log additional artifacts
#         import matplotlib.pyplot as plt
#         import seaborn as sns
        
#         # Feature importance plot
#         plt.figure(figsize=(12, 8))
#         # Create feature names from model's feature names (or create placeholders)
#         feature_indices = np.argsort(importances)[-20:]
#         feature_values = importances[feature_indices]
        
#         # Create feature labels (just use indices if names aren't available)
#         feature_names = [f"Feature {i}" for i in range(len(importances))]
        
#         # Plot horizontal bar chart of feature importance
#         plt.barh(y=np.arange(len(feature_indices)), width=feature_values)
#         plt.yticks(np.arange(len(feature_indices)), [feature_names[i] for i in feature_indices])
#         plt.title('Top 20 Features by Importance')
#         plt.xlabel('Importance')
#         plt.tight_layout()
#         plt.savefig('feature_importance.png')
#         mlflow.log_artifact('feature_importance.png')
        
#         # Model configuration
#         model_config = {
#             "model_type": "RandomForestRegressor",
#             "n_estimators": predictor.model.named_steps['regressor'].n_estimators,
#             "random_state": predictor.model.named_steps['regressor'].random_state,
#             "training_size": len(players),
#             "validation_size": len(X_val)
#         }
#         mlflow.log_dict(model_config, "model_config.json")
        
#         model_uri = f"runs:/{run.info.run_id}/model"
#         mlflow.register_model(model_uri, "fantasy_model")
#         logger.info(f"Model registered as 'fantasy_model' in MLflow.")

#     predictor.save_model(model_path)
#     logger.info(f"Model trained and saved to {model_path}.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--docker", action="store_true", help="Run with Docker MLflow settings")
#     parser.add_argument("--tracking-uri", type=str, default=None, help="Override MLflow tracking URI")
#     args = parser.parse_args()

#     # Priority: CLI arg > env IN_DOCKER > default localhost
#     if args.tracking_uri:
#         uri = args.tracking_uri
#     elif args.docker or os.environ.get("IN_DOCKER", "0") == "1":
#         uri = "http://mlflow:5000"
#     else:
#         uri = "http://localhost:5001"

#     train_and_save_model(tracking_uri=uri)

import requests
from predictor import PlayerPredictor
import os
import logging
import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from elasticsearch import Elasticsearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Elasticsearch client
es_host = os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
es = Elasticsearch(es_host)

def log_to_elasticsearch(index_name, doc):
    try:
        es.index(index=index_name, document=doc)
        logger.info(f"Logged to Elasticsearch index '{index_name}'")
    except Exception as e:
        logger.error(f"Failed to log to Elasticsearch: {e}")

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
    import matplotlib.pyplot as plt
    import seaborn as sns

    if tracking_uri is None:
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
        mlflow.set_tag("project", "Fantasy Premier League Assistant")
        mlflow.set_tag("data_source", "FPL API")
        mlflow.set_tag("data_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.set_tag("model_purpose", "Player point prediction")

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", predictor.model.named_steps['regressor'].n_estimators)
        mlflow.log_param("random_state", predictor.model.named_steps['regressor'].random_state)

        mlflow.log_param("numeric_features", "now_cost, form, minutes, goals_scored, assists, yellow_cards")
        mlflow.log_param("categorical_features", "team_name, next_opponent")
        mlflow.log_param("imputation_strategy", "mean")

        numeric_features = ['now_cost', 'form', 'minutes', 'goals_scored', 'assists', 'yellow_cards']
        categorical_features = ['team_name', 'next_opponent']
        all_features = numeric_features + categorical_features

        X = pd.DataFrame(players)[all_features]
        y = pd.DataFrame(players)['total_points']

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        train_score = predictor.model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)

        val_score = predictor.model.score(X_val, y_val)
        mlflow.log_metric("validation_score", val_score)

        y_pred = predictor.model.predict(X_val)

        mlflow.log_metric("mae", mean_absolute_error(y_val, y_pred))
        mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_val, y_pred)))
        mlflow.log_metric("r2", r2_score(y_val, y_pred))

        # Plot and log correlation matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = X_train[numeric_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        mlflow.log_artifact('correlation_matrix.png')
        plt.close()

        # Distribution of actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.hist(y_val, alpha=0.5, label='Actual Points', bins=20)
        plt.hist(y_pred, alpha=0.5, label='Predicted Points', bins=20)
        plt.legend()
        plt.title('Distribution of Actual vs Predicted Points')
        plt.xlabel('Points')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('prediction_distribution.png')
        mlflow.log_artifact('prediction_distribution.png')
        plt.close()

        # Scatter plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_val, y_pred, alpha=0.5)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
        plt.xlabel('Actual Points')
        plt.ylabel('Predicted Points')
        plt.title('Actual vs Predicted Points')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        mlflow.log_artifact('actual_vs_predicted.png')
        plt.close()

        # Residual analysis
        residuals = y_val - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted Points')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title('Residual Analysis')
        plt.tight_layout()
        plt.savefig('residual_analysis.png')
        mlflow.log_artifact('residual_analysis.png')
        plt.close()

        metrics_summary = {
            'train_score': train_score,
            'validation_score': val_score,
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'r2': r2_score(y_val, y_pred),
            'training_size': len(X_train),
            'validation_size': len(X_val)
        }
        mlflow.log_dict(metrics_summary, "metrics_summary.json")

        importances = predictor.model.named_steps['regressor'].feature_importances_
        ohe = predictor.model.named_steps['preprocessor'].transformers_[1][1]
        categorical_cols = ohe.get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(categorical_cols)
        feature_importance = dict(zip(all_feature_names[:len(importances)], importances))
        mlflow.log_dict(feature_importance, "feature_importance.json")

        mlflow.sklearn.log_model(
            predictor.model,
            artifact_path="model",
            input_example=[input_example]
        )

        # Feature importance plot
        plt.figure(figsize=(12, 8))
        feature_indices = np.argsort(importances)[-20:]
        feature_values = importances[feature_indices]
        feature_names = [f"Feature {i}" for i in range(len(importances))]
        plt.barh(y=np.arange(len(feature_indices)), width=feature_values)
        plt.yticks(np.arange(len(feature_indices)), [feature_names[i] for i in feature_indices])
        plt.title('Top 20 Features by Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        mlflow.log_artifact('feature_importance.png')
        plt.close()

        model_config = {
            "model_type": "RandomForestRegressor",
            "n_estimators": predictor.model.named_steps['regressor'].n_estimators,
            "random_state": predictor.model.named_steps['regressor'].random_state,
            "training_size": len(players),
            "validation_size": len(X_val)
        }
        mlflow.log_dict(model_config, "model_config.json")

        # Elasticsearch logging for Kibana
        run_info = mlflow.get_run(run.info.run_id)
        es_doc = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            "end_time": datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
            "tags": dict(run_info.data.tags),
            "params": dict(run_info.data.params),
            "metrics": dict(run_info.data.metrics),
            "model_name": "fantasy_model",
            "timestamp": datetime.now().isoformat()
        }
        log_to_elasticsearch(index_name="fpl-model-training-logs", doc=es_doc)

        mlflow.register_model(f"runs:/{run.info.run_id}/model", "fantasy_model")
        logger.info(f"Model registered as 'fantasy_model' in MLflow.")

    predictor.save_model(model_path)
    logger.info(f"Model trained and saved to {model_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", action="store_true", help="Run with Docker MLflow settings")
    parser.add_argument("--tracking-uri", type=str, default=None, help="Override MLflow tracking URI")
    args = parser.parse_args()

    if args.tracking_uri:
        uri = args.tracking_uri
    elif args.docker or os.environ.get("IN_DOCKER", "0") == "1":
        uri = "http://mlflow:5000"
    else:
        uri = "http://localhost:5001"

    train_and_save_model(tracking_uri=uri)
