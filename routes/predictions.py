from flask import Blueprint, render_template
from model.predictor import PlayerPredictor
from model.optimizer import SquadOptimizer

bp = Blueprint('predictions', __name__)

@bp.route('/predict', methods=['GET', 'POST'])
def predict_best_squad():
    # Initialize components
    predictor = PlayerPredictor()
    optimizer = SquadOptimizer(budget=100)
    
    # Get players (would come from DataLoader in real implementation)
    players = []  # This would come from your data source
    
    # Make predictions
    players_with_preds = predictor.predict_performance(players)
    
    # Optimize squad
    squad, total_cost, predicted_points = optimizer.optimize(players_with_preds)
    
    return render_template('predict.html', 
                         squad=squad,
                         total_cost=total_cost,
                         predicted_points=predicted_points)