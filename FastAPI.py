from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import requests
import os
from model.predictor import PlayerPredictor

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def fetch_and_categorize_players():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()

    categorized_players = {
        'goalkeepers': [],
        'defenders': [],
        'midfielders': [],
        'forwards': []
    }

    for player in data['elements']:
        player_data = {
            'id': player['id'],
            'name': player['web_name'],
            'cost': player['now_cost'] / 10,
            'points': player['total_points'],
            'goals': player['goals_scored'],
            'assists': player['assists'],
            'selected_percent': player['selected_by_percent'],
            # Features for ML prediction
            'now_cost': player['now_cost'],
            'form': float(player['form']),
            'minutes': player['minutes'],
            'goals_scored': player['goals_scored'],
            'assists': player['assists'],
            'yellow_cards': player['yellow_cards'],
            'team_name': str(player['team_code']),
            'next_opponent': str(player['team_code'])
        }
        if player['element_type'] == 1:
            categorized_players['goalkeepers'].append(player_data)
        elif player['element_type'] == 2:
            categorized_players['defenders'].append(player_data)
        elif player['element_type'] == 3:
            categorized_players['midfielders'].append(player_data)
        elif player['element_type'] == 4:
            categorized_players['forwards'].append(player_data)
    return categorized_players

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    players = fetch_and_categorize_players()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "players": players}
    )

class PlayerInput(BaseModel):
    now_cost: float
    form: float
    minutes: int
    goals_scored: int
    assists: int
    yellow_cards: int
    team_name: str
    next_opponent: str

class PredictRequest(BaseModel):
    players: list[PlayerInput]

@app.post("/predict")
def predict(request: PredictRequest):
    predictor = PlayerPredictor()
    predictor.load_model("model/model.joblib")  # <- Use saved model!
    input_players = [player.model_dump() for player in request.players]  # Pydantic v2+
    predictions = predictor.predict_performance(input_players)
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("FastAPI:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
