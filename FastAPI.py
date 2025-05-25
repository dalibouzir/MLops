from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
from typing import Optional, List
import requests
import os
import logging
from model.predictor import PlayerPredictor

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set of known team codes from your training dataset (adjust as needed)
KNOWN_TEAM_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

class PlayerInput(BaseModel):
    now_cost: float
    form: float
    minutes: int
    goals_scored: int
    assists: int
    yellow_cards: int
    team_name: str
    next_opponent: str

    @validator('team_name', 'next_opponent')
    def validate_team_codes(cls, v):
        try:
            code = int(v)
        except ValueError:
            raise ValueError(f"Team code must be a numeric string, got '{v}'")
        if code not in KNOWN_TEAM_CODES:
            logger.warning(f"Unknown team code '{v}' replaced with '0'")
            return "0"  # Replace unknown team codes with '0'
        return v

class PredictRequest(BaseModel):
    players: Optional[List[PlayerInput]] = [
        PlayerInput(
            now_cost=55,
            form=5.6,
            minutes=1800,
            goals_scored=5,
            assists=7,
            yellow_cards=1,
            team_name="10",      # '10' is unknown -> will be replaced with '0'
            next_opponent="14"
        )
    ]

@app.post("/predict")
def predict(request: PredictRequest = PredictRequest()):
    model_path = "model/fantasy_model.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise HTTPException(status_code=500, detail=f"Model file not found at {model_path}")

    try:
        predictor = PlayerPredictor()
        predictor.load_model(model_path)
        input_players = [player.model_dump() for player in request.players]
        predictions = predictor.predict_performance(input_players)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. See server logs for details.")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Your fetch_and_categorize_players() method from before goes here
    # For brevity, you can add it back unchanged
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("FastAPI:app", host="0.0.0.0", port=8000, reload=True)
