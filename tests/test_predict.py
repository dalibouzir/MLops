from fastapi.testclient import TestClient
from FastAPI import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "players": [
            {
                "now_cost": 85,
                "form": 6.5,
                "minutes": 2500,
                "goals_scored": 12,
                "assists": 9,
                "yellow_cards": 2,
                "team_name": "Arsenal",
                "next_opponent": "Man United"
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert "predicted_points" in result["predictions"][0]
