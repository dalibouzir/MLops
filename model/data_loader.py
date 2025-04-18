import requests
from ..utils.database import get_teams_mapping

class DataLoader:
    def __init__(self):
        self.players = []
        self.teams = {}
        
    def fetch_players(self):
        """Fetch and process player data from FPL API"""
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            self.teams = get_teams_mapping(data)
            
            # Process players data
            self._process_players(data)
            
    def _process_players(self, data):
        """Process raw player data and add additional fields"""
        positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        next_opponents = self._get_next_opponents()
        
        self.players = [
            {
                **player,
                "team_name": self.teams.get(player.get("team"),
                "next_opponent": self.teams.get(next_opponents.get(player.get("team"))),
                "position": positions.get(player.get("element_type")),
            }
            for player in data.get("elements", [])
        ]
    
    def _get_next_opponents(self):
        """Fetch next opponents for each team"""
        fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
        fixtures = fixtures_response.json() if fixtures_response.status_code == 200 else []
        
        next_opponents = {}
        for fixture in fixtures:
            if fixture["event"] and not fixture.get("finished", False):
                home_team, away_team = fixture["team_h"], fixture["team_a"]
                next_opponents[home_team] = away_team
                next_opponents[away_team] = home_team
                
        return next_opponents