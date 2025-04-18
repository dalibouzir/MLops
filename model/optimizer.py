from pulp import LpMaximize, LpProblem, LpVariable, lpSum

class SquadOptimizer:
    def __init__(self, budget=100):
        self.budget = budget
        self.position_limits = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        
    def optimize(self, players):
        """Optimize squad selection using linear programming"""
        num_players = len(players)
        problem = LpProblem("FantasyFootballSquad", LpMaximize)
        
        # Decision variables
        decision_vars = [LpVariable(f"x{i}", cat="Binary") for i in range(num_players)]
        
        # Objective function
        predicted_points = [player['predicted_points'] for player in players]
        problem += lpSum([decision_vars[i] * predicted_points[i] for i in range(num_players)])
        
        # Constraints
        self._add_constraints(problem, decision_vars, players)
        
        # Solve problem
        problem.solve()
        
        # Extract solution
        return self._get_solution(decision_vars, players)
    
    def _add_constraints(self, problem, decision_vars, players):
        """Add all optimization constraints"""
        # Budget constraint
        costs = [player['now_cost'] / 10 for player in players]
        problem += lpSum([decision_vars[i] * costs[i] for i in range(len(players))]) <= self.budget
        
        # Position constraints
        positions = [player['position'] for player in players]
        for pos, limit in self.position_limits.items():
            problem += lpSum([decision_vars[i] for i in range(len(players)) 
                        if positions[i] == pos]) == limit
        
        # Squad size
        problem += lpSum(decision_vars) == 15
        
        # Team constraint (max 3 players per team)
        team_names = [player['team_name'] for player in players]
        unique_teams = set(team_names)
        for team in unique_teams:
            problem += lpSum([decision_vars[i] for i in range(len(players)) 
                          if team_names[i] == team]) <= 3
    
    def _get_solution(self, decision_vars, players):
        """Extract solution from optimization"""
        selected_indices = [i for i in range(len(players)) if decision_vars[i].varValue == 1]
        squad = [players[i] for i in selected_indices]
        
        total_cost = sum(player['now_cost'] / 10 for player in squad)
        total_points = sum(player['predicted_points'] for player in squad)
        
        return squad, total_cost, total_points