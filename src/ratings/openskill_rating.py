from collections import defaultdict
from typing import List, Dict, Any
from openskill.models import PlackettLuce
from .base import RatingSystem

class OpenSkillRating(RatingSystem):
    def __init__(self, mu: float = 25.0, sigma: float = 8.333):
        self.model = PlackettLuce(mu=mu, sigma=sigma)
        self.team_ratings = defaultdict(lambda: self.model.rating())
        self.player_ratings = defaultdict(lambda: self.model.rating())

    def get_team_rating(self, team_id: str):
        return self.team_ratings[team_id]

    def get_player_rating(self, player_id: str):
        return self.player_ratings[player_id]

    def predict_team_win_prob(self, t1: str, t2: str) -> float:
        r1 = self.get_team_rating(t1)
        r2 = self.get_team_rating(t2)
        probs = self.model.predict_win([[r1], [r2]])
        return probs[0]

    def predict_player_win_prob(self, players_1: List[str], players_2: List[str]) -> float:
        r1 = [self.get_player_rating(p) for p in players_1]
        r2 = [self.get_player_rating(p) for p in players_2]
        probs = self.model.predict_win([r1, r2])
        return probs[0]

    def update_team(self, t1: str, t2: str, score_1: int, score_2: int) -> None:
        r1 = self.get_team_rating(t1)
        r2 = self.get_team_rating(t2)
        
        ranks: List[float]
        if score_1 > score_2:
            ranks = [0.0, 1.0]
        elif score_2 > score_1:
            ranks = [1.0, 0.0]
        else:
            ranks = [0.0, 0.0]
            
        new_ratings = self.model.rate([[r1], [r2]], ranks=ranks)
        self.team_ratings[t1] = new_ratings[0][0]
        self.team_ratings[t2] = new_ratings[1][0]

    def update_player(self, players_1: List[str], players_2: List[str], score_1: int, score_2: int) -> None:
        r1 = [self.get_player_rating(p) for p in players_1]
        r2 = [self.get_player_rating(p) for p in players_2]
        
        ranks: List[float]
        if score_1 > score_2:
            ranks = [0.0, 1.0]
        elif score_2 > score_1:
            ranks = [1.0, 0.0]
        else:
            ranks = [0.0, 0.0]
            
        new_ratings = self.model.rate([r1, r2], ranks=ranks)
        for p, player_id in zip(new_ratings[0], players_1):
            self.player_ratings[player_id] = p
        for p, player_id in zip(new_ratings[1], players_2):
            self.player_ratings[player_id] = p
