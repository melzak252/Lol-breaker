import math
from collections import defaultdict
from typing import List
from trueskill import TrueSkill, Rating
from .base import RatingSystem

class TrueSkillRating(RatingSystem):
    def __init__(self, draw_probability: float = 0.0, beta: float = 25.0 / 2.0, tau: float = 25.0 / 300.0, mu: float = 25.0, sigma: float = 25.0 / 3.0):
        self.ts_model = TrueSkill(draw_probability=draw_probability, beta=beta, tau=tau, mu=mu, sigma=sigma)
        self.team_ratings = defaultdict(lambda: self.ts_model.create_rating())
        self.player_ratings = defaultdict(lambda: self.ts_model.create_rating())

    def get_team_rating(self, team_id: str) -> Rating:
        return self.team_ratings[team_id]

    def get_player_rating(self, player_id: str) -> Rating:
        return self.player_ratings[player_id]

    def _expected_score(self, team_a: List[Rating], team_b: List[Rating]) -> float:
        mu_a = sum(r.mu for r in team_a) / len(team_a)
        mu_b = sum(r.mu for r in team_b) / len(team_b)

        sigma2_a = sum(r.sigma ** 2 for r in team_a) / len(team_a)
        sigma2_b = sum(r.sigma ** 2 for r in team_b) / len(team_b)

        delta_mu = mu_a - mu_b
        denom = math.sqrt(sigma2_a + sigma2_b + 2 * (self.ts_model.beta ** 2))

        return float(self.ts_model.cdf(delta_mu / denom))

    def predict_team_win_prob(self, t1: str, t2: str) -> float:
        r1 = self.get_team_rating(t1)
        r2 = self.get_team_rating(t2)
        return self._expected_score([r1], [r2])

    def predict_player_win_prob(self, players_1: List[str], players_2: List[str]) -> float:
        r1 = [self.get_player_rating(p) for p in players_1]
        r2 = [self.get_player_rating(p) for p in players_2]
        return self._expected_score(r1, r2)

    def update_team(self, t1: str, t2: str, score_1: int, score_2: int) -> None:
        r1 = self.get_team_rating(t1)
        r2 = self.get_team_rating(t2)
        
        if score_1 > score_2:
            r1_new, r2_new = self.ts_model.rate([[r1], [r2]])
        elif score_2 > score_1:
            r2_new, r1_new = self.ts_model.rate([[r2], [r1]])
        else:
            r1_new, r2_new = self.ts_model.rate([[r1], [r2]], ranks=[0, 0])
            
        self.team_ratings[t1] = r1_new[0]
        self.team_ratings[t2] = r2_new[0]

    def update_player(self, players_1: List[str], players_2: List[str], score_1: int, score_2: int) -> None:
        r1 = [self.get_player_rating(p) for p in players_1]
        r2 = [self.get_player_rating(p) for p in players_2]
        
        if score_1 > score_2:
            r1_new, r2_new = self.ts_model.rate([r1, r2])
        elif score_2 > score_1:
            r2_new, r1_new = self.ts_model.rate([r2, r1])
        else:
            r1_new, r2_new = self.ts_model.rate([r1, r2], ranks=[0, 0])
            
        for p, player_id in zip(r1_new, players_1):
            self.player_ratings[player_id] = p
        for p, player_id in zip(r2_new, players_2):
            self.player_ratings[player_id] = p
