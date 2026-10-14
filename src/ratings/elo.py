from collections import defaultdict
from typing import List
from .base import RatingSystem

class EloRating(RatingSystem):
    def __init__(self, initial_rating: float = 1500.0, k_team: float = 64.0, k_player: float = 32.0):
        self.team_ratings = defaultdict(lambda: initial_rating)
        self.player_ratings = defaultdict(lambda: initial_rating)
        self.k_team = k_team
        self.k_player = k_player

    def get_team_rating(self, team_id: str) -> float:
        return self.team_ratings[team_id]

    def get_player_rating(self, player_id: str) -> float:
        return self.player_ratings[player_id]

    def _expected_score(self, r1: float, r2: float) -> float:
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def predict_team_win_prob(self, t1: str, t2: str) -> float:
        return self._expected_score(self.get_team_rating(t1), self.get_team_rating(t2))

    def _elo_players_rating(self, players: List[str]) -> float:
        if not players:
            return 1500.0
        return sum(self.get_player_rating(p) for p in players) / len(players)

    def predict_player_win_prob(self, players_1: List[str], players_2: List[str]) -> float:
        r1 = self._elo_players_rating(players_1)
        r2 = self._elo_players_rating(players_2)
        return self._expected_score(r1, r2)

    def update_team(self, t1: str, t2: str, score_1: int, score_2: int) -> None:
        r1 = self.get_team_rating(t1)
        r2 = self.get_team_rating(t2)
        e1 = self._expected_score(r1, r2)
        e2 = self._expected_score(r2, r1)
        
        total_score = score_1 + score_2
        if total_score == 0:
            return
            
        s1 = score_1 / total_score
        s2 = score_2 / total_score
        
        self.team_ratings[t1] += self.k_team * (s1 - e1)
        self.team_ratings[t2] += self.k_team * (s2 - e2)

    def update_player(self, players_1: List[str], players_2: List[str], score_1: int, score_2: int) -> None:
        team_elo_1 = self._elo_players_rating(players_1)
        team_elo_2 = self._elo_players_rating(players_2)
        
        e1 = self._expected_score(team_elo_1, team_elo_2)
        e2 = self._expected_score(team_elo_2, team_elo_1)
        
        total_score = score_1 + score_2
        if total_score == 0:
            return
            
        s1 = score_1 / total_score
        s2 = score_2 / total_score

        for player_1 in players_1:
            player_rank = self.get_player_rating(player_1)
            ratio = team_elo_1 / player_rank if player_rank > 0 else 1.0
            self.player_ratings[player_1] += self.k_player * (s1 - e1) * ratio

        for player_2 in players_2:
            player_rank = self.get_player_rating(player_2)
            ratio = team_elo_2 / player_rank if player_rank > 0 else 1.0
            self.player_ratings[player_2] += self.k_player * (s2 - e2) * ratio
