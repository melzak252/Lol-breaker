import math
from collections import defaultdict
from typing import List, Tuple
from datetime import date
from glicko2 import Player
from .base import RatingSystem

class GlickoRating(RatingSystem):
    def __init__(self, tau: float = 0.5):
        self.tau = tau
        def create_player():
            p = Player()
            p._tau = self.tau
            return p
        self.team_ratings = defaultdict(create_player)
        self.player_ratings = defaultdict(create_player)
        self.player_last_played = {}
        self.team_last_played = {}

    def get_team_rating(self, team_id: str) -> Player:
        return self.team_ratings[team_id]

    def get_player_rating(self, player_id: str) -> Player:
        return self.player_ratings[player_id]

    def _g(self, rd: float) -> float:
        q = math.log(10) / 400
        return 1 / math.sqrt(1 + 3 * (q ** 2) * (rd ** 2) / (math.pi ** 2))

    def _expected_score(self, player_i: Player, player_j: Player) -> float:
        ri = player_i.rating
        rj = player_j.rating
        RDi = player_i.rd
        RDj = player_j.rd

        combined_rd = math.sqrt(RDi**2 + RDj**2)
        g_factor = self._g(combined_rd)

        exponent = -g_factor * (ri - rj) / 400
        return 1 / (1 + 10 ** exponent)

    def predict_team_win_prob(self, t1: str, t2: str) -> float:
        p1 = self.get_team_rating(t1)
        p2 = self.get_team_rating(t2)
        return self._expected_score(p1, p2)

    def _glicko_players_rating(self, players: List[Player]) -> Tuple[float, float]:
        if not players:
            return 1500.0, 350.0
        ratings = [p.rating for p in players]
        rds = [p.rd**2 for p in players]
        return sum(ratings) / len(ratings), math.sqrt(sum(rds) / len(rds))

    def predict_player_win_prob(self, players_1: List[str], players_2: List[str]) -> float:
        p1 = [self.get_player_rating(p) for p in players_1]
        p2 = [self.get_player_rating(p) for p in players_2]
        
        t1_rating, t1_rd = self._glicko_players_rating(p1)
        t2_rating, t2_rd = self._glicko_players_rating(p2)
        
        p1_obj = Player(int(t1_rating), int(t1_rd))
        p1_obj._tau = self.tau
        p2_obj = Player(int(t2_rating), int(t2_rd))
        p2_obj._tau = self.tau
        
        return self._expected_score(p1_obj, p2_obj)

    def update_team(self, t1: str, t2: str, score_1: int, score_2: int) -> None:
        # Note: Glicko update is usually done per match (list of scores), 
        # but to fit the interface we do it per game here.
        # For batch updates, a separate method might be better.
        p1 = self.get_team_rating(t1)
        p2 = self.get_team_rating(t2)
        # Keep the first opponent's pre-game state before mutating either player.
        rating_1, rd_1 = p1.rating, p1.rd

        p1.update_player([p2.rating], [p2.rd], [score_1])
        p2.update_player([rating_1], [rd_1], [score_2])
        
        self.team_ratings[t1] = p1
        self.team_ratings[t2] = p2

    def update_player(self, players_1: List[str], players_2: List[str], score_1: int, score_2: int) -> None:
        p1 = [self.get_player_rating(p) for p in players_1]
        p2 = [self.get_player_rating(p) for p in players_2]

        t1_rating, t1_rd = self._glicko_players_rating(p1)
        t2_rating, t2_rd = self._glicko_players_rating(p2)

        for p, player_id in zip(p1, players_1):
            p.update_player([t2_rating], [t2_rd], [score_1])
            self.player_ratings[player_id] = p

        for p, player_id in zip(p2, players_2):
            p.update_player([t1_rating], [t1_rd], [score_2])
            self.player_ratings[player_id] = p

    def apply_time_decay(self, player: Player, last_played_date: date | None, current_date: date, period_days: int = 7) -> None:
        if last_played_date is None:
            return

        days = (current_date - last_played_date).days
        if days <= 0:
            return

        periods = days // period_days
        for _ in range(periods):
            player.did_not_compete()

    def update_rd_before_match(self, t1: str, t2: str, players_1: List[str], players_2: List[str], match_date: date) -> None:
        for p_id in players_1 + players_2:
            last_time = self.player_last_played.get(p_id)
            self.apply_time_decay(self.get_player_rating(p_id), last_time, match_date)
            self.player_last_played[p_id] = match_date

        for t_id in [t1, t2]:
            last_time = self.team_last_played.get(t_id)
            self.apply_time_decay(self.get_team_rating(t_id), last_time, match_date)
            self.team_last_played[t_id] = match_date
