from abc import ABC, abstractmethod
from typing import List, Dict, Any

class RatingSystem(ABC):
    @abstractmethod
    def get_team_rating(self, team_id: str) -> Any:
        pass

    @abstractmethod
    def get_player_rating(self, player_id: str) -> Any:
        pass

    @abstractmethod
    def update_team(self, t1: str, t2: str, score_1: int, score_2: int) -> None:
        pass

    @abstractmethod
    def update_player(self, players_1: List[str], players_2: List[str], score_1: int, score_2: int) -> None:
        pass

    @abstractmethod
    def predict_team_win_prob(self, t1: str, t2: str) -> float:
        pass

    @abstractmethod
    def predict_player_win_prob(self, players_1: List[str], players_2: List[str]) -> float:
        pass
