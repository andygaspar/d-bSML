import numpy as np
import abc

from Game.board import Board


class Player:
    id: int  # player identifier
    score: int  # player score
    boardsize: int  # grid size

    def __init__(self, id: int, boardsize: int):
        self.id = id
        self.score = 0
        self.boardsize = boardsize

    @abc.abstractmethod
    def get_move(self, state: np.array):
        pass

    def no_score_move(self):
        pass

    def scored(self, newPoints: int):
        pass

    def opponentScored(self, newPoints: int):
        pass

    def invalidMove(self):
        pass

    def add_record(self, next_game_state: np.array, train: bool):
        pass

    def endGameReward(self, win: bool):
        pass

    def update_eps(self, i: int):
        pass
