from Players.player import Player
import numpy as np


class StupidPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)

    def get_move(self, board: np.array) -> int:
        validMoves = np.flatnonzero(board == 0)
        if np.random.rand() > 0.9:
            return np.random.choice(validMoves)
        return min(validMoves)

    def scored(self, newPoints: int):
        self.score += newPoints

    def __str__(self):
        return "Random player"