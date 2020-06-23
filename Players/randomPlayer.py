import numpy as np
from Players.player import Player


class RandomPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)

    def get_move(self, board: np.array) -> int:
        validMoves = np.flatnonzero(board == 0)
        return np.random.choice(validMoves)

    def scored(self, newPoints: int):
        self.score += newPoints

    def __str__(self):
        return "Random player"
