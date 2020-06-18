import numpy as np
from Players.player import Player
from Game.board import Board


class RandomPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)
        # np.random.seed(2)

    def get_move(self, state: np.array) -> int:
        validMoves = np.flatnonzero(state == 0)
        return np.random.choice(validMoves)

    def scored(self, newPoints: int):
        self.score += newPoints
        print("bravo, hai fatto punto")

    def __str__(self):
        return "Random player"
