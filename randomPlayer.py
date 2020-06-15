import numpy as np
from player import Player
from board import Board


class RandomPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)
        #np.random.seed(2)

    def get_move(self, board: Board) -> int:
        validMoves = np.flatnonzero(board.vectorBoard == 0)
        return np.random.choice(validMoves)

    def scored(self, newPoints: int):
        self.score += newPoints
        print("bravo, hai fatto punto")
