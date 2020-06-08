import numpy as np
from player import Player
from board import Board

class RandomPlayer(Player):

    def __init__(self, num: int, boardsize: int):
        super().__init__(num, boardsize)
        np.random.seed(2)

    def get_move(self, board: Board) -> int:
        validMoves = np.flatnonzero(board.vectorBoard == 0)
        return np.random.choice(validMoves)

    def scored(self):
        self.score += 1
        print("bravo, hai fatto punto")
