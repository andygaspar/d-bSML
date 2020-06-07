import numpy as np
from player import Player

class RandomPlayer(Player):

    def __init__(self,num,boardsize):
        super().__init__(num,boardsize)

    def get_move(self, board):
        validMoves = np.array([i for i in range(len(board.vectorBoard)) if board.vectorBoard[i]==0])
        return np.random.choice(validMoves)

    def scored(self):
        self.score += 1
        print("bravo, hai fatto punto")
