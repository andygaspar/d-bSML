import numpy as np
from Players.player import Player

class GreedyPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)

    def get_move(self, board: np.array) -> int:
        N = self.boardsize
        k = 0

        while k < len(board) - 2 * N - 2:
            for j in range(N):            
                open_edges = []
                if board[k] == 0:
                    open_edges.append(k)
                if board[k + 2*N + 1] == 0:
                    open_edges.append(k + 2*N + 1)
                if board[k + N] == 0:
                    open_edges.append(k + N)
                if board[k + N + 1] == 0:
                    open_edges.append(k + N + 1)

                if len(open_edges) == 1:
                    return open_edges[0]
                k += 1
            k += N + 1

        validMoves = np.flatnonzero(board == 0)
        return np.random.choice(validMoves)

    def scored(self, newPoints: int):
        self.score += newPoints

    def __str__(self):
        return "Greedy player"
