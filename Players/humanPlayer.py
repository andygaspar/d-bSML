import sys
from Players.player import Player
import numpy as np
print_err = sys.stderr.write


class HumanPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)

    def get_move(self, board: np.array) -> int:

        print("Tocca a te, caro giocatore ", self.id)

        N = self.boardsize

        while True:
            orizontal = (input("o for orizontal, any key for vertical ") == "o")
            row = int(input("row idx, start from 0: "))
            col = int(input("col idx, start from 0:"))
            if orizontal:
                vec_idx = row * (2 * N + 1) + col
                if (row in range(0, N + 1)) and (col in range(0, N)):
                    if board[vec_idx] == 0:
                        return vec_idx
                    else:
                        print_err("Move already chosen, please try again\n")
                else:
                    print_err("Invalid move, please try again\n")
            else:
                vec_idx = row * (2 * N + 1) + col + N
                if (row in range(0, N)) and (col in range(0, N + 1)):
                    if board[vec_idx] == 0:
                        return vec_idx
                    else:
                        print_err("Move already chosen, please try again\n")
                else:
                    print_err("Invalid move, try again\n")

    def scored(self, newPoints: int):
        self.score += newPoints
        print("bravo, hai fatto punto")

    def __str__(self):
        return "Human player"

    def invalidMove(self):
        print("sempre sti cazzi, da dire alla AI che non sa ancora giocare, che si vergogni")
        raise Exception()