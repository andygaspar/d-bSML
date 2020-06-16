import sys
from Players.player import Player
from Game.board import Board
print_err = sys.stderr.write


class HumanPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init__(id, boardsize)

    def get_move(self, board: Board) -> int:

        print("tocca a te, caro giocatore ", self.id)

        N = self.boardsize

        while True:
            orizontal = (input("o for orizontal, any key for vertical ") == "o")
            row = int(input("row idx, start from 0: "))
            col = int(input("col idx, start from 0:"))
            if orizontal:
                vec_idx = row * (2 * N + 1) + col
                if (row in range(0, N + 1)) and (col in range(0, N)):
                    if board.vectorBoard[vec_idx] == 0:
                        return vec_idx
                    else:
                        print_err("Move already chosen, please try again\n")
                else:
                    print_err("Invalid Move, please try again\n")
            else:
                vec_idx = row * (2 * N + 1) + col + N
                if (row in range(0, N)) and (col in range(0, N + 1)):
                    if board.vectorBoard[vec_idx] == 0:
                        return vec_idx
                    else:
                        print_err("Move already chosen, please try again\n")
                else:
                    print_err("Invalid move, try again\n")

    def scored(self, newPoints: int):
        self.score += newPoints
        print("bravo, hai fatto punto")
