import numpy as np
from player import Player
from board import Board

class HumanPlayer(Player):

    def __init__(self, id: int, boardsize: int):
        super().__init_(id, boardsize)

    def get_move(self, board: Board) -> int:

        print("tocca a te, caro giocatore ", self.id)

        orizontalOrVertical = input("o for orizontal, any key for vertical ")
        row = int(input("row "))
        col = int(input("col "))
        

        N = self.boardsize
        if orizontalOrVertical == "o":
            return row * (2 * N + 1) + col
        else:
            return row * (2 * N + 1) + col + N

    def scored(self):
        self.score += 1
        print("bravo, hai fatto punto")
