import numpy as np
from board import Board
from player import Player
from humanPlayer import HumanPlayer
from randomPlayer import RandomPlayer


class Game:

    def __init__(self, AIvsHuman: bool = False, boardsize: int = 4):
        self.board = Board(boardsize)
        self.numBoxes = 0
        self.players = [RandomPlayer(1, boardsize), RandomPlayer(2, boardsize)]

    def is_valid(self, idx: int) -> bool:
        return not self.board.vectorBoard[idx]  # (1==True gives False, 0 == False gives True)

    def play(self):

        currentPlayer = None
        otherPlayer = None
        turn = 0
        currentTurn = -1
        N = self.board.size
        newNumBoxes = 0
        while turn < (2 * N + 2) * N:

            if newNumBoxes - self.numBoxes == 0:
                currentTurn += 1
                currentPlayer = self.players[currentTurn % 2]
                otherPlayer = self.players[(currentTurn + 1) % 2]
            else:
                currentPlayer.scored()
                self.numBoxes = newNumBoxes
                otherPlayer.opponentScored()

            self.board.print_board()

            move = currentPlayer.get_move(self.board)

            while not self.is_valid(move):
                currentPlayer.invalidMove()
                move = currentPlayer.get_move(self.board)

            self.board.set_board(move)
            turn += 1

            newNumBoxes = self.board.count_boxes()

        self.board.print_board()


g = Game(False, 2)
g.play()