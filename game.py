import numpy as np
from board import Board
from player import Player
from AIPlayer import AIPlayer
from humanPlayer import HumanPlayer
from randomPlayer import RandomPlayer


class Game:

    def __init__(self, AIvsHuman: bool = False, boardsize: int = 4):
        self.board = Board(boardsize)
        self.numBoxes = 0
        if not AIvsHuman: self.players = [RandomPlayer(1, boardsize), RandomPlayer(2, boardsize)]
        else: self.players = [RandomPlayer(1, boardsize), AIPlayer(2, boardsize)]

    def is_valid(self, idx: int) -> bool:
        return not self.board.vectorBoard[idx]  # (1==True gives False, 0 == False gives True)

    def play(self):

        currentPlayer = None
        otherPlayer = None
        turn = 0
        PlayerTurn = -1
        N = self.board.size
        newNumBoxes = 0

        self.board.print_board()

        while turn < (2 * N + 2) * N:
            if newNumBoxes - self.numBoxes == 0:
                PlayerTurn += 1
                currentPlayer = self.players[PlayerTurn % 2]
                otherPlayer = self.players[(PlayerTurn + 1) % 2]
            else:
                currentPlayer.scored(newNumBoxes-self.numBoxes)
                self.numBoxes = newNumBoxes
                otherPlayer.opponentScored()


            move = currentPlayer.get_move(self.board)

            while not self.is_valid(move):
                currentPlayer.invalidMove()
                move = currentPlayer.get_move(self.board)
                print("Invalid Move")

            self.board.set_board(move)
            turn += 1

            newNumBoxes = self.board.count_boxes()
            print(newNumBoxes)
            self.board.print_board()

        print("Players score: " + str([p.score for p in self.players]))

g = Game(True, 3)
g.play()