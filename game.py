import numpy as np
from board import Board
from player import Player
from humanPlayer import HumanPlayer


class Game:

    def __init__(self,AIvsHuman = False, boardsize=4):
        self.board=Board(boardsize)
        self.numBoxes=0
        self.players = [HumanPlayer(1,boardsize),HumanPlayer(2,boardsize)]


    def is_valid(self, move):
        if self.board.vectorBoard[move] == 1:
            return False
        return True


    def play(self):

        currentPlayer = None
        otherPlayer = None
        turn = 0
        currentTurn = -1
        N = self.board.size
        while turn <    6: # N**(N+1)*2:
            newNumBoxes = self.board.count_boxes()

            if newNumBoxes-self.numBoxes == 0:
                currentTurn += 1
                currentPlayer = self.players[currentTurn%2]
                otherPlayer = self.players[(currentTurn+1)%2]
            else:
                currentPlayer.scored()
                self.boxes = newNumBoxes
                otherPlayer.opponentScored()

            self.board.print_board()
            
            move = currentPlayer.get_move()

            while not self.is_valid(move):
                currentPlayer.invalidMove()
                move = currentPlayer.get_move()

            self.board.set_board(move)
            turn += 1
        
        self.board.print_board()


g=Game()
g.play()
