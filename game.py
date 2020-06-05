import numpy as np
from board import Board
from player import Player
from humanPlayer import HumanPlayer


class Game:

    def __init__(self,AIvsHuman = False, boardsize=4):
        self.board=Board(boardsize)
        self.board.print_board()
        self.boxes=0
        self.players = [HumanPlayer(1),HumanPlayer(2)]


    def is_valid(self, is_row, r, c):
        N = self.board.size
        if is_row == True:
            if (r>=N+1 or c>=N or self.board.rows[r,c]==1):
                return False

        if is_row == False:
            if (r>=N or c>=N+1 or self.board.cols[r,c]==1):
                return False

        return True

    def count_boxes(self):
        new_num_boxes=0
        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.rows[i,j]==1 and self.board.rows[i+1,j]==1 and self.board.cols[i,j]==1 and self.board.cols[i,j+1]==1:
                    new_num_boxes+=1
        return new_num_boxes

    def set_move(self,is_row,r,c):
        if self.is_valid(is_row,r,c):
            self.board.set_board(is_row,r,c)
        else:
            print("non valida, e implenta per piacere il sengale alla AI")

    def play(self):
        currentPlayer = None
        turn = 0
        currentTurn = 0
        N = self.board.size
        while turn <    3: # N**(N+1)*2:
            self.board.print_board()
            if self.count_boxes()-self.boxes == 0:
                currentPlayer = self.players[currentTurn%2]
                otherPlayer = self.players[(currentTurn+1)%2]
                move = currentPlayer.get_move()
                while not self.is_valid(move[0],move[1],move[2]):
                    move = currentPlayer.get_move()
                self.set_move(move[0],move[1],move[2])
                currentTurn += 1
            else:
                currentPlayer.scored()
                otherPlayer.opponentScored()

            turn += 1


g=Game()
g.play()
