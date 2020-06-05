import numpy as nu
from player import Player

class HumanPlayer(Player):

    def __init__(self,num):
        super().__init__(num)

    def get_move(self):
        move=[]
        print("tocca a te giocatore ", self.num)
        orizontalOrVertical = input("o for orizontal, any key for vertical ")
        if orizontalOrVertical == "o":
            move.append(True)
        else:
            move.append(False)

        row = int(input("row "))
        move.append(row)
        col = int(input("col "))
        move.append(col)

        return move

    def scored(self):
        self.score += 1

    def opponentScored(self):
        print("al momento sti cazzi, serve per segnelare la penalty al renforcement nel caso AI")