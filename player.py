import numpy as np

class Player:
    id: int  # player identifier
    score: int  # player score
    boardsize: int  # grid size

    def __init__(self, id: int, boardsize: int):
        self.id = id
        self.score = 0
        self.boardsize = boardsize

    def scored(self):
        self.score += 1

    def opponentScored(self):
        print("al momento sti cazzi, serve per segnelare la penalty al renforcement nel caso AI")

    def invalidMove(self):
        print("sempre sti cazzi, da dire alla AI che non sa ancora giocare, che si vergogni")
        raise Exception()
