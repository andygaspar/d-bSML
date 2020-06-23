from Game.board import Board
from Players.player import Player
from GameTraining.Gym.replayMemory import ReplayMemory
from GameTraining.Gym import network
import numpy as np
import torch


class AIPlayer(Player):
    score: int
    invalid: bool
    network: network

    def __init__(self, id_number: int, boardsize: int, network: network):
        super().__init__(id_number, boardsize)
        self.score = 0
        self.invalid = False
        self.network = network
        self.network.network#.eval()
        self.otherPlayer = None

    def get_random_valid_move(self, state: np.array) -> int:
        self.invalid = False
        validMoves = np.flatnonzero(state == 0)
        return np.random.choice(validMoves)

    def get_move(self, state: np.array) -> int:
        if not self.invalid:
            return self.network.get_action(np.append(state, self.score_value()))
        else:
            #or RaiseError?
            return self.get_random_valid_move(state)

    def scored(self, newPoints: int):
        self.score += newPoints

    def invalidMove(self):
        self.invalid = True

    def score_value(self):
        return (self.score - self.otherPlayer.score) / self.boardsize ** 2

    def __str__(self):
        return "AI player"

