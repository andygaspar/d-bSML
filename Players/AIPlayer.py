from Game.board import Board
from Players.player import Player
from GameTraining.Gym.replayMemory import ReplayMemory
from GameTraining.Gym import network
import numpy as np
import torch


class AIPlayer(Player):
    state: Board
    nextState: Board
    action: int
    rewardScored: float
    rewardOpponentScored: float
    rewardInvalidMove: float
    replayBuffer: list  # of Record
    score: int
    network: network

    def __init__(self, id_number: int, boardsize: int, hidden: int):
        super().__init__(id_number, boardsize)

        self.invalid = False
        self.network = network.Network(boardsize, hidden)
        self.network.network.load_state_dict(torch.load('prova.pt'))
        self.network.network.eval()

    def get_move(self, state: np.array) -> int:

        if not self.invalid:
            self.state = state
            return self.network.get_action(state)

        else:
            self.invalid = False
            validMoves = np.flatnonzero(state == 0)
            self.action = np.random.choice(validMoves)
            return self.action

    def scored(self, newPoints: int):
        self.score += newPoints

    def invalidMove(self):
        self.invalid = True

    def __str__(self):
        return "AI player"

