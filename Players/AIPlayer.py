from Game.board import Board
from Players.player import Player
from GameTraining.Gym.replayMemory import ReplayMemory
from GameTraining.Gym import network
import numpy as np


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

    def __init__(self, id_number: int, boardsize: int, rewardScored: float = 10,
                 rewardOpponentScored: float = -10, rewardInvalidMove: float = -100):
        super().__init__(id_number, boardsize)
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardScored = rewardScored
        self.rewardOpponentScored = rewardOpponentScored
        self.replayBuffer = []
        self.state = None
        self.nextState = None
        self.invalid = False
        self.network = network(boardsize, rewardScored, rewardOpponentScored, rewardInvalidMove)

    def get_move(self, state) -> int:

        if not self.invalid:
            self.state = state
            return self.network.get_action(state)

        else:
            self.invalid = False
            validMoves = np.flatnonzero(state.vectorBoard == 0)
            self.action = np.random.choice(validMoves)
            return self.action

    def update(self, record: ReplayMemory):
        self.replayBuffer.append(record)

    def scored(self, newPoints: int):
        self.score += newPoints

    def opponentScored(self, newPoints: int):
        pass

    def invalidMove(self):
        self.invalid = True

    def __str__(self):
        return "AI player"

