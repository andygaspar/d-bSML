from Game.board import Board
from Players.player import Player
from GameTraining.Gym.replayMemory import ReplayMemory
from GameTraining.Gym.network import Network
import numpy as np


class AITrainer(Player):
    state: Board
    nextState: Board
    action: int
    rewardScored: float
    rewardOpponentScored: float
    rewardInvalidMove: float
    replayBuffer: list  # of Record
    score: int

    def __init__(self, id_number: int, boardsize: int, hidden: int, epochs: int, rewardScored: float,
                 rewardOpponentScored: float, rewardInvalidMove: float):
        super().__init__(id_number, boardsize)
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardScored = rewardScored
        self.rewardOpponentScored = rewardOpponentScored
        self.replayBuffer = []
        self.state = None
        self.nextState = None
        self.invalid = False
        self.network = Network(boardsize, hidden, epochs)
        self.replayMemory = ReplayMemory()

    def get_move(self, state) -> int:

        if not self.invalid:
            self.state = state
            return self.network.get_action(state)

        else:
            self.invalid = False
            validMoves = np.flatnonzero(state.vectorBoard == 0)
            self.action = np.random.choice(validMoves)
            return self.action

    def scored(self, newPoints: int):
        self.score += newPoints
        self.replayMemory.add_record(self.state, self.action, self.nextState, self.rewardScored)
        print("bravo, hai fatto punto")

    def opponentScored(self):
        self.replayMemory.add_record(self.state, self.action, self.nextState, self.rewardOpponentScored)

    def invalidMove(self):
        self.invalid = True
        self.replayMemory.add_record(self.state, self.action, self.nextState, self.rewardInvalidMove)

    def train_network(self):
        self.network.update_weights(self.replayMemory.get_sample())
