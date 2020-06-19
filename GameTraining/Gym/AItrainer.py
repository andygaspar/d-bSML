from Game.board import Board
from Players.player import Player
from GameTraining.Gym.replayMemory import ReplayMemory
from GameTraining.Gym.network import Network, NetworkOnlyValid
import numpy as np
from GameTraining.Gym.replayMemory import ReplayMemory


class AITrainer(Player):
    state: np.array
    action: int
    rewardNoScore: float
    rewardScored: float
    rewardOpponentScored: float
    rewardInvalidMove: float
    score: int
    replayMemory: ReplayMemory
    current_reward: int
    gamma: float

    def __init__(self, id_number: int, boardsize: int, hidden: int, epochs: int,
                 rewardNoScore: float, rewardScored: float, rewardOpponentScored: float, rewardInvalidMove: float,
                 use_invalid: bool, sample_size: int, capacity: int, gamma: float):

        super().__init__(id_number, boardsize)
        self.rewardNoScore = rewardNoScore
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardScored = rewardScored
        self.rewardOpponentScored = rewardOpponentScored
        self.state = None
        self.action = None
        self.invalid = False
        self.network = Network(boardsize, hidden, epochs) \
            if use_invalid == True else NetworkOnlyValid(boardsize, hidden, epochs)
        self.replayMemory = ReplayMemory(sample_size, capacity)
        self.current_reward = 0
        self.gamma = gamma

    def get_move(self, state: np.array) -> int:
        self.state = state.copy()

        if not self.invalid:
            self.action = self.network.get_action(state)
            return self.action
        else:
            self.invalid = False
            validMoves = np.flatnonzero(state == 0)
            self.action = np.random.choice(validMoves)
            return self.action

    def no_score_move(self):
        self.current_reward += self.rewardNoScore*0.5

    def scored(self, newPoints: int):
        self.score += newPoints + newPoints**0.5
        self.current_reward += self.rewardScored

    def opponentScored(self, newPoints: int):
        self.score += newPoints/2 + (newPoints/2)**0.5
        self.current_reward += self.rewardOpponentScored

    def invalidMove(self):
        self.invalid = True
        self.current_reward += self.rewardInvalidMove

    def endGameReward(self, win: bool):
        if win:
            self.current_reward = 20
        else:
            self.current_reward = -20


    def add_record(self, nextState: np.array, train: bool):
        self.replayMemory.add_record(self.state, self.action, nextState.copy(), self.current_reward)
        if train:
            self.train_network()
        self.current_reward = 0

    def train_network(self):
        if self.replayMemory.size < self.replayMemory.sampleSize:
            return
        self.network.update_weights(self.replayMemory.get_sample(), self.gamma)

    def __str__(self):
        return "AI trainer player"
