from Players.player import Player
from GameTraining.Gym.network import Network
import numpy as np
from GameTraining.Gym.replayMemory import ReplayMemory


class AITrainer(Player):
    rewardNoScore: float
    rewardScored: float
    rewardOpponentScored: float
    rewardInvalidMove: float
    rewardScoresInRow: float
    rewardWinning: float
    rewardLosing: float
    state: np.array
    action: int
    invalid: bool
    network: Network.network
    replayMemory: ReplayMemory
    current_reward: int
    score: int
    gamma: float
    fixed_batch: bool
    softmax: bool

    def __init__(self, id_number: int, boardsize: int, hidden: int,
                 rewardNoScore: float, rewardScored: float, rewardOpponentScored: float, rewardInvalidMove: float,
                 rewardScoresInRow: float, rewardWinning: float, rewardLosing: float, only_valid: bool, sample_size: int, capacity: int, gamma: float, limited_batch: bool = False, softmax: bool = False):

        super().__init__(id_number, boardsize)
        self.rewardNoScore = rewardNoScore
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardScored = rewardScored
        self.rewardOpponentScored = rewardOpponentScored
        self.rewardScoresInRow = rewardScoresInRow
        self.rewardWinning = rewardWinning
        self.rewardLosing - rewardLosing
        self.network = Network(boardsize, hidden, only_valid, softmax)
        self.state = None
        self.action = None
        self.invalid = False
        self.replayMemory = ReplayMemory(sample_size, capacity)
        self.current_reward = 0
        self.score = 0
        self.gamma = gamma
        self.fixed_batch = limited_batch
        self.softmax = softmax

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
        self.rewardScoresInRow = 0
        self.current_reward += self.rewardNoScore

    def scored(self, newPoints: int):
        self.score += newPoints
        self.current_reward += newPoints*self.rewardScored + self.rewardScoresInRow*self.rewardScored
        if newPoints > 1:
            self.current_reward += newPoints * self.rewardScored
        self.rewardScoresInRow += 1

    def opponentScored(self, newPoints: int):
        self.score += newPoints
        self.current_reward += (newPoints*self.rewardOpponentScored + self.rewardScoresInRow*self.rewardOpponentScored)/2
        self.current_reward += self.rewardOpponentScored

    def invalidMove(self):
        self.invalid = True
        self.current_reward += self.rewardInvalidMove

    def endGameReward(self, win: bool):
        if win:
            self.current_reward += 100
        else:
            self.current_reward += -100

    def add_record(self, nextState: np.array, train: bool):
        if self.fixed_batch:
            if self.replayMemory.size < self.replayMemory.sampleSize:
                self.replayMemory.add_record(self.state, self.action, nextState.copy(), self.current_reward)
        else:
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
