from Players.AIPlayer import AIPlayer
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
    model_network: Network
    target_network: Network
    hidden: int
    replayMemory: ReplayMemory
    current_reward: int
    score: int
    gamma: float
    fixed_batch: bool
    eps_greedy_value: float
    softmax: bool
    numgames: int
    eps_min: int
    decay: float
    double_Q_learning: bool


    def __init__(self, id_number: int, boardsize: int, hidden: int,
                 rewardNoScore: float, rewardScored: float, rewardOpponentScored: float, rewardInvalidMove: float,
                 rewardScoresInRow: float, rewardWinning: float, rewardLosing: float, only_valid: bool,
                 sample_size: int, capacity: int, gamma: float, numgames: int, eps_min: float, decay: float,
                 fixed_batch: bool = False, eps_greedy_value: float = 0., softmax: bool = False,
                 double_Q_learning: bool = False):

        super().__init__(id_number, boardsize)
        self.rewardNoScore = rewardNoScore
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardScored = rewardScored
        self.rewardOpponentScored = rewardOpponentScored
        self.rewardScoresInRow = rewardScoresInRow
        self.rewardWinning = rewardWinning
        self.rewardLosing = rewardLosing
        self.model_network = Network(boardsize, hidden, only_valid, softmax)
        self.target_network = Network(boardsize, hidden, only_valid, softmax)
        self.state = None
        self.action = None
        self.invalid = False
        self.replayMemory = ReplayMemory(sample_size, capacity)
        self.current_reward = 0
        self.score = 0
        self.gamma = gamma
        self.fixed_batch = fixed_batch
        self.eps_greedy_value = eps_greedy_value
        self.eps_min = eps_min
        self.decay = decay
        self.softmax = softmax
        self.numgames = numgames
        self.double_Q_learning = double_Q_learning

    def get_random_valid_move(self, state: np.array) -> int:
        self.invalid = False
        validMoves = np.flatnonzero(state == 0)
        self.action = np.random.choice(validMoves)
        return self.action

    def get_move(self, state: np.array) -> int:
        self.state = state.copy()
        if np.random.rand() > self.eps_greedy_value:
            if not self.invalid:
                self.action = self.model_network.get_action(state)
                return self.action
            else:
                return self.get_random_valid_move(state)
        else:
            return self.get_random_valid_move(state)

    def update_eps(self, iteration: int):
        self.eps_greedy_value = self.eps_min + (1 - self.eps_min) * np.exp(- self.decay * iteration)

    def no_score_move(self):
        self.rewardScoresInRow = 0
        self.current_reward += self.rewardNoScore

    def scored(self, newPoints: int):
        self.score += newPoints
        bonus_points_in_a_row = self.rewardScoresInRow * self.rewardScored
        self.current_reward += newPoints * self.rewardScored + bonus_points_in_a_row
        if newPoints > 1:
            bonus_multiple_points_at_once = newPoints * self.rewardScored
            self.current_reward += bonus_multiple_points_at_once
        self.rewardScoresInRow += 1

    def opponentScored(self, newPoints: int):
        bonus_points_in_a_row = self.rewardScoresInRow * self.rewardOpponentScored
        self.current_reward += newPoints * self.rewardOpponentScored + bonus_points_in_a_row
        if newPoints > 1:
            bonus_multiple_points_at_once = newPoints * self.rewardOpponentScored
            self.current_reward += bonus_multiple_points_at_once
        self.rewardScoresInRow += 1

    def invalidMove(self):
        self.invalid = True
        self.current_reward += self.rewardInvalidMove

    def endGameReward(self, win: bool):
        if win:
            self.current_reward += self.rewardWinning
        else:
            self.current_reward += - self.rewardLosing

    def add_record(self, nextState: np.array):
        if self.fixed_batch:
            if self.replayMemory.size < self.replayMemory.sampleSize:
                self.replayMemory.add_record(self.state, self.action, nextState.copy(), self.current_reward)
        else:
            self.replayMemory.add_record(self.state, self.action, nextState.copy(), self.current_reward)
        self.current_reward = 0

    def train_model_network(self):
        if self.replayMemory.size < self.replayMemory.sampleSize:
            return
        self.model_network.update_weights(self.replayMemory.get_sample(), self.gamma, self.target_network)

    def update_target_network(self):
        self.target_network.take_weights(self.model_network)

    def get_trained_player(self, id_number: int) -> AIPlayer:
        return AIPlayer(id_number, self.boardsize, self.model_network)

    def __str__(self):
        return "AI trainer player"
