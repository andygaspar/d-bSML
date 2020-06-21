from GameTraining.Gym.AItrainer import AITrainer
import numpy as np


class InitialTrainer(AITrainer):

    def __init__(self, id_number: int, boardsize: int, hidden: int,
                 rewardNoScore: float, rewardScored: float, rewardOpponentScored: float, rewardInvalidMove: float,
                 rewardScoresInRow: float, rewardWinning: float, rewardLosing: float, only_valid: bool,
                 sample_size: int, capacity: int, gamma: float, numgames: int, eps_min: float, decay: float,
                 fixed_batch: bool = False, eps_greedy_value: float = 1., softmax: bool = False):
        super().__init__(id_number, boardsize, hidden, rewardNoScore, rewardScored, rewardOpponentScored,
                         rewardInvalidMove, rewardScoresInRow, rewardWinning, rewardLosing, only_valid,
                         sample_size, capacity, gamma, numgames, eps_min, decay, fixed_batch, eps_greedy_value, softmax)

    def get_move(self, state: np.array) -> int:
        self.state = state.copy()
        validMoves = np.flatnonzero(state == 0)
        self.action = np.random.choice(validMoves)
        return self.action

    def add_record(self, nextState: np.array, train: bool):
        self.replayMemory.add_record(self.state, self.action, nextState.copy(), self.current_reward)
        self.current_reward = 0

    def __str__(self):
        return "AI trainer player"


