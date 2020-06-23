from GameTraining.Gym.StateMultiplier import StateMultiplier
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
    eps_min: float
    decay: float
    double_q_interval: int
    double_q_counter: int

    def __init__(self, id_number: int, boardsize: int, hidden: int, rewardScored: float, rewardInvalidMove: float,
                 rewardWinning: float, rewardLosing: float, only_valid: bool, sample_size: int, capacity: int,
                 gamma: float, numgames: int, eps_min: float, eps_decay: float, fixed_batch: bool = False,
                 softmax: bool = False, double_q_interval: int = 0):

        super().__init__(id_number, boardsize)
        self.rewardNoScore = 0
        self.rewardInvalidMove = rewardInvalidMove
        self.rewardScored = rewardScored
        self.rewardWinning = rewardWinning
        self.rewardLosing = rewardLosing
        self.model_network = Network(boardsize, hidden, only_valid, softmax)
        self.target_network = Network(boardsize, hidden, only_valid, softmax)
        self.state = None
        self.stateScore = None
        self.action = None
        self.invalid = False
        self.replayMemory = ReplayMemory(sample_size, capacity)
        self.current_reward = 0
        self.score = 0
        self.gamma = gamma
        self.fixed_batch = fixed_batch
        self.eps_greedy_value = 1.
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.softmax = softmax
        self.numgames = numgames
        self.double_q_interval = double_q_interval
        self.double_q_counter = 0

    def get_random_valid_move(self, state: np.array) -> int:
        self.invalid = False
        validMoves = np.flatnonzero(state == 0)
        self.action = np.random.choice(validMoves)
        return self.action

    def get_move(self, state: np.array) -> int:
        self.state = state.copy()
        self.stateScore = self.score_value()
        if np.random.rand() > self.eps_greedy_value:
            if not self.invalid:
                self.action = self.model_network.get_action(self.state_with_score(self.state))
                return self.action
            else:
                return self.get_random_valid_move(state)
        else:
            return self.get_random_valid_move(state)

    def update_eps(self, iteration: int):
        self.eps_greedy_value = self.eps_min + (1 - self.eps_min) * np.exp(- self.eps_decay * iteration)

    def no_score_move(self):
        self.rewardScoresInRow = 0
        self.current_reward += self.rewardNoScore

    def scored(self, newPoints: int):
        self.score += newPoints
        self.current_reward += newPoints * self.rewardScored

    def opponentScored(self, newPoints: int):
        self.current_reward -= newPoints * self.rewardScored

    def invalidMove(self):
        self.invalid = True
        self.current_reward += self.rewardInvalidMove

    def endGameReward(self, win: bool):
        if win:
            self.current_reward += self.rewardWinning
        else:
            self.current_reward += self.rewardLosing

    def add_record(self, nextState: np.array, done: bool):
        self.store_in_memory(self.state, nextState, self.action, done)
        st = self.state.copy()
        nx_st = nextState.copy()
        act = self.action
        for i in range(3):
            st, nx_st, act = StateMultiplier.rotate(st, nx_st, act)
            self.store_in_memory(st, nx_st, act, done)

            st_ref, nx_st_ref, act_ref = StateMultiplier.reflect(st, nx_st, act)
            self.store_in_memory(st_ref, nx_st_ref, act_ref, done)

        st, nx_st, act = StateMultiplier.reflect(self.state, nextState, self.action)
        self.store_in_memory(st, nx_st, act, done)
        self.current_reward = 0

    def train_model_network(self):
        if self.replayMemory.size < self.replayMemory.sampleSize:
            return
        for i in range(5):
            self.model_network.update_weights(self.replayMemory.get_sample(), self.gamma, self.target_network)
        self.double_q_counter += 1

        if self.double_q_interval == 0:
            return
        if self.double_q_counter % self.double_q_interval == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.take_weights(self.model_network)

    def get_trained_player(self, id_number: int) -> AIPlayer:
        trained_network = Network(self.boardsize, self.model_network.hidden,
                                  self.model_network.only_valid_actions, self.model_network.softmax)
        trained_network.take_weights(self.model_network)
        return AIPlayer(id_number, self.boardsize, trained_network)

    def score_value(self):
        return self.score / self.boardsize ** 2

    def store_in_memory(self, state: np.array, nextState: np.array, action: int, done: bool):
        if self.fixed_batch:
            if self.replayMemory.size < self.replayMemory.sampleSize:
                self.replayMemory.add_record(self.state_with_score(state), action,
                                             self.next_state_with_score(nextState), self.current_reward, done)
        else:
            self.replayMemory.add_record(self.state_with_score(state), action,
                                         self.next_state_with_score(nextState), self.current_reward, done)
        self.current_reward = 0

    def state_with_score(self, state: np.array):
        return np.append(state, self.stateScore)

    def next_state_with_score(self, nextState: np.array):
        return np.append(nextState, self.score_value())

    def print_board(self, vect: np.array) -> str:
        k = 0
        N = self.boardsize
        for i in range(N):
            orizontal = ""
            vertical = ""
            for j in range(N):

                if vect[k] == 0:
                    orizontal += "  __ "
                else:
                    orizontal += "  ** "
                k += 1

            for j in range(N + 1):
                if vect[k] == 0:
                    vertical += "|    "
                else:
                    vertical += "*    "
                k += 1
            print(orizontal)
            print(vertical)
            print(vertical)

        orizontal = ""
        for j in range(N):
            if vect[k] == 0:
                orizontal += "  __ "
            else:
                orizontal += "  ** "
            k += 1
        print(orizontal, "\n\n")

    def __str__(self):
        return "AI trainer player"
