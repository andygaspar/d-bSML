import numpy as np
from GameTraining.Gym.AItrainer import AITrainer
from GameTraining.Gym.StateMultiplier import StateMultiplier


class GreedyPlayerWithMemory(AITrainer):

    def __init__(self, id: int, boardsize: int, hidden, rewardScored: float, rewardInvalidMove: float,
                 rewardWinning: float, rewardLosing: float, only_valid: bool, sample_size: int, capacity: int,
                 gamma: float, numgames: int, eps_min: float, eps_decay: float, fixed_batch: bool = False,
                 softmax: bool = False, double_q_interval: int = 0):
        super().__init__(id, boardsize, hidden, rewardScored, rewardInvalidMove, rewardWinning, rewardLosing,
                         only_valid, sample_size, capacity, gamma, numgames, eps_min, eps_decay, fixed_batch,
                         softmax, double_q_interval=double_q_interval)

        self.save = False

    def get_move(self, board: np.array) -> int:
        self.state = board.copy()
        self.stateScore = self.score_value()

        N = self.boardsize
        k = 0

        while k < len(board) - 2 * N - 2:
            for j in range(N):
                open_edges = []
                if board[k] == 0:
                    open_edges.append(k)
                if board[k + 2 * N + 1] == 0:
                    open_edges.append(k + 2 * N + 1)
                if board[k + N] == 0:
                    open_edges.append(k + N)
                if board[k + N + 1] == 0:
                    open_edges.append(k + N + 1)

                if len(open_edges) == 1:
                    return open_edges[0]
                k += 1
            k += N + 1

        validMoves = np.flatnonzero(board == 0)
        self.action = np.random.choice(validMoves)
        return self.action

    def add_record(self, nextState: np.array, done: bool):
        if self.save:
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
        return

    def update_target_network(self):
        return

    def __str__(self):
        return "Greedy player"
