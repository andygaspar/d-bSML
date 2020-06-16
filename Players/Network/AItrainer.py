from Game.board import Board
from Players.player import Player
from Players.Network.record import Record
import numpy as np
import torch
from torch import nn, optim
from IPython import display


class AItrainer(Player):
    state: Board
    nextState: Board
    action: int
    rewardScored: float
    rewardOpponentScored: float
    rewardInvalidMove: float
    replayBuffer: list  # of Record
    score: int

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

    def get_move(self, state) -> int:

        if not self.invalid:
            self.state = state

            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # print('Device: {}'.format(device))
            # torch.manual_seed(41)
            n = self.boardsize
            N = (2 * n + 2) * n  # num samples = classes
            D = 1  # dimensions
            H = 100  # num hidden units
            epochs = 1000

            X = torch.from_numpy(state.vectorBoard).reshape(N, D).type(dtype=torch.float32)

            model = nn.Sequential(
                nn.Linear(D, H),
                nn.ReLU(),
                nn.Linear(H, 1))
            # model.to(device)
            # criterion = torch.autograd.Variable(torch.zeros(1), requires_grad=True)

            criterion = torch.nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

            for e in range(epochs):
                y_pred = model(X)
                loss = criterion
                if e % 100 == 0: print("[EPOCH]: {}, [LOSS]: {}".format(e, loss.item()))
                display.clear_output(wait=True)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.action = np.argmax(y_pred.detach().numpy())
            self.invalid = False
            print(model.forward(X))

            # validMoves = np.flatnonzero(state.vectorBoard == 0)
            # self.action = np.random.choice(validMoves)
            return self.action

        else:
            self.invalid = False
            validMoves = np.flatnonzero(state.vectorBoard == 0)
            self.action = np.random.choice(validMoves)
            return self.action

    def update(self, record: Record):
        self.replayBuffer.append(record)

    def scored(self, newPoints: int):
        self.score += newPoints
        self.update(Record(self.state, self.action, self.nextState, self.rewardScored))
        print("bravo, hai fatto punto")

    def opponentScored(self):
        self.update(Record(self.state, self.action, self.nextState, self.rewardOpponentScored))

    def invalidMove(self):
        self.invalid = True
        self.update(Record(self.state, self.action, self.nextState, self.rewardInvalidMove))

    def updateWeights(self):
        pass
