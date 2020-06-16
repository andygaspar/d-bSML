from Game.board import Board
from GameTraining.Gym.replayMemory import ReplayMemory
import numpy as np
import torch
from torch import nn, optim
from IPython import display


class Network:
    inputDimension: int  # dimensions
    hidden: int
    epochs: int

    def __init__(self, boardsize: int, hidden: int, epochs: int):
        self.inputDimension = (2 * boardsize + 2) * boardsize  # dimensions
        self.hidden = hidden
        self.epochs = epochs
        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.inputDimension))

    def get_action(self, state) -> int:
        X = torch.from_numpy(state.vectorBoard).reshape(1, self.inputDimension).type(dtype=torch.float32)
        action = np.argmax(self.network(X).detach().numpy().flatten())
        print(self.network.forward(X))
        return action

    def update_weights(self, batch):
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3, weight_decay=1e-5)
        states, actions, nextStates, rewards = batch
        X = torch.Tensor([el.vectorBoard.tolist() for el in states]).reshape(len(states), self.inputDimension)

        for e in range(self.epochs):
            q_current = self.network(X)
            q_target =
            loss = criterion
            if e % 100 == 0: print("[EPOCH]: {}, [LOSS]: {}".format(e, loss.item()))
            display.clear_output(wait=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
