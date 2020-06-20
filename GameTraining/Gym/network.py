from Game.board import Board
from GameTraining.Gym.replayMemory import ReplayMemory
import numpy as np
import torch
from torch import nn, optim
from IPython import display


class Network:
    inputDimension: int  # dimensions
    hidden: int

    def __init__(self, boardsize: int, hidden: int):
        self.loss = 0
        self.inputDimension = (2 * boardsize + 2) * boardsize  # dimensions
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, self.inputDimension)
        )

    def get_action(self, state: np.array, only_valid: bool) -> int:
        X = torch.from_numpy(state).reshape(1, self.inputDimension).type(dtype=torch.float32) / len(state)
        with torch.no_grad():
            Q_values = self.network(X)
        action = torch.argmax(torch.flatten(Q_values)).item()

        if only_valid:
            while state[action] == 1:
                Q_values[0, action] = torch.min(Q_values).item() - 10
                action = torch.argmax(torch.flatten(Q_values)).item()
        return action

    def update_weights(self, batch: tuple, gamma: float):
        criterion = torch.nn.MSELoss()

        optimizer = optim.Adam(self.network.parameters(), lr=1e-4, weight_decay=1e-5)
        # optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)

        states, actions, nextStates, rewards = batch
        X = torch.tensor([el.tolist() for el in states]).reshape(len(states), self.inputDimension) / len(states[0])
        X_next = torch.tensor([el.tolist() for el in nextStates]).reshape(len(nextStates), self.inputDimension) / len(states[0])

        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        curr_Q = self.network(X).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.network(X_next)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + gamma * max_next_Q

        loss = criterion(curr_Q, expected_Q.detach())
        self.loss = loss.item()
        display.clear_output(wait=True)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        optimizer.step()

    def save_weights(self):
        torch.save(self.network.state_dict(), 'prova.pt')
