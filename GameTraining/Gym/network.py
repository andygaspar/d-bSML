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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: {}'.format(self.device))
        self.loss = 0
        self.inputDimension = (2 * boardsize + 2) * boardsize  # dimensions
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, self.inputDimension*2),
            nn.ReLU(),
            nn.Linear(self.inputDimension*2, self.inputDimension*3),
            nn.ReLU(),
            nn.Linear(self.inputDimension*3, self.inputDimension*2),
            nn.ReLU(),
            nn.Linear(self.inputDimension*2, self.inputDimension),
            nn.ReLU(),
            nn.Linear(self.inputDimension, self.inputDimension),
        )
        self.network.to(self.device)

    def get_action(self, state: np.array) -> int:
        X = torch.from_numpy(state).to(self.device).reshape(1, self.inputDimension).type(dtype=torch.float32) / len(state)
        with torch.no_grad():
            Q_values = self.network(X)
        return torch.argmax(torch.flatten(Q_values)).item()

    def update_weights(self, batch: tuple, gamma: float):
        criterion = torch.nn.MSELoss()

        optimizer = optim.Adam(self.network.parameters(), lr=1e-4, weight_decay=1e-5)
        # optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)

        states, actions, nextStates, rewards = batch
        X = torch.tensor([el.tolist() for el in states]).to(self.device).reshape(len(states), self.inputDimension) / len(states[0])
        X_next = torch.tensor([el.tolist() for el in nextStates]).to(self.device).reshape(len(nextStates), self.inputDimension) / len(states[0])

        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)

        curr_Q = self.network(X).gather(1, actions.unsqueeze(1)).to(self.device)
        curr_Q = curr_Q.squeeze(1).to(self.device)
        next_Q = self.network(X_next).to(self.device)
        max_next_Q = torch.max(next_Q, 1)[0].to(self.device)
        expected_Q = (rewards + gamma * max_next_Q).to(self.device)

        loss = criterion(curr_Q, expected_Q.detach())
        self.loss = loss.item()
        display.clear_output(wait=True)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        optimizer.step()


class NetworkOnlyValid(Network):

    def __init__(self, boardsize: int, hidden: int):
        super().__init__(boardsize, hidden)

    def get_action(self, state: np.array) -> int:
        X = torch.from_numpy(state).to(self.device).reshape(1, self.inputDimension).type(dtype=torch.float32) / len(state)
        q_values = self.network(X).to(self.device)
        action = torch.argmax(torch.flatten(q_values)).item()

        while state[action] == 1:
            q_values[0, action] = torch.min(q_values).item() - 10
            action = torch.argmax(torch.flatten(q_values)).item()
        return action

    def save_weights(self):
        torch.save(self.network.state_dict(), 'prova.pt')
