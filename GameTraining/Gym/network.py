import numpy as np
import torch
from torch import nn, optim
from IPython import display


class Network:
    device: torch.device
    inputDimension: int  # dimensions
    hidden: int
    network: torch.nn.Sequential
    only_valid_actions: bool
    softmax: bool

    def __init__(self, boardsize: int, hidden: int, ony_valid_actions: bool, softmax: bool):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss = [0]
        self.inputDimension = (2 * boardsize + 2) * boardsize + 1
        self.outputDimension = (2 * boardsize + 2) * boardsize
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, 2 * self.hidden),
            nn.ReLU(),
            # nn.Linear(self.inputDimension * 2, self.inputDimension * 3),
            # nn.ReLU(),
            # # nn.Linear(self.inputDimension * 3, self.inputDimension * 2),
            # # nn.ReLU(),
            # # nn.Linear(self.inputDimension * 2, self.inputDimension),
            # nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.outputDimension),
        )
        self.network.to(self.device)
        torch.cuda.current_device()
        print(torch.cuda.is_available())
        self.only_valid_actions = ony_valid_actions
        self.softmax = softmax
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.optimizer = optim.SGD(self.network.parameters(), lr=1e-2, momentum=0.9)

    def sample_action(self, Q_values: torch.tensor) -> int:
        if self.softmax:
            Q_values = Q_values.softmax(0).flatten().cpu().numpy()
            Q_values_probabilities = Q_values / np.sum(Q_values)
            action = np.random.choice(range(len(Q_values_probabilities)), size=1, p=Q_values_probabilities)[0]
            return action
        else:
            return torch.argmax(torch.flatten(Q_values)).item()

    def get_action(self, state: np.array) -> int:
        X = torch.from_numpy(state).to(self.device).reshape(1, self.inputDimension).type(dtype=torch.float32)
        with torch.no_grad():
            Q_values = torch.flatten(self.network(X)).to(self.device)
        action = self.sample_action(Q_values)

        if self.only_valid_actions:
            while state[action] == 1:
                Q_values[action] = torch.min(Q_values).item() - 10
                action = self.sample_action(Q_values)

        return action

    def update_weights(self, batch: tuple, gamma: float, target_network):
        criterion = torch.nn.MSELoss()

        states, actions, nextStates, rewards, dones = batch
        minibatch_number = 5
        if sum(dones)>0:
            pass
        size = int(len(states) / minibatch_number)
        self.loss = []
        if sum(dones) > 0:
            pass
        for i in range(minibatch_number):

            X = torch.tensor([el.tolist() for el in states[i * size:(i + 1) * size]]).to(self.device).\
                reshape(-1, self.inputDimension)
            X_next = torch.tensor([el.tolist() for el in nextStates[i * size:(i + 1) * size]]).to(self.device)\
                .reshape(-1, self.inputDimension)
            actions_new = torch.tensor(actions[i * size:(i + 1) * size]).to(self.device)
            rewards_new = torch.tensor(rewards[i * size:(i + 1) * size]).to(self.device)
            dones_new = torch.tensor(dones[i * size:(i + 1) * size], dtype=int).to(self.device)

            curr_Q = self.network(X).gather(1, actions_new.unsqueeze(1)).to(self.device)
            curr_Q = curr_Q.squeeze(1)

            with torch.no_grad():
                next_Q = target_network.network(X_next).to(self.device)
                max_next_Q = torch.max(next_Q, 1)[0]
                expected_Q = (rewards_new + (1 - dones_new) * gamma * max_next_Q).to(self.device)

            loss = criterion(curr_Q, expected_Q.detach())
            self.loss.append(loss.item())
            display.clear_output(wait=True)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
            self.optimizer.step()

    def take_weights(self, model_network):
        self.network.load_state_dict(model_network.network.state_dict())

    def save_weights(self, filename: str):
        torch.save(self.network.state_dict(), filename + '.pt')
