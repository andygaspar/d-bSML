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
        self.loss = 0
        self.inputDimension = (2 * boardsize + 2) * boardsize  # dimensions
        self.hidden = hidden
        self.epochs = epochs
        self.network = nn.Sequential(
            nn.Linear(self.inputDimension, self.inputDimension))


    def get_action(self, state: Board) -> int:
        X = torch.from_numpy(state).reshape(1, self.inputDimension).type(dtype=torch.float32) / len(state.vectorBoard)
        Q_values = self.network(X)
        return torch.argmax(torch.flatten(Q_values)).item()

    def update_weights(self, batch: tuple, gamma: float):
        # *******************++ da implementare
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3, weight_decay=1e-5)
        states, actions, nextStates, rewards = batch
        X = torch.tensor([el.tolist() for el in states]).reshape(len(states), self.inputDimension) / len(states[0].vectorBoard)
        X_next = torch.tensor([el.tolist() for el in nextStates]).reshape(len(nextStates),
                                                                                      self.inputDimension) / len(states[0].vectorBoard)
        for e in range(self.epochs):
            q_current_matrix = self.network(X)
            q_actions_done = torch.tensor([row[action] for row, action in zip(q_current_matrix, actions)])
            q_target_matrix = self.network(X_next)
            q_target_max = torch.tensor([torch.max(row) for row in q_target_matrix])
            loss = torch.nn.functional.mse_loss(q_actions_done, rewards + gamma * q_target_max)
            if e % 100 == 0: print("[EPOCH]: {}, [LOSS]: {}".format(e, loss.item()))
            display.clear_output(wait=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class NetworkOnlyValid(Network):

    def __init__(self, boardsize: int, hidden: int, epochs: int):
        super().__init__(boardsize, hidden, epochs)

    def get_action(self, state: np.array) -> int:
        X = torch.from_numpy(state).reshape(1, self.inputDimension).type(dtype=torch.float32) / len(state)
        q_values = self.network(X)
        action = torch.argmax(torch.flatten(q_values)).item()

        while state[action] == 1:
            q_values[0, action] = torch.min(q_values).item() - 10
            action = torch.argmax(torch.flatten(q_values)).item()

        return action

    def update_weights(self, batch: tuple, gamma: float):
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=1e-2, weight_decay=1e-3)

        states, actions, nextStates, rewards = batch
        X = torch.tensor([el.tolist() for el in states]).reshape(len(states), self.inputDimension) / len(states[0])
        X_next = torch.tensor([el.tolist() for el in nextStates]).reshape(len(nextStates), self.inputDimension) / len(states[0])

        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        #q_current_matrix = self.network(X)
        #q_actions_done = torch.tensor([row[action] for row, action in zip(q_current_matrix, actions)], requires_grad=True)

        #q_target_matrix = self.network(X_next)
        #valid_qvalues_matrix = [q_target_matrix[i][self.get_action(state)] for i, state in enumerate(nextStates)]

        #q_target_matrix = self.network(X_next)
        #valid_states = int(not nextStates)
        #q_target_valid = q_target_matrix[valid_states]

        #q_target_max = torch.tensor([torch.max(row) for row in valid_qvalues_matrix], requires_grad=True)
        #loss = criterion(q_actions_done, torch.tensor(rewards) + gamma * q_target_max)
        #policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

        curr_Q = self.network(X).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.network(X_next)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + gamma * max_next_Q

        loss = criterion(curr_Q, expected_Q.detach())
        self.loss = loss.item()
        #print("{}".format(loss.item()))
        display.clear_output(wait=True)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
        optimizer.step()
