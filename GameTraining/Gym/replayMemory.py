from typing import List, Any

from Game.board import Board
import numpy as np




class ReplayMemory:
    states: List[Board]
    actions: List[int]
    nextStates: List[Board]
    rewards: List[float]
    sampleSize: int
    capacity: int

    def __init__(self, sample_size: int, capacity: int):
        self.states = []
        self.actions = []
        self.nextStates = []
        self.rewards = []
        self.sampleSize = sample_size
        self.capacity = capacity

    def add_record(self, state: Board, action: int, nextState: Board, reward: float):
        if len(self.actions) >= self.capacity:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.nextStates = self.nextStates[1:]
            self.rewards = self.rewards[1:]
        self.states.append(state)
        self.actions.append(action)
        self.nextStates.append(nextState)
        self.rewards.append(reward)

    def get_sample(self):
        random_idx = np.random.sample(self.sampleSize, range(len(self.actions)))
        return self.states[random_idx], self.actions[random_idx], self.nextStates[random_idx], self.rewards[random_idx]
