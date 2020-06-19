from typing import List
import numpy as np


class ReplayMemory:
    size: int
    states: List[np.array]
    actions: List[int]
    nextStates: List[np.array]
    rewards: List[float]
    sampleSize: int
    capacity: int

    def __init__(self, sample_size: int, capacity: int):
        self.size = 0
        self.states = []
        self.actions = []
        self.nextStates = []
        self.rewards = []
        self.sampleSize = sample_size
        self.capacity = capacity

    def add_record(self, state: np.array, action: int, nextState: np.array, reward: float):
        if len(self.actions) >= self.capacity:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.nextStates = self.nextStates[1:]
            self.rewards = self.rewards[1:]
            self.size -= 1
        self.states.append(state)
        self.actions.append(action)
        self.nextStates.append(nextState)
        self.rewards.append(reward)
        self.size += 1

    def get_sample(self):
        random_idx = np.random.choice(range(self.size), size=self.sampleSize, replace=False).astype(int)
        return [self.states[i] for i in random_idx], [self.actions[i] for i in random_idx], \
               [self.nextStates[i] for i in random_idx], [self.rewards[i] for i in random_idx]
