from typing import List
import numpy as np
from csv import writer


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

    def export_memory(self):
        with open("replay_memory/states.csv", 'a+', newline='') as write_obj:
            for state in self.states:
                csv_writer = writer(write_obj)
                csv_writer.writerow(state)

        with open("replay_memory/next_states.csv", 'a+', newline='') as write_obj:
            for next_state in self.nextStates:
                csv_writer = writer(write_obj)
                csv_writer.writerow(next_state)

        with open("replay_memory/actions.csv", 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(self.actions)

        with open("replay_memory/rewards.csv", 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(self.rewards)

    def import_memory(self):
        import csv
        with open("replay_memory/states.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for state in csv_reader:
                self.states.append(np.array(state))

        with open("replay_memory/next_states.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for next_state in csv_reader:
                self.nextStates.append(np.array(next_state))

        with open("replay_memory/rewards.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            reward = []
            for rew in csv_reader:
                reward += rew
            self.rewards = reward

        with open("replay_memory/actions.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            actions = []
            for acts in csv_reader:
                actions += acts
            self.actions = np.array(actions).astype(int).tolist()
