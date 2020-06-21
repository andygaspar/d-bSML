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

    def import_memory(self, path):

        import csv
        states = []
        with open(path+"replay_memory/states.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for state in csv_reader:
                states.append(np.array(state))

        next_states = []
        with open(path+"replay_memory/next_states.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for next_state in csv_reader:
                next_states.append(np.array(next_state))

        with open(path+"replay_memory/rewards.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            reward = []
            for rew in csv_reader:
                reward += rew

        with open(path+"replay_memory/actions.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            actions = []
            for acts in csv_reader:
                actions += acts
            actions = np.array(actions).astype(int).tolist()

        random_idx = np.random.choice(range(len(states)), size=self.capacity, replace=False).astype(int)
        self.states = [states[i] for i in random_idx]
        self.nextStates = [next_states[i] for i in random_idx]
        self.rewards = [reward[i] for i in random_idx]
        self.actions = [actions[i] for i in random_idx]
        self.size = len(self.states)

