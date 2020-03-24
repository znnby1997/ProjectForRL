from collections import namedtuple
import random

Dynamics = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'is_end'])


class ExperienceMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, is_end):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Dynamics(state, action, reward, next_state, is_end)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch):
        return random.sample(self.memory, batch)

    def __len__(self):
        return len(self.memory)
