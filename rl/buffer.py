from abc import ABCMeta, abstractmethod

import numpy as np

import random
from collections import namedtuple, deque

import torch

EPSILON = 1e-5          # small number for numeric stability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        pass
    
    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def ready(self):
        pass
    

class UniformReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, params):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.params = params
        self.action_size = action_size
        self.memory = deque(maxlen=self.params.get('buffer_size', 100000))  
        self.batch_size = self.params.get('batch_size', 64)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(self.params.get('seed', 1234))
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def ready(self):
        """Return the current size of internal memory."""
        return len(self.memory) > self.batch_size


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store prioritized experience tuples."""

    def __init__(self, action_size, params):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.params = params
        self.action_size = action_size
        self.buffer_size = self.params.get('buffer_size', 100000)
        self.memory = deque(maxlen=self.buffer_size)
        self.priorities = deque(maxlen=selfbuffer_size)
        self.batch_size = self.params.get('batch_size', 64)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(self.params.get('seed', 1234))
        self.alpha = self.params.get('alpha', 0.4)
    
    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        self.priorities.append(priority)
    
    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        priorities = np.array(self.priorities).reshape(-1)
        priorities = np.power(priorities + EPSILON, self.alpha)  # add a small value epsilon to ensure numeric stability
        p = priorities/np.sum(priorities)                              # compute a probability density over the priorities
        sampled_indices = np.random.choice(np.arange(len(p)), size=self.batch_size, p=p)  # choose random indices given p
        experiences = [self.memory[i] for i in sampled_indices]     # subset the experiences
        p = np.array([p[i] for i in sampled_indices]).reshape(-1)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        weights = np.power(len(experiences) * p, -beta)

        weights /= weights.max()
        weights = torch.from_numpy(weights).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, sampled_indices)

    def update(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority

    def ready(self):
        """Return the current size of internal memory."""
        return len(self.memory) > self.batch_size
