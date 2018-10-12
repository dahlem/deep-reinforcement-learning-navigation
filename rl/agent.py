# coding: utf-8
from abc import ABCMeta, abstractmethod

import numpy as np

import random

from . model import QNetwork, DuelingQNetwork
from . buffer import UniformReplayBuffer, PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

EPSILON = 1e-5          # small number for numeric stability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.params = params
        self.state_size = self.params.get('state_size', None)
        self.action_size = self.params.get('action_size', None)
        self.seed = self.params.get('seed', 1234)

        # Q-Network
        self.qnetwork_local = self.params.get('network_type', None)(self.state_size, self.action_size, self.seed).to(device)
        self.qnetwork_target = self.params.get('network_type', None)(self.state_size, self.action_size, self.seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params.get('alpha', 0.001))

        # Replay memory: to be defined in derived classes
        self.memory = None
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, policy):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            policy (GLIEPolicy): Policy, e.g., EpsilonGreedy, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # we just do a forward evaluation
        with torch.no_grad():       # don't compute gradients in the process
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # mark the network as trainable

        return policy.apply(action_values.cpu().data.numpy())


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    @abstractmethod
    def step(self, state, action, reward, next_state, done, beta):
        pass

    @abstractmethod
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        pass


class DDQN_UER_Agent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            qnetwork: list of either 'qnetwork' or 'dueling'
        """
        super().__init__(params)
        self.update_every = self.params.get('update_every', 4)
        self.gamma = self.params.get('gamma', 0.99)
        self.tau = self.params.get('tau', 0.4)
        
        # Replay memory
        self.memory = UniformReplayBuffer(self.action_size, params.get('experience_params', None))
    
    def step(self, state, action, reward, next_state, done, beta):
        # Save experience in replay memory
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.ready():
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        
class DDQN_PER_Agent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            qnetwork: list of either 'quentwork' or 'dueling'
        """
        super().__init__(params)
        
        self.update_every = self.params.get('update_every', 4)
        self.gamma = self.params.get('gamma', 0.99)
        self.tau = self.params.get('tau', 0.4)
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(self.action_size, params)

    def step(self, state, action, reward, next_state, done, beta):
        # Save experience in replay memory
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()  # we just do a forward evaluation
        with torch.no_grad():       # don't compute gradients
            Q_s_amax = self.qnetwork_local(next_state).detach().max(1)[0].unsqueeze(1).cpu().data.numpy()[0]
            Q_s_a = self.qnetwork_local(state).cpu().data.numpy()[0][action]
        self.qnetwork_local.train() # mark the network as trainable

        error = np.abs(Q_s_a - (reward + self.gamma * Q_s_amax))

        self.memory.add(state, action, reward, next_state, done, error[0])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.ready():
                experiences = self.memory.sample(beta)
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss  = (Q_expected - Q_targets).pow(2).reshape(weights.shape) * weights
        priorities = loss + EPSILON
        self.memory.update(indices, priorities.data.cpu().numpy())

        loss = loss.mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
