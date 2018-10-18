# coding: utf-8
from abc import ABCMeta, abstractmethod

import numpy as np

import random

from . model import QNetwork, DuelingQNetwork
from . buffer import UniformReplayBuffer, PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

import logging
logger = logging.getLogger(__name__)


EPSILON = 1e-5          # small number for numeric stability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **network_type** (string) --- can either be "QNetwork" or "DuelingQNetwork"
        * **network_params** (dict) --- parameters for the network architecture
        * **lr** (float) --- the learning rate
        * **gamma** (float) -- the q-learning discount factor
        * **tau** (float) -- the soft update factor
        """
        logger.debug('Parameter: %s', params)

        self.params = params
        self.gamma = self.params.get('gamma', 0.99)
        self.tau = self.params.get('tau', 0.4)
        
        # Q-Network
        self.model = self.params.get('network_type', None)(self.params.get('network_params', None)).to(device)
        self.target_model = self.params.get('network_type', None)(self.params.get('network_params', None)).to(device)
        # use the same network parameters
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.get('lr', 0.001))

        # Replay memory: to be defined in derived classes
        self.memory = None
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, policy):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        * **state** (array_like) --- current state
        * **policy** (GLIEPolicy) --- Policy, e.g., EpsilonGreedy, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()  # we just do a forward evaluation
        with torch.no_grad():       # don't compute gradients in the process
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train() # mark the network as trainable

        return policy.apply(action_values.cpu().data.numpy())


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
        * **local_model** (PyTorch model) --- weights will be copied from
        * **target_model** (PyTorch model) --- weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    @abstractmethod
    def step(self, state, action, reward, next_state, done, beta):
        """Perform a step in the environment given a state, action, reward,
        next state, and done experience.

        Params
        ======
        * **state** (torch.Variable) --- the current state
        * **action** (torch.Variable) --- the current action
        * **reward** (torch.Variable) --- the current reward
        * **next_state** (torch.Variable) --- the next state
        * **done** (torch.Variable) --- the done indicator
        * **beta** (float) --- a potentially tempered beta value for prioritzed replay sampling

        """
        pass

    @abstractmethod
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        * **experiences** (Tuple[torch.Variable]) --- tuple of (s, a, r, s', done) tuples 
        """
        pass


class DDQN_UER_Agent(Agent):
    """Double q-learning agent with a uniform replay buffer."""

    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **update_every** (int) --- trigger a learning process after every n-th step
        * **experience_params** (dict) -- for example:
        'experience_params': {
            'seed': 184,
            'buffer_size': 100000,
            'batch_size': 64
        }
        """
        super().__init__(params)
        self.update_every = self.params.get('update_every', 4)
        
        # Replay memory
        self.memory = UniformReplayBuffer(params.get('experience_params', None))
    
    def step(self, state, action, reward, next_state, done, beta):
        """Perform a step in the environment and trigger a learning procedure
        after every n-th step.

        See super class for parameter descriptions.
        """
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
                self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        * **experiences** (Tuple[torch.Variable]) --- tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        
class DDQN_PER_Agent(Agent):
    """Double Q-learning agent with prioritzed replay buffer."""

    def __init__(self, params):
        """Initialize an Agent object.
        
        Params
        ======
        * **update_every** (int) --- trigger a learning process after every n-th step
        * **experience_params** (dict) -- for example:
        'experience_params': {
            'seed': 184,
            'buffer_size': 100000,
            'batch_size': 64
        }
        """
        super().__init__(params)
        
        self.update_every = self.params.get('update_every', 4)
        
        # Replay memory
        self.memory = PrioritizedReplayBuffer(params.get('experience_params', None))

    def step(self, state, action, reward, next_state, done, beta):
        """Perform a step in the environment and trigger a learning procedure
        after every n-th step.

        See super class for parameter descriptions.
        """
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
                self.learn(experiences)

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        * **experiences** (Tuple[torch.Variable]) --- tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones, weights, indices = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
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
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
