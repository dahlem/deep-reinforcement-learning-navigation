import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, params):
        super(Agent, self).__init__()
        logger.debug('Parameter: %s', params)

        self.env = params.get('environment', None)
        self.max_t = params.get('max_t', None)
        self.state_size = params['agent_params']['network_params'].get('state_size', None)
        self.action_size = params['agent_params']['network_params'].get('action_size', None)
        self.gamma = params['agent_params'].get('gamma', None)
        self.brain_name = params.get('brain_name', None)
        
        hidden_layers = params['agent_params']['network_params'].get('hidden_layers', None)
        
        self.layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.layers.append(nn.Linear(hidden_layers[-1], self.action_size))
        
    def set_weights(self, weights):
        offset = 0
        for linear in self.layers:
            (out_size, in_size) = linear.weight.data.size()
            fc_W = torch.from_numpy(weights[offset:offset + in_size * out_size].reshape(in_size, out_size))
            fc_b = torch.from_numpy(weights[offset + in_size * out_size:offset + (in_size + 1) * out_size])
            linear.weight.data.copy_(fc_W.view_as(linear.weight.data))
            linear.bias.data.copy_(fc_b.view_as(linear.bias.data))
            offset += (in_size + 1) * out_size

    def get_weights_dim(self):
        dim = 0
        for linear in self.layers:
            (in_size, out_size) = linear.weight.data.size()
            dim += (in_size + 1) * out_size
        return dim
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for i, linear in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = F.relu(linear(x))
            else:
                x = linear(x)
        return x.cpu().data.numpy()
        
    def evaluate(self, weights, policy):
        self.set_weights(weights)
        episode_return = 0.0
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        for t in range(self.max_t):
            state = torch.from_numpy(state).float().to(device)
            action = policy.apply(self.forward(state))
            env_info = self.env.step(action)[self.brain_name]        # send the action to the environment
            state = env_info.vector_observations[0]        # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            episode_return += reward * math.pow(self.gamma, t)
            if done:
                break
        return episode_return
