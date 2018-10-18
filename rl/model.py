import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Q-Network."""

    def __init__(self, params):
        """Initialize parameters and build model.
        Params
        ======
        * **state_size** (int) --- Dimension of each state
        * **action_size** (int) --- Dimension of each action
        * **seed** (int): Random seed
        * **hidden_layers** (array-like) --- array of the hidden layers
        * **dropout** (float) --- dropout rate
        """
        super(QNetwork, self).__init__()
        seed = params.get('seed', 0)
        self.seed = torch.manual_seed(seed)
        self.state_size = params.get('state_size', None)
        self.action_size = params.get('action_size', None)
        
        hidden_layers = params.get('hidden_layers', None)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], self.action_size)
        self.dropout = nn.Dropout(p = params.get('dropout', 0.05))

        for linear in self.hidden_layers:
            self.init_weights(linear)
        self.init_weights(self.output)

    def init_weights(self, m):
        """
        Initialize the weights of the linear layers with the uniform Xavier initializer
        """
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        return x


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network"""

    def __init__(self, params):
        """Initialize parameters and build model.
        Params
        ======
        * **state_size** (int) --- Dimension of each state
        * **action_size** (int) --- Dimension of each action
        * **seed** (int): Random seed
        * **hidden_layers** (array-like) --- array of the hidden layers
        * **dropout** (float) --- dropout rate
        """
        super(DuelingQNetwork, self).__init__()
        seed = params.get('seed', 0)
        self.seed = torch.manual_seed(seed)
        self.state_size = params.get('state_size', None)
        self.action_size = params.get('action_size', None)
        
        hidden_layers = params.get('hidden_layers', None)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        # dueling layers
        self.fc1_value = nn.Linear(in_features=hidden_layers[-1], out_features=16)
        self.fc1_advantage = nn.Linear(in_features=hidden_layers[-1], out_features=16)

        self.fc2_value = nn.Linear(in_features=16, out_features=1)
        self.fc2_advantage = nn.Linear(in_features=16, out_features=self.action_size)

        self.dropout = nn.Dropout(p = params.get('dropout', 0.05))

        # initialize the weights
        for linear in self.hidden_layers:
            self,init_weights(linear)
        self.init_weights(self.fc1_value)
        self.init_weights(self.fc1_advantage)
        self.init_weights(self.fc2_value)
        self.init_weights(self.fc2_advantage)

    def init_weights(self, m):
        """
        Initialize the weights of the linear layers with the uniform Xavier initializer
        """
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        batch_size = x.size(0)
        
        for linear in self.hidden_layers:
            x = self.dropout(F.relu(linear(x)))

        # pass through value layers
        value = self.dropout(F.relu(self.fc1_value(x)))
        value = self.dropout(self.fc2_value(value).expand(batch_size, self.action_size)) # expand the value to the larger action_size
        
        # pass through advantage layers
        advantage = self.dropout(F.relu(self.fc1_advantage(x)))
        advantage = self.dropout(self.fc2_advantage(advantage))
        
        x = value + advantage - advantage.mean(1).unsqueeze(1).expand(batch_size, self.action_size)

        return x
