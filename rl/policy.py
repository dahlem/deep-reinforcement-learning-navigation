from abc import ABC, abstractmethod

import numpy as np
import random


class GLIEPolicy(ABC):
    """
    An abstract base class for GLIE Policies.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def apply(self, action_values):
        """
        Apply the policy on given action values.

        Params
        ======
        * **action_values** (array-like) --- the array of action values
        """
        pass
    
    @abstractmethod
    def decay(self):
        """
        Decay the explorative nature of the policy.
        """
        pass


class EpsilonGreedy(GLIEPolicy):
    """
    Epsilon-greedy policy with epsilon decay given a dictionary of parameters.

    Params
    ======
    * **eps_start** (float) --- the epsilon value to start with
    * **eps_end** (float) --- the epsilon value to end with
    * **eps_decay** (float) --- the decay rate for the epsilon value
    """
    def __init__(self, params):
        super().__init__()
        self.eps = params.get('eps_start', 1.0)
        self.eps_end = params.get('eps_end', 0.01)
        self.eps_decay = params.get('eps_decay', 0.995)
        
    def apply(self, action_values):
        if random.random() > self.eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(len(action_values[0])))

    def decay(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)
