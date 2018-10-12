from abc import ABC, abstractmethod

import numpy as np
import random


class GLIEPolicy(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def apply(self, action_values):
        pass
    
    @abstractmethod
    def decay(self):
        pass


class EpsilonGreedy(GLIEPolicy):
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
