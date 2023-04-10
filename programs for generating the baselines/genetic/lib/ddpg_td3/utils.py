import numpy as np
from collections import deque
import random
        
class OUProcess():

    """
    Ornstein-Uhlenbeck Process to generate noise 
    Source: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_dim=1, action_min=0, action_max=1, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):

        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.reset()
    
    def __call__(self, action, t=0):

        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        
        return np.clip(action + ou_state, self.action_min, self.action_max)
        
    def evolve_state(self):

        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        
        return self.state
    
    def reset(self):

        self.state = np.ones(self.action_dim) * self.mu