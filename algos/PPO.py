import gymnasium as gym
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import matplotlib.pyplot as plt

import time
from collections import deque


class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(self, x):
        return self.network(x)
    

# TODO
def train(env_name, hidden_dim=64, total_episodes=1000, lr=1e-2, gamma=0.99, eps=1e-7):
    
    pass