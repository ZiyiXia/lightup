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
    

def train(env_name, hidden_dim=64, total_episodes=1000, lr=1e-2, gamma=0.99, eps=1e-7):
    # set training device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # create gym environment
    env = gym.make(env_name)
    
    # create the policy network according to the state and action space
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
    optimizer = Adam(policy.parameters(), lr=lr)
    
    training_loss = []
    training_reward = []

    # train for one episode
    def train_one_episode():
        obs, _ = env.reset()
        rewards = []
        actions = []
        log_probs = []
        
        while True:
            # construct the trajectory and compute log probability
            obs = torch.tensor(obs).reshape(1, -1).to(device)
            logits = policy(obs)
            probs = Categorical(F.softmax(logits, dim=-1))
            action = probs.sample()
            log_prob = probs.log_prob(action)
            
            # move one step forward in the env
            obs, reward, terminated, truncated, _ = env.step(action.item())
            
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            if terminated or truncated:
                Vs = deque()
                v = 0
                for r in rewards[::-1]:
                    v = r + gamma * v
                    Vs.appendleft(v)
                Vs = torch.as_tensor(Vs).to(device)
                Vs = (Vs - Vs.mean()) / (Vs.std() + eps)
                loss = -(torch.cat(log_probs) * Vs).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                return loss, sum(rewards)

    for i in range(total_episodes):
        loss, reward = train_one_episode()
        if (i+1) % 100 == 0:
            print(f"Episode {i+1}")
        training_loss.append(loss.cpu().detach().numpy())
        training_reward.append(reward)
        
    return training_loss, training_reward

if __name__ == "__main__":
    begin = time.time()
    loss, rew = train(env_name='CartPole-v1', total_episodes=1000)
    end = time.time()
    print(f"total time: {end-begin}")
    
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    ax1.plot(loss)
    ax1.set_title('loss')
    ax2.plot(rew)
    ax2.set_title('reward')
    plt.savefig('res.png')