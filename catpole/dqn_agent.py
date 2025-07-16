# catpole/dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim_discrete):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim_discrete)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim_discrete, lr=1e-3, gamma=0.99, epsilon=0.1, buffer_capacity=10000, model_path=""):
        self.state_dim = state_dim
        self.action_dim_discrete = action_dim_discrete
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_path = model_path

        self.q_net = QNet(state_dim, action_dim_discrete)
        self.target_net = QNet(state_dim, action_dim_discrete)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.q_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded existing model from {self.model_path}")
        self.target_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, eval_mode=False):
        epsilon = 0.0 if eval_mode else self.epsilon
        if random.random() < epsilon:
            return random.randint(0, self.action_dim_discrete - 1)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return self.q_net(state).argmax().item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        d = torch.FloatTensor(d).unsqueeze(1)

        q = self.q_net(s).gather(1, a)
        with torch.no_grad():
            max_q = self.target_net(s_).max(1, keepdim=True)[0]
            target = r + self.gamma * max_q * (1 - d)
        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save_model(self, episode, avg_reward):
        checkpoint = {
            'episode': episode,
            'avg_reward': avg_reward,
            'model_state_dict': self.q_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.model_path)
        print(f"Model saved at episode {episode} with average reward {avg_reward:.2f}")