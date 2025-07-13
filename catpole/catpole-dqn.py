import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

# --- 環境 ---
env = gym.make("InvertedPendulum-v4", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 1  # 連続行動だが、DQNでは離散化して使う

# --- ネットワーク ---
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)  # 離散化した3つの行動

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


q_net = QNet()
target_net = QNet()
if os.path.exists("best_model.pth"):
    checkpoint = torch.load("best_model.pth")
    q_net.load_state_dict(checkpoint['model_state_dict'])
    
    target_net.load_state_dict(q_net.state_dict())
    print("Loaded existing model from best_model.pth")
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
replay_buffer = deque(maxlen=10000)

# --- ハイパーパラメータ ---
BATCH = 64
GAMMA = 0.99
EPSILON = 0.1
MAX_EP = 300
UPDATE_FREQ = 10

def get_discrete_action(idx):
    return np.array([[-1.0, 0.0, 1.0][idx]])

def select_action(state):
    if random.random() < EPSILON:
        return random.randint(0, 2)
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        return q_net(state).argmax().item()

def train():
    if len(replay_buffer) < BATCH:
        return
    batch = random.sample(replay_buffer, BATCH)
    s, a, r, s_, d = zip(*batch)
    s = torch.FloatTensor(s)
    a = torch.LongTensor(a).unsqueeze(1)
    r = torch.FloatTensor(r).unsqueeze(1)
    s_ = torch.FloatTensor(s_)
    d = torch.FloatTensor(d).unsqueeze(1)

    q = q_net(s).gather(1, a)
    with torch.no_grad():
        max_q = target_net(s_).max(1, keepdim=True)[0]
        target = r + GAMMA * max_q * (1 - d)
    loss = nn.MSELoss()(q, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def save_model(episode, avg_reward):
    checkpoint = {
        'episode': episode,
        'avg_reward': avg_reward,
        'model_state_dict': q_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, "best_model.pth")
    print(f"Model saved at episode {episode} with average reward {avg_reward:.2f}")


# --- メインループ ---

max_total_rewards = 0  # <-- initialize once, outside the loop
total_rewards = []
for ep in range(MAX_EP):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):
        env.render()
        a_idx = select_action(state)
        action = get_discrete_action(a_idx)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.append((state, a_idx, reward, next_state, done))
        state = next_state
        total_reward += reward
        train()
        if done:
            break
        
    print(f"Episode {ep}, Reward: {total_reward:.2f}")
    total_rewards.append(total_reward)

    if ep % UPDATE_FREQ == 0:
        
        # 直近の10エピソードの平均報酬を計算
        avg_reward = sum(total_rewards) / len(total_rewards)
        total_rewards.clear()
        print(f"Episode {ep}, Average Reward: {avg_reward:.2f}")
        if avg_reward > max_total_rewards:
            max_total_rewards = avg_reward
            save_model(ep, avg_reward)
            print(f"New best model saved with average reward: {avg_reward:.2f}")

        # ターゲットネットワークの更新
        target_net.load_state_dict(q_net.state_dict())

env.close()
