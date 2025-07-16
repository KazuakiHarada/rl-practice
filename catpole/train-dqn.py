# catpole/train_dqn.py (またはメインの実行スクリプト)
import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent # assuming dqn_agent.py is in the same directory


# --- 環境 ---
env = gym.make("InvertedPendulum-v4", render_mode="human")
state_dim = env.observation_space.shape[0]
action_mapping = [-1.0, 0.0, 1.0] # 離散行動のマッピング
action_dim_discrete = len(action_mapping)

def get_discrete_action(idx):
    return np.array([action_mapping[idx]])

# --- ハイパーパラメータ ---
BATCH_SIZE = 64
MAX_EPISODES = 300
UPDATE_FREQ = 10 # ターゲットネットワークとモデル保存の頻度

# path to save the model
model_name = "model714_2"

# --- エージェントの初期化 ---
agent = DQNAgent(state_dim, action_dim_discrete, buffer_capacity=10000, model_path=model_name + ".pth")

# --- メインループ ---

max_total_rewards = -float('inf')
total_rewards_list = [] # 直近の報酬を保存するためのリスト

if os.path.exists(model_path_):
    checkpoint = torch.load(model_path_)
    max_total_rewards = checkpoint.get('avg_reward', -float('inf'))
    episode_start = checkpoint.get('episode', 0)
    print(f"Resuming training from episode {episode_start} with max total rewards: {max_total_rewards:.2f}")
else:
    episode_start = 0
    print("Starting new training session.")

for ep in range(episode_start, MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200): # 最大ステップ数
        env.render()
        a_idx = agent.select_action(state)
        action = get_discrete_action(a_idx)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # --- ここで中心からの距離に応じてペナルティを追加 ---
        cart_position = next_state[0] 
        pole_angle = next_state[1]
        penalty = (cart_position ** 2) * 10  + (pole_angle ** 2) * 10
        reward -= penalty
        
        if truncated:
            reward -= 50
        
        # ----------------

        #print(reward, cart_position)

        done = terminated or truncated

        agent.replay_buffer.push(state, a_idx, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.train(BATCH_SIZE)

        if done:
            break
        
    print(f"Episode {ep}, Reward: {total_reward:.2f}")
    total_rewards_list.append(total_reward)

    if ep > 0 and ep % UPDATE_FREQ == 0:
        # 直近のUPDATE_FREQエピソードの平均報酬を計算
        avg_reward = sum(total_rewards_list[-UPDATE_FREQ:]) / UPDATE_FREQ
        print(f"Episode {ep}, Average Reward (last {UPDATE_FREQ} eps): {avg_reward:.2f}")
        
        if avg_reward > max_total_rewards:
            max_total_rewards = avg_reward
            agent.save_model(ep, avg_reward)
            print(f"New best model saved with average reward: {avg_reward:.2f}")

        # ターゲットネットワークの更新
        agent.update_target_network()

print(f"Training completed. Best average reward: {max_total_rewards:.2f}")
env.close()

# --- 報酬推移のグラフを保存・表示 ---
plt.figure()
plt.plot(range(episode_start, episode_start + len(total_rewards_list)), total_rewards_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode vs Total Reward")
plt.grid()
plt.savefig(model_name + "_reward_curve.png")
plt.show()