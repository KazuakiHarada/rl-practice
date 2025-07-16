import gymnasium as gym
import numpy as np
import torch
import os
from dqn_agent import DQNAgent

action_mapping = [-1.0, 0.0, 1.0]  # 離散行動のマッピング
def get_discrete_action(idx):
    return np.array([action_mapping[idx]])

def evaluate_agent(model_path, num_episodes=10, max_steps=300, render=False):
    """
    指定した重みファイルでエージェントを評価し、平均step数を返す
    Args:
        model_path (str): 学習済みモデルのパス
        num_episodes (int): 評価試行回数
        max_steps (int): 1エピソードあたりの最大step数
        render (bool): 描画するかどうか
    Returns:
        float: 平均step数
    """
    env = gym.make("InvertedPendulum-v4", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim_discrete = len(action_mapping)

    agent = DQNAgent(state_dim, action_dim_discrete, buffer_capacity=1, model_path=model_path)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        if 'model_state_dict' in checkpoint:
            agent.q_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded existing model from {model_path}")
            agent.q_net.eval()
        # agent.load_model(checkpoint)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    step_counts = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        for t in range(max_steps):
            if render:
                env.render()
            a_idx = agent.select_action(state, eval_mode=True) if hasattr(agent, 'select_action') else agent.select_action(state)
            action = get_discrete_action(a_idx)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                step_counts.append(t + 1)
                break
        else:
            step_counts.append(max_steps)
        print(f"Episode {ep+1}: steps = {step_counts[-1]}")
    env.close()
    avg_steps = sum(step_counts) / len(step_counts)
    print(f"Average steps over {num_episodes} episodes: {avg_steps:.2f}")
    return avg_steps

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Render environment visually")
    args = parser.parse_args()
    evaluate_agent(args.model, args.episodes, args.max_steps, args.render)
