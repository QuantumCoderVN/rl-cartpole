import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

def evaluate(model_path="models/models/ppo_cartpole.zip.zip", n_episodes=10, render=False, seed=123):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ Evaluate on device: {device}")

    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    model = DQN.load(model_path, device=device)

    returns = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
        print(f"Episode {ep+1}: return = {ep_ret}")

    env.close()
    print(f"\nDQN — mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")

if __name__ == "__main__":
    evaluate()
