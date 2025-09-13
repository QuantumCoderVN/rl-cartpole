import gymnasium as gym
import numpy as np

def run_random(n_episodes=5, render=False, seed=42):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    returns = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)
        print(f"Episode {ep+1}: return = {ep_ret}")

    print(f"\nRandom agent — mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    env.close()

if __name__ == "__main__":
    run_random()
