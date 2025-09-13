import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def make_env(seed=0, render=False, monitor=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if monitor:
        env = Monitor(env)
    return env

def train_ppo(timesteps=200_000, save_dir="models", model_name="ppo_cartpole", seed=0):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ Using device: {device}")

    env = make_env(seed, render=False, monitor=True)
    eval_env = make_env(seed+123, render=False, monitor=False)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        seed=seed,
        verbose=1,
        device=device,
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnRewardThreshold(475.0, verbose=1),
    )

    model.learn(total_timesteps=timesteps, progress_bar=True, callback=callback)
    path = os.path.join(save_dir, f"{model_name}.zip")
    model.save(path)
    env.close(); eval_env.close()
    print(f"✅ Saved PPO model to: {path}")

if __name__ == "__main__":
    train_ppo()
