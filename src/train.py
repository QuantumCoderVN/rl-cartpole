import os
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def make_env(seed=0, render=False, monitor=True):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if monitor:
        env = Monitor(env)
    return env

def train_dqn(
    timesteps=50_000,
    save_dir="models",
    model_name="dqn_cartpole",
    seed=0,
    reward_threshold=475.0
):
    os.makedirs(save_dir, exist_ok=True)

    # chọn thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"▶ Using device: {device}")

    env = make_env(seed=seed, render=False, monitor=True)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        train_freq=4,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=1,
        seed=seed,
        device=device,   # chạy GPU
    )

    eval_env = make_env(seed=seed+123, render=False, monitor=False)

    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=5_000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best,
    )

    model.learn(total_timesteps=timesteps, progress_bar=True, callback=eval_callback)

    final_path = os.path.join(save_dir, f"{model_name}.zip")
    model.save(final_path)
    env.close(); eval_env.close()
    print(f"✅ Saved model to: {final_path}")

if __name__ == "__main__":
    train_dqn()
