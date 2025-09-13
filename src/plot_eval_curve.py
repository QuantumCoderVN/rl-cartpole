# src/plot_eval_curve.py
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_eval_curve(
    npz_path="models/evaluations.npz",
    out_dir="reports",
    out_name="eval_curve.png",
    title="DQN on CartPole-v1 — Evaluation Mean Reward"
):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Không tìm thấy {npz_path}. Hãy chạy train trước (src/train.py) để tạo evaluations.npz."
        )

    data = np.load(npz_path)
    timesteps = data["timesteps"]              # shape: (n_eval, )
    results = data["results"]                  # shape: (n_eval, n_eval_episodes)
    # ep_lengths = data["ep_lengths"]          # nếu muốn vẽ độ dài episode

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, label="Mean Eval Reward")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="±1 std"
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Saved plot to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    plot_eval_curve()
