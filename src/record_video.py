# src/record_video.py
import os
import argparse
from uuid import uuid4
from datetime import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO

def record(
    model_path="models/dqn_cartpole.zip",
    algo="dqn",
    episodes=3,
    seed=7,
    base_out_dir="videos",
    fps=None,              # ví dụ 30
    every=1                # ghi mỗi 'every' episode (mặc định ghi tất cả)
):
    # Thư mục có timestamp + 6 ký tự random để tránh warning overwrite
    tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(base_out_dir, f"{algo}_{tag}_{uuid4().hex[:6]}")
    os.makedirs(out_dir, exist_ok=True)

    # Tạo env ghi hình (rgb_array). Có thể set fps qua metadata nếu muốn.
    base_env = gym.make("CartPole-v1", render_mode="rgb_array")
    if fps is not None:
        # Gymnasium/moviepy dùng metadata['render_fps'] để xuất video
        base_env.metadata["render_fps"] = int(fps)

    # Chỉ ghi mỗi N episode (episode_trigger)
    def episode_trigger(ep_idx: int) -> bool:
        return (ep_idx % max(1, int(every))) == 0

    env = RecordVideo(
        base_env,
        video_folder=out_dir,
        episode_trigger=episode_trigger
    )

    # Tải model
    if algo.lower() == "ppo":
        model = PPO.load(model_path)
    else:
        model = DQN.load(model_path)

    # Chạy và ghi hình
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    print(f"🎥 Videos saved to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/dqn_cartpole.zip")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out_dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=None, help="Override render_fps (e.g., 30)")
    parser.add_argument("--every", type=int, default=1, help="Record every N episodes")
    args = parser.parse_args()

    record(
        model_path=args.model_path,
        algo=args.algo,
        episodes=args.episodes,
        seed=args.seed,
        base_out_dir=args.out_dir,
        fps=args.fps,
        every=args.every,
    )
