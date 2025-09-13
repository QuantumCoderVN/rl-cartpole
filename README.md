# RL CartPole — Gymnasium + Stable-Baselines3 (DQN/PPO)

A tiny yet professional RL project that trains an agent to balance a pole in **CartPole-v1** using **Gymnasium** and **Stable-Baselines3**. The repo includes:

* Clean project structure & virtual env workflow
* Random baseline → DQN training → evaluation
* (Optional) PPO training for quick convergence
* Learning-curve plotting (mean ± std from `evaluations.npz`)
* Video recording with timestamped folders
* GPU/CPU setup notes (Windows & Linux/Mac)

---

## 1) Project structure

```
rl-cartpole/
├── env/                     # Python virtual environment (not committed)
├── src/
│   ├── random_baseline.py   # random agent for reference
│   ├── train.py             # DQN training (Gymnasium + SB3)
│   ├── evaluate.py          # evaluate trained agent
│   ├── plot_eval_curve.py   # plot mean reward ± std over timesteps
│   └── record_video.py      # save .mp4 videos with timestamps
├── notebooks/               # (optional) experiments
├── models/                  # saved models (ignored by git)
├── videos/                  # recorded videos (ignored by git)
├── reports/                 # plots (e.g. eval_curve.png)
├── requirements.txt
├── .gitignore
└── README.md
```

**Recommended `.gitignore`**

```
env/
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
models/
videos/
*.mp4
*.npz
```

---

## 2) Environment setup

> Use a virtual environment to keep things reproducible and clean.

### Windows (PowerShell / CMD)

```bat
python -m venv env
./env/Scripts/activate
pip install --upgrade pip
pip install -U "gymnasium[classic-control]" "stable-baselines3[extra]" shimmy numpy matplotlib moviepy
```

### Linux / macOS (bash/zsh)

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -U "gymnasium[classic-control]" "stable-baselines3[extra]" shimmy numpy matplotlib moviepy
```

> **Why Gymnasium?** It’s the maintained drop‑in replacement for `gym` and works with NumPy 2+. If you used `gym`, migrate to `gymnasium` to avoid `np.bool8`/compat errors.

---

## 3) (Optional) GPU acceleration with PyTorch

If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build. On Windows with driver reporting **CUDA 12.x**, the common choice is **cu121**:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

> **Laptop (Windows) with Intel iGPU + NVIDIA dGPU**: set the venv `python.exe` to use **High-performance NVIDIA GPU** in *Windows Settings → System → Display → Graphics* and *NVIDIA Control Panel → Manage 3D settings → Program settings*.

> **Note on PPO**: For MLP policies, PPO often trains efficiently on **CPU**. DQN benefits from GPU.

---

## 4) Quickstart (run order)

All commands assume the virtual environment is **activated** and you are at the repo root.

### 4.1 Random baseline

```bash
python src/random_baseline.py
```

Expected: very low scores (baseline for comparison).

### 4.2 Train DQN

```bash
python src/train.py
```

* Saves best model + `evaluations.npz` under `models/`.
* Console prints: `▶ Using device: cuda` if GPU is detected.

**Speed/quality knobs** (edit `train.py`):

* `timesteps`: e.g. `50_000` (quick), `200_000` (better)
* DQN hyperparams: `learning_rate`, `buffer_size`, `batch_size`, `exploration_fraction`, …

### 4.3 Evaluate a trained model

```bash
python src/evaluate.py
```

* Prints mean return over N episodes (configurable in the script).

### 4.4 Plot the learning curve

```bash
python src/plot_eval_curve.py
```

* Reads `models/evaluations.npz` → writes `reports/eval_curve.png`.

### 4.5 Record videos (timestamped folder)

```bash
# DQN model
python src/record_video.py --model_path models/dqn_cartpole.zip --algo dqn --episodes 5 --fps 30

# PPO model (if you trained PPO)
python src/record_video.py --model_path models/ppo_cartpole.zip --algo ppo --episodes 5 --fps 30
```

* Uses `render_mode="rgb_array"` + MoviePy. If missing: `pip install moviepy`.
* Creates a unique subfolder under `videos/` (no overwriting).

---

## 5) (Optional) PPO training

If you add a `src/train_ppo.py` (as suggested during mentoring), train via:

```bash
python src/train_ppo.py
```

Then evaluate by editing `model_path` in `src/evaluate.py` to `models/ppo_cartpole.zip` or by passing a CLI argument if you added one.

> PPO often reaches >400 mean return quickly on CartPole with MLP policy. For speed, feel free to set `device='cpu'` when instantiating PPO.

---

## 6) Expected results

* **Random Agent**: low returns (dozens of steps). Good as a sanity check.
* **DQN (50k timesteps)**: mean return typically **>100–200** depending on seed.
* **DQN (200k timesteps + tuned hyperparams)**: can reach **>400**.
* **PPO (200k timesteps, MLP)**: frequently achieves **>400–475**.

Use `reports/eval_curve.png` to verify that the mean eval reward increases and stabilizes.

> RL is stochastic. Seeds, hyperparameters, and hardware (CPU/GPU) can change results.

---

## 7) Troubleshooting

**Gym vs Gymnasium**

* Error like `AttributeError: module 'numpy' has no attribute 'bool8'` → you are on NumPy 2 with old `gym`. Fix: migrate to **Gymnasium** and adjust `reset/step` signatures.

**Progress bar missing**

* If you set `progress_bar=True` and get an import error: `pip install tqdm rich` or install `stable-baselines3[extra]`.

**MoviePy not installed**

* `DependencyNotInstalled: MoviePy is not installed` → `pip install moviepy` (already in Quickstart).

**“Overwriting existing videos” warning**

* Our `record_video.py` generates a **fresh timestamped folder** and lets the wrapper create it, avoiding overwrites.

**PPO GPU warning**

* SB3 warns that PPO (MLP) is usually CPU‑friendly. You may set `device='cpu'` to silence it; recording videos is unaffected.

**Windows paths with spaces**

* Use quotes around paths, e.g. `"D:\1. QUANTUM MACHINE LEARNING\..."`.

---

## 8) Reproducibility

* Set `seed` consistently when creating envs and models.
* Exact determinism is hard in RL (parallelism, GPU kernels). Aim for **statistical** reproducibility (similar mean returns).

---

## 9) Development and Git workflow

Typical cycle:

```bash
git status
# stage cleanly (safe because of .gitignore)
git add -A
git commit -m "DQN/PPO training, eval, plots, video recorder"
git push
```

Keep heavy artifacts (`models/*.zip`, `videos/*.mp4`, `*.npz`) out of git unless you use Git LFS.

---

## 10) Acknowledgements

* [Gymnasium](https://gymnasium.farama.org/) — maintained replacement for OpenAI Gym
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — reliable RL baselines (DQN, PPO, etc.)
* [MoviePy](https://zulko.github.io/moviepy/) — easy video writing from frames

---

### Contact & Notes

This README was generated as part of a guided mini‑project to help newcomers *see* RL in action within \~30–45 minutes. Extend it by adding notebooks, more algorithms (A2C/SAC/TD3), and environment wrappers (normalization, frame‑stacking, etc.).
