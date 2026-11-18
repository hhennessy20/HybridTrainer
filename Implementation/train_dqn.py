"""Training pipeline for Dueling Double Deep Q-Network (HybridTrainer Ckpt 3).

Trains the continuous-state neural policy across synthetic user interaction episodes,
tracking loss, episode rewards, adherence, and logging checkpoints.
"""

import os
import json
import csv
import random
import time
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
os.environ["MPLCONFIGDIR"] = str(OUTPUTS_DIR / ".mpl_cache")

from typing import Dict, Any, List
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from Baseline.config import (
    SEED_TRAIN,
    WEIGHT_COMPLETION,
    WEIGHT_SATISFACTION,
    WEIGHT_RPE_ERROR,
    WEIGHT_FATIGUE,
)
from Baseline.environment.schemas import UserState, validate_workout_action
from Baseline.environment.reward import compute_simulated_feedback, step_state
from Baseline.environment.synthetic_users import generate_scenarios
from Implementation.state_encoder import encode_user_state, STATE_DIM
from Implementation.replay_buffer import ReplayBuffer
from Implementation.dqn_agent import DuelingDDQNAgent


def train_dqn(
    episodes: int = 800,
    steps_per_episode: int = 15,
    batch_size: int = 64,
    lr: float = 5e-4,
    gamma: float = 0.95,
    tau: float = 0.005,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.996,
    warmup_steps: int = 300,
    output_dir: Path = OUTPUTS_DIR,
    seed: int = SEED_TRAIN,
) -> DuelingDDQNAgent:
    """Execute training loop for Dueling Double DQN."""
    output_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Training Dueling Double DQN: {episodes} episodes, {steps_per_episode} steps/ep, lr={lr}, gamma={gamma}")

    agent = DuelingDDQNAgent(
        state_dim=STATE_DIM,
        action_dim=9,
        hidden_dim=128,
        lr=lr,
        gamma=gamma,
        tau=tau,
    )
    replay_buffer = ReplayBuffer(capacity=25000, seed=seed)

    # Generate synthetic training user scenarios across training seed range
    train_users = generate_scenarios(start_seed=seed, count=episodes)

    epsilon = epsilon_start
    training_history: List[Dict[str, Any]] = []
    total_env_steps = 0
    t0 = time.time()

    for ep in range(1, episodes + 1):
        state = train_users[ep - 1]
        ep_reward = 0.0
        ep_losses: List[float] = []
        ep_adherence: List[float] = []
        ep_satisfaction: List[float] = []

        for st in range(steps_per_episode):
            s_vec = encode_user_state(state)
            s_mask = agent.compute_action_mask(state)

            # Epsilon-greedy action selection
            action_idx = agent.select_action_idx(state, epsilon=epsilon, rng=rng)
            action_dict = agent.action_to_dict(action_idx, state)

            # Validate and calculate simulated feedback/reward
            val_result = validate_workout_action(action_dict, state)
            feedback = compute_simulated_feedback(state, action_dict, val_result)

            # Step user state to next workout day
            next_state = step_state(state, action_dict, feedback)
            next_s_vec = encode_user_state(next_state)
            next_s_mask = agent.compute_action_mask(next_state)

            done = (st == steps_per_episode - 1)
            replay_buffer.push(
                state=s_vec,
                action=action_idx,
                reward=feedback.reward,
                next_state=next_s_vec,
                done=done,
                action_mask=s_mask,
                next_action_mask=next_s_mask,
            )

            total_env_steps += 1
            ep_reward += feedback.reward
            ep_adherence.append(feedback.completion)
            ep_satisfaction.append(feedback.satisfaction)

            # Training optimization step if warmup is complete
            if len(replay_buffer) >= warmup_steps:
                batch = replay_buffer.sample(batch_size=batch_size, device=agent.device)
                loss_val = agent.train_step(batch)
                ep_losses.append(loss_val)

            state = next_state

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        mean_ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        mean_ep_adh = float(np.mean(ep_adherence))
        mean_ep_sat = float(np.mean(ep_satisfaction))

        ep_record = {
            "episode": ep,
            "total_steps": total_env_steps,
            "epsilon": round(epsilon, 4),
            "episode_reward": round(ep_reward, 3),
            "mean_reward_per_step": round(ep_reward / steps_per_episode, 3),
            "mean_loss": round(mean_ep_loss, 5),
            "mean_adherence": round(mean_ep_adh, 3),
            "mean_satisfaction": round(mean_ep_sat, 3),
        }
        training_history.append(ep_record)

        # Logging
        if ep % 50 == 0 or ep == 1 or ep == episodes:
            elapsed = time.time() - t0
            print(
                f"Ep {ep:3d}/{episodes:3d} | "
                f"Steps: {total_env_steps:5d} | "
                f"Reward/Step: {ep_record['mean_reward_per_step']:+.3f} | "
                f"Adh: {mean_ep_adh:.2f} | "
                f"Sat: {mean_ep_sat:.2f} | "
                f"Loss: {mean_ep_loss:.4f} | "
                f"Eps: {epsilon:.3f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        # Save periodic checkpoints
        if ep in [200, 400, 600, 800]:
            ckpt_path = CHECKPOINTS_DIR / f"dqn_checkpoint_ep{ep}.pt"
            agent.save(ckpt_path)

    # Save final model
    final_model_path = output_dir / "dqn_model.pt"
    agent.save(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Save training history JSON and CSV
    hist_json_path = output_dir / "training_history.json"
    with open(hist_json_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    hist_csv_path = output_dir / "training_history.csv"
    with open(hist_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(training_history[0].keys()))
        writer.writeheader()
        writer.writerows(training_history)

    print(f"Training history saved to {hist_json_path}")

    # Plot training loss & reward curves
    plot_training_curves(training_history, output_dir / "dqn_training_curve.png")

    return agent


def plot_training_curves(history: List[Dict[str, Any]], save_path: Path):
    """Plot publication-quality training curves for loss and composite reward."""
    episodes = [h["episode"] for h in history]
    rewards = [h["mean_reward_per_step"] for h in history]
    losses = [h["mean_loss"] for h in history]
    adherence = [h["mean_adherence"] for h in history]
    satisfaction = [h["mean_satisfaction"] for h in history]

    # Calculate rolling averages (window = 25)
    window = 25
    def rolling(data):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    roll_ep = episodes[window - 1 :]
    roll_rew = rolling(rewards)
    roll_adh = rolling(adherence)
    roll_sat = rolling(satisfaction)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)

    # 1. Composite Reward Curve
    axs[0].plot(episodes, rewards, color="#3a86ff", alpha=0.25, label="Raw Ep.")
    axs[0].plot(roll_ep, roll_rew, color="#1d3557", linewidth=2.0, label="25-Ep MA")
    axs[0].set_title("(a) Step Reward Progression", fontsize=10, fontweight="bold")
    axs[0].set_xlabel("Training Episodes", fontsize=9)
    axs[0].set_ylabel("Reward ($R_t$)", fontsize=9)
    axs[0].grid(True, linestyle="--", alpha=0.3)
    axs[0].legend(loc="lower right", fontsize=8)

    # 2. Smooth L1 Loss Curve
    loss_ep = [h["episode"] for h in history if h["mean_loss"] > 0]
    loss_val = [h["mean_loss"] for h in history if h["mean_loss"] > 0]
    roll_loss = rolling(loss_val) if len(loss_val) >= window else loss_val
    roll_loss_ep = loss_ep[window - 1 :] if len(loss_val) >= window else loss_ep
    axs[1].plot(loss_ep, loss_val, color="#e63946", alpha=0.25, label="Raw Loss")
    axs[1].plot(roll_loss_ep, roll_loss, color="#9e2a2b", linewidth=2.0, label="25-Ep MA")
    axs[1].set_title("(b) Smooth L1 Bellman Loss", fontsize=10, fontweight="bold")
    axs[1].set_xlabel("Training Episodes", fontsize=9)
    axs[1].set_ylabel("Huber Loss", fontsize=9)
    axs[1].grid(True, linestyle="--", alpha=0.3)
    axs[1].legend(loc="upper right", fontsize=8)

    # 3. Adherence & Satisfaction
    axs[2].plot(roll_ep, roll_adh, color="#2a9d8f", linewidth=1.8, label="Adherence ($C_t$)")
    axs[2].plot(roll_ep, roll_sat, color="#e76f51", linewidth=1.8, label="Satisfaction ($S_t$)")
    axs[2].set_title("(c) Behavioral Metrics (25-Ep MA)", fontsize=10, fontweight="bold")
    axs[2].set_xlabel("Training Episodes", fontsize=9)
    axs[2].set_ylabel("Metric Score [0, 1]", fontsize=9)
    axs[2].set_ylim(0.0, 1.05)
    axs[2].grid(True, linestyle="--", alpha=0.3)
    axs[2].legend(loc="lower right", fontsize=8)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    train_dqn()
