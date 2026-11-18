import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.config import (
    RL_DEFAULT_TRAIN_EPISODES,
    RL_STEPS_PER_EPISODE,
    RL_LEARNING_RATE,
    RL_DISCOUNT_FACTOR,
    RL_EPSILON_START,
    RL_EPSILON_END,
    RL_EPSILON_DECAY,
    RL_QTABLE_PATH,
    OUTPUTS_DIR,
    SEED_TRAIN,
    SEED_VAL,
)
from Baseline.policies.rl_qlearning import TabularQLearningPolicy
from Baseline.environment.synthetic_users import generate_scenarios

def generate_svg_learning_curve(history: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate a clean vector SVG plot of the RL learning curve with moving average."""
    if not history:
        return

    episodes = [h["episode"] for h in history]
    rewards = [h["episode_reward"] for h in history]

    # Calculate moving average (window 20)
    w = 20
    ma = []
    for i in range(len(rewards)):
        start = max(0, i - w + 1)
        ma.append(sum(rewards[start : i + 1]) / (i - start + 1))

    width, height = 700, 400
    pad_left, pad_right, pad_top, pad_bottom = 60, 40, 40, 50
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    min_y = min(min(rewards), -1.0)
    max_y = max(max(rewards), 6.0)
    y_range = max(1e-5, max_y - min_y)
    x_range = max(1, len(episodes))

    def to_screen(x: float, y: float):
        sx = pad_left + (x / x_range) * plot_w
        sy = pad_top + (1.0 - (y - min_y) / y_range) * plot_h
        return sx, sy

    # Build raw polyline points and smoothed polyline points
    raw_pts = " ".join([f"{to_screen(i, r)[0]:.1f},{to_screen(i, r)[1]:.1f}" for i, r in enumerate(rewards)])
    ma_pts = " ".join([f"{to_screen(i, m)[0]:.1f},{to_screen(i, m)[1]:.1f}" for i, m in enumerate(ma)])

    # Gridlines and Y ticks
    grid_svg = []
    num_ticks = 5
    for t in range(num_ticks + 1):
        tick_val = min_y + (t / num_ticks) * y_range
        _, ty = to_screen(0, tick_val)
        grid_svg.append(f'<line x1="{pad_left}" y1="{ty:.1f}" x2="{width - pad_right}" y2="{ty:.1f}" stroke="#e2e8f0" stroke-width="1" />')
        grid_svg.append(f'<text x="{pad_left - 10}" y="{ty + 4:.1f}" font-family="sans-serif" font-size="11" text-anchor="end" fill="#64748b">{tick_val:.1f}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width / 2}" y="24" font-family="sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">Tabular Q-Learning Training Curve</text>
  {''.join(grid_svg)}
  <!-- Axes -->
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{height - pad_bottom}" stroke="#94a3b8" stroke-width="1.5" />
  <line x1="{pad_left}" y1="{height - pad_bottom}" x2="{width - pad_right}" y2="{height - pad_bottom}" stroke="#94a3b8" stroke-width="1.5" />
  <text x="{pad_left + plot_w / 2}" y="{height - 15}" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#475569">Training Episode</text>
  <text x="18" y="{pad_top + plot_h / 2}" font-family="sans-serif" font-size="12" text-anchor="middle" fill="#475569" transform="rotate(-90 18 {pad_top + plot_h / 2})">Episode Cumulative Reward</text>
  <!-- Data series -->
  <polyline points="{raw_pts}" fill="none" stroke="#93c5fd" stroke-width="1" opacity="0.6" />
  <polyline points="{ma_pts}" fill="none" stroke="#2563eb" stroke-width="2.5" />
  <!-- Legend -->
  <rect x="{width - 210}" y="35" width="160" height="45" fill="#f8fafc" stroke="#cbd5e1" rx="4" />
  <line x1="{width - 200}" y1="50" x2="{width - 175}" y2="50" stroke="#93c5fd" stroke-width="1.5" />
  <text x="{width - 170}" y="53" font-family="sans-serif" font-size="11" fill="#334155">Episode Reward</text>
  <line x1="{width - 200}" y1="67" x2="{width - 175}" y2="67" stroke="#2563eb" stroke-width="2.5" />
  <text x="{width - 170}" y="70" font-family="sans-serif" font-size="11" fill="#334155">Moving Avg ({w} ep)</text>
</svg>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)

def train_rl(
    episodes: int = RL_DEFAULT_TRAIN_EPISODES,
    steps_per_episode: int = RL_STEPS_PER_EPISODE,
    alpha: float = RL_LEARNING_RATE,
    gamma: float = RL_DISCOUNT_FACTOR,
    seed: int = SEED_TRAIN,
    val_seed: int = SEED_VAL,
    output_dir: Path = OUTPUTS_DIR,
) -> TabularQLearningPolicy:
    """Train Q-learning agent on synthetic scenarios and evaluate on validation scenarios."""
    print(f"Training Tabular Q-Learning: {episodes} episodes, {steps_per_episode} steps/ep, alpha={alpha}, gamma={gamma}")

    policy = TabularQLearningPolicy()
    t0 = time.perf_counter()

    history = policy.train(
        num_episodes=episodes,
        steps_per_episode=steps_per_episode,
        alpha=alpha,
        gamma=gamma,
        eps_start=RL_EPSILON_START,
        eps_end=RL_EPSILON_END,
        eps_decay=RL_EPSILON_DECAY,
        base_seed=seed,
    )
    elapsed = time.perf_counter() - t0

    # Save Q-table
    qtable_path = output_dir / "rl_qtable.json"
    policy.save(qtable_path)

    # Save training history
    history_path = output_dir / "rl_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_config": {
                    "episodes": episodes,
                    "steps_per_episode": steps_per_episode,
                    "alpha": alpha,
                    "gamma": gamma,
                    "seed": seed,
                    "elapsed_seconds": round(elapsed, 3),
                    "num_states_observed": len(policy.q_table),
                },
                "history": history,
            },
            f,
            indent=2,
        )

    # Generate learning curve SVG
    svg_path = output_dir / "rl_learning_curve.svg"
    generate_svg_learning_curve(history, svg_path)

    # Validation evaluation on held-out validation scenarios
    val_scenarios = generate_scenarios(val_seed, count=25)
    val_rewards = []
    for s in val_scenarios:
        _, val_res, fb, _ = policy.evaluate_on_state(s)
        val_rewards.append(fb.reward)

    mean_val_reward = sum(val_rewards) / len(val_rewards)

    first_10_mean = sum(h["episode_reward"] for h in history[:10]) / 10.0
    last_10_mean = sum(h["episode_reward"] for h in history[-10:]) / 10.0

    print(f"Training completed in {elapsed:.2f} seconds ({len(policy.q_table)} states observed).")
    print(f"Initial 10-ep mean reward: {first_10_mean:.3f}")
    print(f"Final 10-ep mean reward:   {last_10_mean:.3f}")
    print(f"Validation mean reward:    {mean_val_reward:.3f} (across 25 validation scenarios)")
    print(f"Saved Q-table:        {qtable_path}")
    print(f"Saved Training Log:   {history_path}")
    print(f"Saved Learning Curve: {svg_path}")

    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tabular Q-learning policy on synthetic episodes.")
    parser.add_argument("--episodes", type=int, default=RL_DEFAULT_TRAIN_EPISODES, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=RL_STEPS_PER_EPISODE, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=SEED_TRAIN, help="Random seed for training episodes")
    args = parser.parse_args()

    train_rl(episodes=args.episodes, steps_per_episode=args.steps, seed=args.seed)
