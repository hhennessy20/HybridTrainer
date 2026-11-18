"""Tabular Q-learning baseline policy for HybridTrainer.

Implements a compact, CPU-based discrete Q-learning agent with transparent
state and action discretization, training on synthetic user episodes.
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from Baseline.environment.schemas import UserState, validate_workout_action
from Baseline.environment.reward import compute_simulated_feedback, step_state
from Baseline.environment.exercises import get_exercises_for_equipment
from Baseline.policies.base import BasePolicy
from Baseline.config import (
    RL_LEARNING_RATE,
    RL_DISCOUNT_FACTOR,
    RL_EPSILON_START,
    RL_EPSILON_END,
    RL_EPSILON_DECAY,
    RL_QTABLE_PATH,
)

# Discrete macro-actions
DISCRETE_ACTIONS = [
    # idx: (name, category, target_intensity, duration_ratio, volume)
    {"action_id": 0, "name": "Active Recovery", "category": "recovery", "intensity": 3.0, "dur_ratio": 0.50, "volume": "low"},
    {"action_id": 1, "name": "Mobility & Stretch", "category": "flexibility", "intensity": 3.5, "dur_ratio": 0.65, "volume": "low"},
    {"action_id": 2, "name": "Light Endurance", "category": "endurance", "intensity": 4.5, "dur_ratio": 0.70, "volume": "low"},
    {"action_id": 3, "name": "Moderate Endurance", "category": "endurance", "intensity": 6.5, "dur_ratio": 0.85, "volume": "moderate"},
    {"action_id": 4, "name": "Hard Endurance", "category": "endurance", "intensity": 8.0, "dur_ratio": 1.00, "volume": "high"},
    {"action_id": 5, "name": "Moderate Strength", "category": "strength", "intensity": 6.0, "dur_ratio": 0.80, "volume": "moderate"},
    {"action_id": 6, "name": "Heavy Strength", "category": "strength", "intensity": 8.0, "dur_ratio": 1.00, "volume": "high"},
    {"action_id": 7, "name": "High-Intensity Interval", "category": "hiit", "intensity": 7.5, "dur_ratio": 0.65, "volume": "high"},
    {"action_id": 8, "name": "General Circuit", "category": "general", "intensity": 5.5, "dur_ratio": 0.75, "volume": "moderate"},
]

NUM_ACTIONS = len(DISCRETE_ACTIONS)

class TabularQLearningPolicy(BasePolicy):
    def __init__(self, name: str = "TabularRL", qtable_path: Optional[Path] = None):
        super().__init__(name=name)
        self.qtable_path = Path(qtable_path) if qtable_path else RL_QTABLE_PATH
        # Q-table: key "fatigue_bin,time_bin,fitness_bin,goal_bin" -> list of float Q-values
        self.q_table: Dict[str, List[float]] = {}
        self.training_history: List[Dict[str, Any]] = []

        if self.qtable_path and self.qtable_path.exists():
            self.load(self.qtable_path)

    @staticmethod
    def discretize_state(state: UserState) -> str:
        """Map continuous/categorical UserState into discrete key (fatigue, time, fitness, goal)."""
        # Fatigue bin (3 bins)
        if state.fatigue < 0.35:
            f_bin = 0
        elif state.fatigue < 0.70:
            f_bin = 1
        else:
            f_bin = 2

        # Time bin (3 bins)
        if state.available_time_min <= 30:
            t_bin = 0
        elif state.available_time_min <= 50:
            t_bin = 1
        else:
            t_bin = 2

        # Fitness level (3 bins)
        fit_bin = {"beginner": 0, "intermediate": 1, "advanced": 2}.get(state.fitness_level, 1)

        # Goal (4 bins)
        g_bin = {"endurance": 0, "strength": 1, "general_fitness": 2, "weight_loss": 3}.get(state.goal, 2)

        return f"{f_bin},{t_bin},{fit_bin},{g_bin}"

    def get_q_values(self, state_key: str) -> List[float]:
        """Return Q-values for state_key, initializing to 0.0 if unseen."""
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * NUM_ACTIONS
        return self.q_table[state_key]

    def select_action_idx(self, state: UserState, epsilon: float = 0.0, rng: Optional[random.Random] = None) -> int:
        """Epsilon-greedy action selection."""
        r = rng if rng else random
        if r.random() < epsilon:
            return r.randint(0, NUM_ACTIONS - 1)
        state_key = self.discretize_state(state)
        q_vals = self.get_q_values(state_key)
        max_q = max(q_vals)
        best_indices = [i for i, q in enumerate(q_vals) if abs(q - max_q) < 1e-9]
        return r.choice(best_indices)

    def action_to_dict(self, action_idx: int, state: UserState) -> Dict[str, Any]:
        """Map discrete action index to concrete action dictionary matching available equipment."""
        action_spec = DISCRETE_ACTIONS[action_idx]
        cat = action_spec["category"]
        duration = int(round(state.available_time_min * action_spec["dur_ratio"]))
        duration = max(15, min(state.available_time_min, duration))
        intensity = action_spec["intensity"]
        volume = action_spec["volume"]

        # Select exercises matching user available equipment
        avail_exercises = get_exercises_for_equipment(state.available_equipment)
        filtered = [ex for ex in avail_exercises if ex.category.startswith(cat) or cat.startswith(ex.category)]
        if not filtered:
            filtered = avail_exercises

        picked = filtered[:2]
        chosen = []
        for ex in picked:
            item: Dict[str, Any] = {
                "name": ex.name,
                "category": ex.category,
                "equipment": ex.required_equipment,
            }
            if ex.default_duration_min is not None:
                item["duration_min"] = max(5, duration // max(1, len(picked)))
            else:
                item["sets"] = ex.default_sets or 3
                item["reps"] = ex.default_reps or 10
            chosen.append(item)

        return {
            "workout_type": cat,
            "target_intensity": intensity,
            "estimated_duration_min": duration,
            "volume": volume,
            "exercises": chosen,
            "rationale": (
                f"Q-learning policy action #{action_idx} ({action_spec['name']}): "
                f"intensity {intensity:.1f}, duration {duration} min, volume {volume}."
            ),
        }

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Greedy recommendation at evaluation time."""
        action_idx = self.select_action_idx(state, epsilon=0.0)
        return self.action_to_dict(action_idx, state)

    def train(
        self,
        num_episodes: int = 500,
        steps_per_episode: int = 5,
        alpha: float = RL_LEARNING_RATE,
        gamma: float = RL_DISCOUNT_FACTOR,
        eps_start: float = RL_EPSILON_START,
        eps_end: float = RL_EPSILON_END,
        eps_decay: float = RL_EPSILON_DECAY,
        base_seed: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Train Q-learning agent on synthetic user episodes."""
        from Baseline.environment.synthetic_users import generate_synthetic_scenario

        rng = random.Random(base_seed)
        eps = eps_start
        history = []

        for ep in range(num_episodes):
            ep_seed = base_seed + ep
            state = generate_synthetic_scenario(seed=ep_seed)
            ep_reward = 0.0

            for _ in range(steps_per_episode):
                s_key = self.discretize_state(state)
                action_idx = self.select_action_idx(state, epsilon=eps, rng=rng)
                action_dict = self.action_to_dict(action_idx, state)

                # Shared validation and simulated feedback
                val_result = validate_workout_action(action_dict, state)
                feedback = compute_simulated_feedback(state, action_dict, val_result)
                reward = feedback.reward
                ep_reward += reward

                # Next state
                next_state = step_state(state, action_dict, feedback)
                next_key = self.discretize_state(next_state)

                # Tabular Q-update
                q_vals = self.get_q_values(s_key)
                next_q_vals = self.get_q_values(next_key)
                max_next_q = max(next_q_vals)

                old_q = q_vals[action_idx]
                q_vals[action_idx] = old_q + alpha * (reward + gamma * max_next_q - old_q)

                state = next_state

            eps = max(eps_end, eps * eps_decay)
            history.append({
                "episode": ep + 1,
                "episode_reward": round(ep_reward, 4),
                "mean_step_reward": round(ep_reward / steps_per_episode, 4),
                "epsilon": round(eps, 4),
            })

        self.training_history = history
        return history

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save Q-table and metadata to JSON."""
        target = Path(filepath) if filepath else self.qtable_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_states_observed": len(self.q_table),
            "num_actions": NUM_ACTIONS,
            "actions": DISCRETE_ACTIONS,
            "q_table": self.q_table,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Optional[Path] = None) -> None:
        """Load Q-table from JSON."""
        target = Path(filepath) if filepath else self.qtable_path
        if not target.exists():
            return
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.q_table = data.get("q_table", {})
