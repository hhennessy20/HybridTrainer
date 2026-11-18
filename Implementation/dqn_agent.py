"""Dueling Double Deep Q-Network (DDQN) Agent for HybridTrainer."""

import random
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Baseline.environment.schemas import UserState, WorkoutAction, validate_workout_action
from Baseline.environment.exercises import get_exercises_for_equipment, CATALOGUE_BY_NAME
from Baseline.policies.rl_qlearning import DISCRETE_ACTIONS, NUM_ACTIONS
from Baseline.policies.base import BasePolicy
from Implementation.model import DuelingDQN
from Implementation.state_encoder import encode_user_state, encode_user_state_tensor, STATE_DIM


class DuelingDDQNAgent(BasePolicy):
    """Deep Q-Network agent operating over continuous 30-dim user state vectors.
    
    Implements:
    - Dueling Q-architecture (V(s) + A(s, a))
    - Double Q-learning target formulation to eliminate overestimation bias
    - Dynamic constraint-aware action masking
    - Huber (Smooth L1) loss with gradient clipping
    - Epsilon-greedy exploration with exponential decay
    """

    def __init__(
        self,
        name: str = "DuelingDQN",
        state_dim: int = STATE_DIM,
        action_dim: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        super().__init__(name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device or torch.device("cpu")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        # Primary and Target Networks
        self.policy_net = DuelingDQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()

        self.total_steps = 0
        self.training_losses: List[float] = []

        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load(self.checkpoint_path)

    @staticmethod
    def compute_action_mask(state: UserState) -> np.ndarray:
        """Compute binary action feasibility mask based on user state constraints.
        
        Ensures that:
        1. If user is severely fatigued (>=0.80), dangerous high-intensity actions are masked out.
        2. If available time is under 25 minutes, 1.0 duration ratio actions are masked out.
        3. If a category is in user's dispreferred categories, mask it out.
        4. Active recovery and mobility are always retained as safe fallbacks.
        """
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)

        # 1. High fatigue protection
        if state.fatigue >= 0.80:
            for idx, act in enumerate(DISCRETE_ACTIONS):
                if act["intensity"] >= 8.0:
                    mask[idx] = 0.0

        # 2. Time constraint protection
        if state.available_time_min < 25:
            for idx, act in enumerate(DISCRETE_ACTIONS):
                if act["dur_ratio"] >= 0.90:
                    mask[idx] = 0.0

        # 3. Dispreferred category avoidance
        dispref = {d.lower().strip() for d in state.dispreferred_categories}
        for idx, act in enumerate(DISCRETE_ACTIONS):
            if act["category"].lower() in dispref:
                mask[idx] = 0.0

        # Safety: at least Active Recovery (idx 0) and Mobility (idx 1) are always valid
        mask[0] = 1.0
        mask[1] = 1.0
        return mask

    def select_action_idx(
        self,
        state: UserState,
        epsilon: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Select action via epsilon-greedy policy with action masking."""
        r = rng if rng else random
        mask_np = self.compute_action_mask(state)

        if r.random() < epsilon:
            valid_indices = [i for i, m in enumerate(mask_np) if m > 0.5]
            return r.choice(valid_indices)

        state_tensor = encode_user_state_tensor(state, device=self.device)
        mask_tensor = torch.from_numpy(mask_np).to(self.device)

        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor, action_mask=mask_tensor)
            action_idx = int(torch.argmax(q_values, dim=-1).item())

        return action_idx

    def action_to_dict(self, action_idx: int, state: UserState) -> Dict[str, Any]:
        """Convert discrete action index into structured workout dictionary."""
        spec = DISCRETE_ACTIONS[action_idx]
        category = spec["category"]
        duration = int(round(state.available_time_min * spec["dur_ratio"]))
        duration = max(15, min(state.available_time_min, duration))
        intensity = float(spec["intensity"])
        volume = spec["volume"]

        # Filter exercises matching user equipment and category
        avail_exercises = get_exercises_for_equipment(state.available_equipment)
        matching = [ex for ex in avail_exercises if ex.category.startswith(category) or category.startswith(ex.category)]
        if not matching:
            matching = avail_exercises

        picked = matching[:2]
        chosen = []
        for ex in picked:
            item = {
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
            "workout_type": category,
            "target_intensity": intensity,
            "estimated_duration_min": duration,
            "volume": volume,
            "exercises": chosen,
            "rationale": (
                f"Dueling DQN action #{action_idx} ({spec['name']}): "
                f"intensity {intensity:.1f}, duration {duration}m, volume {volume}."
            ),
        }

    def recommend(self, state: UserState) -> Dict[str, Any]:
        """Greedy evaluation recommendation."""
        action_idx = self.select_action_idx(state, epsilon=0.0)
        return self.action_to_dict(action_idx, state)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform one Double-DQN gradient update step.
        
        Double Q-Learning objective:
            target = r + gamma * (1 - done) * Q_target(s', argmax_{a'} Q_online(s', a'))
        """
        self.policy_net.train()
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]
        action_masks = batch.get("action_masks")
        next_action_masks = batch.get("next_action_masks")

        # Current predicted Q(s, a)
        all_q = self.policy_net(states, action_mask=action_masks)
        current_q = all_q.gather(1, actions)

        # Target Q calculation with Double DQN decoupling
        with torch.no_grad():
            # Online network selects the greedy action for next state: a* = argmax Q_online(s', a')
            next_q_online = self.policy_net(next_states, action_mask=next_action_masks)
            next_actions = torch.argmax(next_q_online, dim=-1, keepdim=True)

            # Target network evaluates that chosen action: Q_target(s', a*)
            next_q_target_all = self.target_net(next_states, action_mask=next_action_masks)
            next_q = next_q_target_all.gather(1, next_actions)

            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft update of target network parameters: theta_targ = tau * theta + (1 - tau) * theta_targ
        with torch.no_grad():
            for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.total_steps += 1
        loss_val = float(loss.item())
        self.training_losses.append(loss_val)
        return loss_val

    def save(self, filepath: Path):
        """Save model checkpoint and optimizer state."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            filepath,
        )

    def load(self, filepath: Path):
        """Load model checkpoint."""
        try:
            ckpt = torch.load(filepath, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["model_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)
