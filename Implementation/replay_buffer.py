"""Experience replay buffer for Deep Q-Network training."""

import random
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity experience replay buffer storing state transitions."""

    def __init__(self, capacity: int = 20000, seed: int = 42):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: Optional[np.ndarray] = None,
        next_action_mask: Optional[np.ndarray] = None,
    ):
        """Save a transition tuple."""
        transition = {
            "state": state.astype(np.float32),
            "action": int(action),
            "reward": float(reward),
            "next_state": next_state.astype(np.float32),
            "done": float(done),
            "action_mask": action_mask.astype(np.float32) if action_mask is not None else None,
            "next_action_mask": next_action_mask.astype(np.float32) if next_action_mask is not None else None,
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions uniformly at random."""
        batch = self.rng.sample(self.buffer, batch_size)

        states = torch.from_numpy(np.array([b["state"] for b in batch], dtype=np.float32)).to(device)
        actions = torch.tensor([b["action"] for b in batch], dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.from_numpy(np.array([b["next_state"] for b in batch], dtype=np.float32)).to(device)
        dones = torch.tensor([b["done"] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

        result = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

        if batch[0]["action_mask"] is not None:
            result["action_masks"] = torch.from_numpy(
                np.array([b["action_mask"] for b in batch], dtype=np.float32)
            ).to(device)
        if batch[0]["next_action_mask"] is not None:
            result["next_action_masks"] = torch.from_numpy(
                np.array([b["next_action_mask"] for b in batch], dtype=np.float32)
            ).to(device)

        return result

    def __len__(self) -> int:
        return len(self.buffer)

