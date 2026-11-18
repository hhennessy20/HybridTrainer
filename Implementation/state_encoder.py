"""State encoding and normalization module for HybridTrainer Deep RL.

Transforms continuous, categorical, and multi-hot attributes of UserState into
a normalized continuous feature vector s_t in R^30.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.environment.schemas import UserState

# Canonical vocabulary definitions
FITNESS_LEVELS = ["beginner", "intermediate", "advanced"]
GOALS = ["endurance", "strength", "general_fitness", "weight_loss"]
EQUIPMENT_LIST = [
    "bodyweight",
    "dumbbells",
    "barbell",
    "kettlebell",
    "resistance_bands",
    "pullup_bar",
    "treadmill",
    "stationary_bike",
]
WORKOUT_CATEGORIES = ["endurance", "strength", "hiit", "flexibility", "recovery"]

STATE_DIM = (
    5  # continuous features: fatigue, stress, completion_rate, target_rpe, available_time
    + len(FITNESS_LEVELS)  # 3
    + len(GOALS)  # 4
    + len(EQUIPMENT_LIST)  # 8
    + len(WORKOUT_CATEGORIES)  # 5 (preferred)
    + len(WORKOUT_CATEGORIES)  # 5 (dispreferred)
)  # Total: 30 dimensions


def encode_user_state(state: UserState) -> np.ndarray:
    """Encode a UserState dataclass into a continuous 30-dim numpy feature vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # 1. Continuous normalized scalar features (5 dims)
    vec[idx] = float(np.clip(state.fatigue, 0.0, 1.0))
    idx += 1
    vec[idx] = float(np.clip(state.stress, 0.0, 1.0))
    idx += 1
    vec[idx] = float(np.clip(state.recent_completion_rate, 0.0, 1.0))
    idx += 1
    # Target RPE normalized [1.0, 10.0] -> [0.0, 1.0]
    rpe_norm = (state.target_rpe - 1.0) / 9.0
    vec[idx] = float(np.clip(rpe_norm, 0.0, 1.0))
    idx += 1
    # Available time normalized [15, 90] -> [0.0, 1.0]
    time_norm = (state.available_time_min - 15.0) / 75.0
    vec[idx] = float(np.clip(time_norm, 0.0, 1.0))
    idx += 1

    # 2. Fitness level one-hot (3 dims)
    fit_level = state.fitness_level.lower().strip()
    for fl_idx, fl in enumerate(FITNESS_LEVELS):
        if fl == fit_level:
            vec[idx + fl_idx] = 1.0
            break
    idx += len(FITNESS_LEVELS)

    # 3. Goal one-hot (4 dims)
    goal = state.goal.lower().strip()
    for g_idx, g in enumerate(GOALS):
        if g == goal:
            vec[idx + g_idx] = 1.0
            break
    idx += len(GOALS)

    # 4. Equipment multi-hot (8 dims)
    avail_eq = {e.lower().strip() for e in state.available_equipment}
    avail_eq.add("bodyweight")  # Always available
    for eq_idx, eq in enumerate(EQUIPMENT_LIST):
        if eq in avail_eq:
            vec[idx + eq_idx] = 1.0
    idx += len(EQUIPMENT_LIST)

    # 5. Preferred categories multi-hot (5 dims)
    pref_cats = {c.lower().strip() for c in state.preferred_categories}
    for c_idx, c in enumerate(WORKOUT_CATEGORIES):
        if c in pref_cats:
            vec[idx + c_idx] = 1.0
    idx += len(WORKOUT_CATEGORIES)

    # 6. Dispreferred categories multi-hot (5 dims)
    dispref_cats = {c.lower().strip() for c in state.dispreferred_categories}
    for c_idx, c in enumerate(WORKOUT_CATEGORIES):
        if c in dispref_cats:
            vec[idx + c_idx] = 1.0
    idx += len(WORKOUT_CATEGORIES)

    return vec


def encode_user_state_tensor(state: UserState, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Encode UserState directly into a 2D PyTorch Tensor (1, STATE_DIM)."""
    arr = encode_user_state(state)
    return torch.from_numpy(arr).unsqueeze(0).to(device)
