"""Neural network architectures for HybridTrainer Deep Q-Networks (DQN).

Includes:
1. Standard Deep Q-Network (DQN) MLP
2. Dueling Deep Q-Network (DuelingDQN) with decoupled State-Value V(s) and Action-Advantage A(s, a)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture for continuous user state workout recommendation.
    
    Decouples state value V(s) from action advantages A(s, a):
        Q(s, a; theta, alpha, beta) = V(s; theta, beta) + (A(s, a; theta, alpha) - mean_{a'} A(s, a'; theta, alpha))
    
    This enables precise value learning for low-variance recovery states independent
    of specific exercise modality choices.
    """

    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature representation trunk
        self.feature_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Value stream: V(s) in R^1
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a) in R^|A|
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Dueling architecture with optional action masking.
        
        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,)
            action_mask: Optional binary tensor (batch_size, action_dim) where 1 indicates valid action.
        
        Returns:
            q_values: Tensor of shape (batch_size, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Extract shared latent state features
        features = self.feature_trunk(state)

        # Compute state value and action advantages
        values = self.value_stream(features)  # (batch_size, 1)
        advantages = self.advantage_stream(features)  # (batch_size, action_dim)

        # Dueling aggregation with centered advantage
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))

        # Apply action masking if provided (masks out infeasible actions with -1e9)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            q_values = torch.where(action_mask > 0.5, q_values, torch.tensor(-1e9, device=q_values.device))

        return q_values


class StandardDQN(nn.Module):
    """Standard Deep Q-Network baseline architecture with uniform MLP output."""

    def __init__(
        self,
        state_dim: int = 30,
        action_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass predicting Q-values directly from state."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        q_values = self.net(state)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            q_values = torch.where(action_mask > 0.5, q_values, torch.tensor(-1e9, device=q_values.device))
        return q_values

