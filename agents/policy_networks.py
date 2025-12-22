"""
policy_networks.py
==================

Neural network definitions for multi–agent reinforcement learning in the
inventory management project.  Actors (policies) map from per–agent
observations to distributions over discrete actions.  The centralized
critic maps from a concatenated global state to a scalar value estimate.

These networks are implemented in PyTorch and kept deliberately
lightweight.  They support both feedforward (MLP) and recurrent
architectures; however only the MLP actor is used by default.  The
critic is also an MLP.

"""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLPActor(nn.Module):
    """Simple multi–layer perceptron policy for discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits over actions.

        Args:
            obs: Tensor of shape (batch, obs_dim).
        Returns:
            Tensor of shape (batch, action_dim) representing unnormalized
            log probabilities (logits).
        """
        return self.net(obs)

    def get_action(self, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Sample an action given a single observation.

        Args:
            obs: 1D tensor of size ``obs_dim``.

        Returns:
            A tuple ``(action, log_prob)`` where ``action`` is an int and
            ``log_prob`` is a tensor of shape ``()`` (scalar).
        """
        logits = self.forward(obs.unsqueeze(0))  # Add batch dimension
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy of given actions.

        Args:
            obs: Tensor of shape (batch, obs_dim).
            actions: Tensor of shape (batch,) containing integer actions.

        Returns:
            Tuple ``(log_probs, entropy)`` where both have shape (batch,).
        """
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class MLPCritic(nn.Module):
    """Centralized critic that takes the global state and returns a scalar value."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate for given state.

        Args:
            state: Tensor of shape (batch, state_dim).

        Returns:
            Tensor of shape (batch, 1) containing scalar value estimates.
        """
        return self.net(state)
