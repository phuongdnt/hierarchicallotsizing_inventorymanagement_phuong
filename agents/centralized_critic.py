"""
centralized_critic.py
=====================

This module defines a centralized critic wrapper for multi–agent
reinforcement learning.  The critic estimates the state value function
for the entire system by operating on the concatenated observations of
all agents.  It is implemented as a thin wrapper around the
MLPCritic network defined in :mod:`policy_networks`.

The wrapper exposes methods to compute value estimates and update
parameters via a provided optimizer.  It does not maintain its own
optimizer; training code should construct an optimizer and pass the
critic parameters to it.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
from .policy_networks import MLPCritic


class CentralizedCritic:
    """Wrapper around a PyTorch critic network for centralized value estimation."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        self.net = MLPCritic(state_dim, hidden_dim)

    def evaluate(self, states: torch.Tensor) -> torch.Tensor:
        """Return value estimates for a batch of global states.

        Args:
            states: Tensor of shape (batch_size, state_dim).

        Returns:
            Tensor of shape (batch_size, 1) containing value estimates.
        """
        return self.net(states)

    def parameters(self) -> List[nn.Parameter]:
        """Return the parameters of the underlying network."""
        return list(self.net.parameters())
