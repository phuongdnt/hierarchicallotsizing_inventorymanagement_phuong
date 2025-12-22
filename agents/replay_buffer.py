"""
replay_buffer.py
=================

This module contains a simple experience buffer for storing trajectories
in episodic environments.  For policy gradient methods such as PPO,
trajectories are collected and then used once to compute returns and
advantages; the buffer is then cleared.  The implementation here is
minimal and does not support random sampling – it simply stores
transitions in order of collection.  For off–policy algorithms (e.g.
DQN, MADDPG) you may wish to extend this class to allow sampling
arbitrary batches.

Each transition stored in the buffer is a dictionary containing the
observations, actions, log probabilities, rewards, next observations
and done flags for all agents at one time step.
"""

from __future__ import annotations

from typing import List, Dict, Any


class ReplayBuffer:
    """A simple FIFO replay buffer for storing full trajectories."""

    def __init__(self) -> None:
        self.storage: List[Dict[str, Any]] = []

    def add(self, transition: Dict[str, Any]) -> None:
        """Append a transition to the buffer.

        Parameters
        ----------
        transition: Dict[str, Any]
            A dictionary with keys ``obs``, ``actions``, ``log_probs``,
            ``rewards``, ``next_obs``, ``done``.  Each value should be
            a list (length equal to the number of agents) or a nested
            structure.
        """
        self.storage.append(transition)

    def clear(self) -> None:
        """Remove all stored transitions."""
        self.storage.clear()

    def __len__(self) -> int:
        return len(self.storage)

    def as_dict(self) -> Dict[str, List[Any]]:
        """Convert stored transitions into a dict of lists for training.

        Returns a dictionary mapping each field name to a list of that
        field across all stored transitions.  This makes it easier to
        convert the buffer into tensors during training.
        """
        keys = ["obs", "actions", "log_probs", "rewards", "next_obs", "done"]
        data = {k: [] for k in keys}
        for item in self.storage:
            for k in keys:
                data[k].append(item[k])
        return data
