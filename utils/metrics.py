"""
metrics.py
==========

Functions to compute evaluation metrics for inventory management
experiments.  These metrics summarise costs, service level and the
bullwhip effect across an entire episode.  They are designed to be
used after running evaluation episodes with trained policies.

The module depends only on basic Python and Numpy structures and on
functions from :mod:`inventory_management_RL_Lot.envs.reward_functions`.
"""

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from ..envs.reward_functions import bullwhip_effect


def compute_episode_costs(reward_history: List[List[float]]) -> List[float]:
    """Compute total cost per agent over an episode given reward history.

    The environment returns negative costs as rewards.  To obtain
    positive cost values, this function sums the negative of each
    reward across the episode for each agent.

    Args:
        reward_history: List of rewards for each time step, where each
            element is a list of per-agent rewards.

    Returns:
        List of total costs for each agent.
    """
    num_agents = len(reward_history[0])
    totals = [0.0 for _ in range(num_agents)]
    for rewards in reward_history:
        for i, r in enumerate(rewards):
            # rewards[i] is a list [r_value] in our environment API
            cost = -r[0] if isinstance(r, list) else -r
            totals[i] += cost
    return totals


def compute_bullwhip(order_history: List[List[int]]) -> List[float]:
    """Compute bullwhip coefficients from order history."""
    return bullwhip_effect(order_history)


def summarise_evaluation(env: Any) -> Dict[str, Any]:
    """Summarise evaluation results stored in the environment.

    Extracts the bullwhip effect results (if any) and returns them in
    a dictionary.  This helper can be extended to include other
    environment-specific metrics.
    """
    summary: Dict[str, Any] = {}
    if hasattr(env, "get_eval_bw_res"):
        summary["bullwhip"] = env.get_eval_bw_res()
    return summary
