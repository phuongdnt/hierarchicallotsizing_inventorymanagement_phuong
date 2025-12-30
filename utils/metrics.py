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


def compute_service_levels(
    demand_history: List[List[int]], 
    fulfilled_history: List[List[int]]
) -> List[float]:
    """Compute service level (fill rate) for each agent.
    
    Service Level (Type 1 - Fill Rate) is defined as:
        Service Level = Total Fulfilled Demand / Total Demand
    
    A value of 1.0 indicates all demand was satisfied immediately.
    A value of 0.8 indicates 80% of demand was fulfilled, 20% was backordered.
    
    Args:
        demand_history: List of demand sequences for each agent.
            Each element is a list of demand values at each time step.
        fulfilled_history: List of fulfilled demand sequences for each agent.
            Each element is a list of fulfilled amounts at each time step.
    
    Returns:
        List of service levels (0.0 to 1.0) for each agent.
    
    Example:
        >>> demand_history = [[10, 15, 20], [8, 12, 16]]
        >>> fulfilled_history = [[10, 12, 18], [8, 10, 14]]
        >>> compute_service_levels(demand_history, fulfilled_history)
        [0.889, 0.889]  # (10+12+18)/(10+15+20) and (8+10+14)/(8+12+16)
    """
    service_levels = []
    num_agents = len(demand_history)
    
    for i in range(num_agents):
        total_demand = sum(demand_history[i])
        total_fulfilled = sum(fulfilled_history[i])
        
        if total_demand > 0:
            sl = total_fulfilled / total_demand
            # Clamp to [0, 1] to handle edge cases
            sl = max(0.0, min(1.0, sl))
        else:
            sl = 1.0  # No demand means perfect service
        
        service_levels.append(sl)
    
    return service_levels


def compute_service_levels_from_backlog(
    backlog_history: List[List[int]],
    demand_history: List[List[int]]
) -> List[float]:
    """Alternative service level calculation using backlog history.
    
    Service Level = 1 - (Total Backlog Created / Total Demand)
    
    This is an approximation when fulfilled_history is not available.
    Note: This may be less accurate than the direct fill rate calculation
    because backlog accumulates over time.
    
    Args:
        backlog_history: List of backlog sequences for each agent.
        demand_history: List of demand sequences for each agent.
    
    Returns:
        List of service levels (0.0 to 1.0) for each agent.
    """
    service_levels = []
    num_agents = len(backlog_history)
    
    for i in range(num_agents):
        total_demand = sum(demand_history[i])
        # Sum of new backlog created each period
        # Note: backlog[t] includes carry-over from previous periods
        # This is an approximation
        total_backlog = sum(backlog_history[i])
        
        if total_demand > 0:
            # This gives average backlog rate, not exact fill rate
            sl = max(0.0, 1.0 - (total_backlog / total_demand))
        else:
            sl = 1.0
        
        service_levels.append(sl)
    
    return service_levels


def compute_cycle_service_level(
    backlog_history: List[List[int]]
) -> List[float]:
    """Compute Type 2 Service Level (Cycle Service Level).
    
    Cycle Service Level = Number of periods without stockout / Total periods
    
    This measures the probability of not having a stockout in any given period.
    
    Args:
        backlog_history: List of backlog sequences for each agent.
    
    Returns:
        List of cycle service levels (0.0 to 1.0) for each agent.
    """
    service_levels = []
    num_agents = len(backlog_history)
    
    for i in range(num_agents):
        backlogs = backlog_history[i]
        if not backlogs:
            service_levels.append(1.0)
            continue
        
        total_periods = len(backlogs)
        periods_without_stockout = sum(1 for b in backlogs if b == 0)
        
        sl = periods_without_stockout / total_periods
        service_levels.append(sl)
    
    return service_levels


def compute_average_inventory(inventory_history: List[List[int]]) -> List[float]:
    """Compute average inventory level for each agent.
    
    Args:
        inventory_history: List of inventory sequences for each agent.
    
    Returns:
        List of average inventory levels for each agent.
    """
    avg_inventories = []
    for inv_seq in inventory_history:
        if inv_seq:
            avg_inventories.append(float(np.mean(inv_seq)))
        else:
            avg_inventories.append(0.0)
    return avg_inventories


def compute_inventory_turnover(
    total_demand: List[int],
    avg_inventory: List[float]
) -> List[float]:
    """Compute inventory turnover ratio for each agent.
    
    Inventory Turnover = Total Demand / Average Inventory
    
    Higher turnover indicates more efficient inventory management.
    
    Args:
        total_demand: Total demand faced by each agent.
        avg_inventory: Average inventory level for each agent.
    
    Returns:
        List of turnover ratios for each agent.
    """
    turnovers = []
    for demand, inv in zip(total_demand, avg_inventory):
        if inv > 0:
            turnovers.append(demand / inv)
        else:
            turnovers.append(float('inf'))  # Perfect turnover if no inventory held
    return turnovers


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


def compute_all_metrics(
    reward_history: List[List[float]],
    order_history: List[List[int]],
    demand_history: List[List[int]],
    fulfilled_history: List[List[int]],
    backlog_history: List[List[int]]
) -> Dict[str, List[float]]:
    """Compute all evaluation metrics in one call.
    
    Args:
        reward_history: Rewards per timestep per agent.
        order_history: Orders per agent over time.
        demand_history: Demand faced per agent over time.
        fulfilled_history: Fulfilled demand per agent over time.
        backlog_history: Backlog per agent over time.
    
    Returns:
        Dictionary with all computed metrics.
    """
    return {
        "costs": compute_episode_costs(reward_history),
        "bullwhip": compute_bullwhip(order_history),
        "fill_rate_service_level": compute_service_levels(demand_history, fulfilled_history),
        "cycle_service_level": compute_cycle_service_level(backlog_history),
    }
