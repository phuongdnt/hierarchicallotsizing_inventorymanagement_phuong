"""
reward_functions.py
===================
Utility functions for computing costs and service metrics.

This module provides functions to calculate various inventory management
metrics including:
- Holding cost
- Backlog (backorder) cost
- Ordering cost
- Bullwhip effect
- Fill rate
- Service level (multiple definitions)
"""
from __future__ import annotations
from typing import List
import numpy as np


def holding_cost(inventory: List[int], costs: List[float]) -> float:
    """Calculate total holding cost across all agents.
    
    Args:
        inventory: Current inventory level for each agent.
        costs: Per-unit holding cost for each agent.
    
    Returns:
        Total holding cost.
    """
    return float(sum(inv * c for inv, c in zip(inventory, costs)))


def backlog_cost(backlog: List[int], costs: List[float]) -> float:
    """Calculate total backlog (backorder) cost across all agents.
    
    Args:
        backlog: Current backlog level for each agent.
        costs: Per-unit backlog cost for each agent.
    
    Returns:
        Total backlog cost.
    """
    return float(sum(b * c for b, c in zip(backlog, costs)))


def ordering_cost(actions: List[int], fixed_cost: float) -> float:
    """Calculate total fixed ordering cost.
    
    Fixed cost is incurred whenever an agent places a non-zero order.
    
    Args:
        actions: Order quantities for each agent.
        fixed_cost: Fixed cost per order.
    
    Returns:
        Total fixed ordering cost.
    """
    return float(sum(fixed_cost for a in actions if a > 0))


def bullwhip_effect(order_history: List[List[int]]) -> List[float]:
    """Calculate the bullwhip effect coefficient for each agent.
    
    Bullwhip effect is measured as the Coefficient of Variation (CV)
    of order quantities: CV = standard_deviation / mean
    
    A CV > 1 indicates high variability (strong bullwhip).
    A CV close to 0 indicates stable ordering.
    
    Args:
        order_history: List of order sequences for each agent.
    
    Returns:
        List of bullwhip coefficients (CV) for each agent.
    """
    cvs = []
    for hist in order_history:
        if not hist or np.mean(hist) < 1e-6:
            cvs.append(0.0)
        else:
            cvs.append(float(np.std(hist) / np.mean(hist)))
    return cvs


def fill_rate(demands: List[int], fulfilled: List[int]) -> float:
    """Calculate fill rate (Type 1 Service Level).
    
    Fill Rate = Total Fulfilled / Total Demand
    
    Args:
        demands: List of demand values over time.
        fulfilled: List of fulfilled demand values over time.
    
    Returns:
        Fill rate between 0.0 and 1.0.
    """
    total_demand = float(sum(demands))
    total_fulfilled = float(sum(fulfilled))
    if total_demand == 0:
        return 1.0
    return min(1.0, total_fulfilled / total_demand)


def service_level(
    demands: List[int], 
    fulfilled: List[int]
) -> float:
    """Calculate service level (fill rate).
    
    Service Level = Total Fulfilled Demand / Total Demand
    
    This is the standard definition used in inventory management.
    A value of 1.0 means all demand was satisfied immediately.
    
    Args:
        demands: List of demand values for each period.
        fulfilled: List of fulfilled demand values for each period.
    
    Returns:
        Service level between 0.0 and 1.0.
    """
    total_demand = sum(demands)
    total_fulfilled = sum(fulfilled)
    
    if total_demand == 0:
        return 1.0
    
    return min(1.0, total_fulfilled / total_demand)


def cycle_service_level(backlogs: List[int]) -> float:
    """Calculate cycle service level (Type 2 Service Level).
    
    Cycle Service Level = Periods without stockout / Total Periods
    
    This measures the probability of not having a stockout.
    
    Args:
        backlogs: List of backlog values for each period.
    
    Returns:
        Cycle service level between 0.0 and 1.0.
    """
    if not backlogs:
        return 1.0
    
    periods_without_stockout = sum(1 for b in backlogs if b == 0)
    return periods_without_stockout / len(backlogs)


def ready_rate(
    demand: int,
    on_hand_before: int,
    received: int
) -> float:
    """Calculate ready rate for a single period.
    
    Ready Rate = min(Available, Demand) / Demand
    
    Args:
        demand: Demand in this period.
        on_hand_before: Inventory on hand before receiving.
        received: Quantity received this period.
    
    Returns:
        Ready rate for this period (0.0 to 1.0).
    """
    if demand == 0:
        return 1.0
    
    available = on_hand_before + received
    fulfilled = min(available, demand)
    return fulfilled / demand
