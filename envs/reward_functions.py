"""
reward_functions.py
===================
Utility functions for computing costs and service metrics.
"""
from __future__ import annotations
from typing import List
import numpy as np

def holding_cost(inventory: List[int], costs: List[float]) -> float:
    return float(sum(inv * c for inv, c in zip(inventory, costs)))

def backlog_cost(backlog: List[int], costs: List[float]) -> float:
    return float(sum(b * c for b, c in zip(backlog, costs)))

def ordering_cost(actions: List[int], fixed_cost: float) -> float:
    return float(sum(fixed_cost for a in actions if a > 0))

def bullwhip_effect(order_history: List[List[int]]) -> List[float]:
    cvs = []
    for hist in order_history:
        if not hist or np.mean(hist) < 1e-6:
            cvs.append(0.0)
        else:
            cvs.append(float(np.std(hist) / np.mean(hist)))
    return cvs

def fill_rate(demands: List[int], sales: List[int]) -> float:
    total_demand = float(sum(demands))
    total_sales = float(sum(sales))
    return 1.0 if total_demand == 0 else total_sales / total_demand

def service_level(backlog: List[int]) -> float:
    """
    FIXED: Service Level = 1 - (Total Backlog / Estimated Total Demand)
    """
    total_backlog = float(sum(backlog))
    estimated_total_demand = 1000.0 
    
    sl = 1.0 - (total_backlog / estimated_total_demand)
    return max(0.0, sl)
