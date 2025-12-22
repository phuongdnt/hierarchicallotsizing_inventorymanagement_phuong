"""
reward_functions.py
===================

Utility functions for computing costs and service metrics in multi–echelon
inventory management.  These functions can be used independently from
the environment logic and are particularly useful during evaluation
outside the training loop.  The functions operate on basic Python
structures (lists of numbers) to avoid dependencies on the environment
implementation.

The reward functions here mirror those used in :mod:`serial_env` and
:mod:`network_env` but allow external callers (e.g. evaluation scripts)
to compute metrics directly from recorded histories.
"""

from __future__ import annotations

from typing import List
import numpy as np


def holding_cost(inventory: List[int], costs: List[float]) -> float:
    """Compute the total holding cost given inventory levels and per-unit costs."""
    return float(sum(inv * c for inv, c in zip(inventory, costs)))


def backlog_cost(backlog: List[int], costs: List[float]) -> float:
    """Compute the total backlog (backorder) cost given backlog and per-unit costs."""
    return float(sum(b * c for b, c in zip(backlog, costs)))


def ordering_cost(actions: List[int], fixed_cost: float) -> float:
    """Compute the total fixed cost for non-zero orders.

    Assumes a fixed cost is incurred whenever an agent places a non-zero
    order in ``actions``.  Variable ordering costs are not included
    here; those would depend on a price schedule.
    """
    return float(sum(fixed_cost for a in actions if a > 0))


def bullwhip_effect(order_history: List[List[int]]) -> List[float]:
    """Compute the coefficient of variation of each agent's order history.

    Parameters
    ----------
    order_history : List[List[int]]
        Nested list where ``order_history[i]`` contains the sequence of
        orders placed by agent ``i`` over an episode.

    Returns
    -------
    List[float]
        List of coefficient of variation values (standard deviation divided
        by mean) for each agent.  If an agent never orders (mean = 0)
        its bullwhip metric is defined as zero.
    """
    cvs = []
    for hist in order_history:
        if not hist or np.mean(hist) < 1e-6:
            cvs.append(0.0)
        else:
            cvs.append(float(np.std(hist) / np.mean(hist)))
    return cvs


def fill_rate(demands: List[int], sales: List[int]) -> float:
    """Compute the fill rate (proportion of demand met immediately).

    Fill rate is defined as the ratio of total sales to total demand.
    Sales should include only demand that is fulfilled on time (i.e.
    not backordered or lost).  A value of 1.0 indicates all demand is
    satisfied immediately.
    """
    total_demand = float(sum(demands))
    total_sales = float(sum(sales))
    return 1.0 if total_demand == 0 else total_sales / total_demand


def service_level(backlog: List[int]) -> float:
    """Compute the service level defined as 1 minus the backorder ratio.

    Service level is a simple measure of customer satisfaction: the
    proportion of demand that is not backordered.  If the backlog list
    contains outstanding customer demand for each period, the service
    level can be computed as

    ``1.0 - (sum(backlog) / (sum(backlog) + sum(sales)))``.

    In practice, service level is often tracked per period; this
    implementation provides a coarse aggregate measure for an episode.
    """
    total_backlog = float(sum(backlog))
    return 1.0 - total_backlog / (total_backlog + 1.0) if total_backlog > 0 else 1.0
