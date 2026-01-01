"""
hybrid_planner.py
=================

This module defines a helper class for combining reinforcement learning
policies with heuristic lot–sizing algorithms.  During training or
deployment, an RL agent may propose order quantities that violate
operational constraints (e.g. minimum lot size, fixed cost penalties).
The :class:`HybridPlanner` intercepts these actions and refines them
using a specified heuristic before sending them to the environment.

Currently the planner supports two heuristics: a genetic algorithm
(:func:`ga_lotsizing.optimise_order`) and a placeholder DALSA method
(:func:`dalsa_module.dalsa_optimize`).  You can extend the planner to
incorporate additional heuristics or decision rules.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .ga_lotsizing import optimise_order
from .dalsa_module import dalsa_optimize


class HybridPlanner:
    """Refine RL actions using a lot–sizing heuristic."""

    def __init__(
        self,
        env: Any,
        horizon: int = 5,
        use_ga: bool = True,
        use_dalsa: bool = False,
        ga_params: Optional[dict] = None,
    ) -> None:
        """Create a planner.

        Args:
            env: The environment instance (must expose inventory, backlog and
                cost parameters).  Used to read current state and forecast.
            horizon: Number of future periods considered by the heuristic.
            use_ga: Whether to use the genetic algorithm heuristic.
            use_dalsa: Whether to use the DALSA heuristic.  If both
                ``use_ga`` and ``use_dalsa`` are True, GA takes precedence.
            ga_params: Optional dictionary of parameters for the GA
                (keys: ``pop_size``, ``generations``, ``mutation_rate``).
        """
        self.env = env
        self.horizon = horizon
        self.use_ga = use_ga
        self.use_dalsa = use_dalsa
        self.ga_params = ga_params or {}

    def refine_actions(self, actions: List[int]) -> List[int]:
        """Refine a list of RL actions using the chosen heuristic.

        Each agent's proposed action (order quantity) is potentially
        replaced by a near–optimal quantity computed by GA or DALSA.  The
        heuristic uses the agent's current inventory, a simple demand
        forecast and cost parameters from the environment.  Agents that
        propose an action of zero will retain a zero order.

        Returns:
            A list of refined order quantities, same length as ``actions``.
        """
        refined = actions.copy()
        # Retrieve environment state
        inventory = self.env.inventory
        backlog = self.env.backlog
        # Determine demand forecasts for each agent
        # For leaf nodes in a network environment, use external demand
        # For serial environment, forecast using upcoming demand_list
        forecasts: List[List[int]] = []
        def build_forecast(seq: List[int]) -> List[int]:
            if not seq:
                return [0 for _ in range(self.horizon)]
            start = self.env.step_num
            return [seq[t] if t < len(seq) else seq[-1] for t in range(start, start + self.horizon)]

        if hasattr(self.env, "external_demand_list") and self.env.external_demand_list:
            # network environment: external_demand_list is a list of lists per retailer
            # Construct a forecast for each agent
            leaf_indices = getattr(self.env, "leaf_indices", [])
            # For each agent, if leaf: use external demand; else forecast sum of child RL actions
            for i in range(self.env.agent_num):
                if i in leaf_indices:
                    idx = leaf_indices.index(i)
                    seq = self.env.external_demand_list[idx]
                    forecast = build_forecast(seq)
                else:
                    # For internal nodes, approximate demand forecast as constant equal
                    # to sum of recent orders from children or use current backlog
                    # Here we simply repeat backlog/demand
                    demand_est = int(sum(actions[child] for child in self.env.child_ids[i]))
                    forecast = [demand_est for _ in range(self.horizon)]
                forecasts.append(forecast)
        else:
            # serial environment: use demand_list from env
            seq = getattr(self.env, "demand_list", [])
            forecast_base = build_forecast(seq)
            for _ in range(self.env.agent_num):
                forecasts.append(forecast_base)
        # Refine each action using the chosen heuristic
        for i, act in enumerate(actions):
            if act <= 0:
                continue
            # Determine cost parameters
            h_cost = self.env.holding_cost[i] if hasattr(self.env, "holding_cost") else 1.0
            b_cost = self.env.backlog_cost[i] if hasattr(self.env, "backlog_cost") else 1.0
            f_cost = self.env.fixed_cost if hasattr(self.env, "fixed_cost") else 0.0
            max_order = self.env.action_dim - 1
            init_inv = inventory[i] + backlog[i]  # effective on-hand inventory
            forecast = forecasts[i]
            if self.use_ga:
                qty = optimise_order(
                    current_inventory=init_inv,
                    demand_forecast=forecast,
                    holding_cost=h_cost,
                    backlog_cost=b_cost,
                    fixed_cost=f_cost,
                    max_order=max_order,
                    horizon=self.horizon,
                    pop_size=self.ga_params.get("pop_size", 20),
                    generations=self.ga_params.get("generations", 30),
                    mutation_rate=self.ga_params.get("mutation_rate", 0.1),
                )
            elif self.use_dalsa:
                qty = dalsa_optimize(
                    current_inventory=init_inv,
                    demand_forecast=forecast,
                    holding_cost=h_cost,
                    backlog_cost=b_cost,
                    fixed_cost=f_cost,
                    max_order=max_order,
                    horizon=self.horizon,
                )
            else:
                qty = act
            # Clamp to action space
            refined[i] = max(0, min(qty, max_order))
        return refined
