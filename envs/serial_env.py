"""
serial_env.py
==============

This module implements a simplified multi-echelon inventory management
environment for reinforcement learning.  The environment models a serial
supply chain with a fixed number of agents (or echelons) where each agent
decides how much to order from its upstream neighbour to satisfy the
downstream demand.  The environment supports backorders, pipeline
inventories (lead time), holding costs, backlog costs, and optional fixed
ordering costs.  It also provides facilities for loading evaluation
sequences and generating random training demand via the ``data_loader``
module.

The design of this environment is inspired by the Beer Game simulation
and follows the interface expectations of multi-agent RL algorithms such
as MADDPG or HAPPO.  Each agent observes a subset of the system state
relevant to its ordering decision: its own inventory and backlog, the
most recent order from the agent immediately downstream (for upstream
agents), and the outstanding pipeline orders.

Example usage::

    from inventory_management_RL_Lot.envs.serial_env import SerialInventoryEnv
    env = SerialInventoryEnv(level_num=3, lead_time=2)
    obs = env.reset(train=True)
    done = [False] * env.agent_num
    while not all(done):
        actions = [env.action_dim // 2 for _ in range(env.agent_num)]  # simple policy
        obs, rewards, done, info = env.step(actions, one_hot=False)

"""

from __future__ import annotations

import numpy as np
import random
from typing import List, Tuple, Optional, Any
from pathlib import Path

from ..data_loader import load_eval_data, generate_training_demand


class SerialInventoryEnv:
    """Multi-echelon serial supply chain environment.

    Parameters
    ----------
    level_num : int
        Number of agents/echelons in the supply chain.
    lead_time : int
        Shipping lead time between consecutive levels.  All levels share
        the same lead time for simplicity.
    episode_len : int
        Length of each episode (number of time steps).
    action_dim : int
        Size of the discrete action space.  Actions are interpreted as
        order quantities from 0 to ``action_dim-1``.
    init_inventory : int
        Initial inventory level for each agent at the beginning of an
        episode.
    init_outstanding : int
        Initial outstanding (pipeline) orders for each agent and period
        in the pipeline.  A list of length ``lead_time`` is created
        where each entry is ``init_outstanding``.  This approximates an
        initially full supply line.
    holding_cost : List[float]
        Per-unit holding cost for each agent.  Should be length
        ``level_num``.
    backlog_cost : List[float]
        Per-unit backorder cost for each agent.  Should be length
        ``level_num``.
    fixed_cost : float
        Fixed ordering cost incurred whenever an agent places a
        non-zero order.  Set to zero to disable fixed costs.
    price_discount : bool
        If true, apply a price discount schedule depending on the order
        quantity.  The discount schedule is provided via
        ``discount_schedule``.
    discount_schedule : List[float]
        Discount multipliers for buckets of order quantities.  The
        default schedule assumes actions are divided into equal-sized
        buckets of 5 units.  For example, with an action space of size
        21, index 0 means ordering 0 units; indices 1-5 correspond to
        bucket 1, 6-10 bucket 2, 11-15 bucket 3, 16-20 bucket 4.  The
        multiplier at ``discount_schedule[k]`` is multiplied by the
        variable cost (which is currently zero by default) for bucket
        ``k``.
    eval_data_dir : Optional[str]
        Path to a directory containing evaluation demand sequences.
        When ``train=False`` in ``reset()``, the environment will cycle
        through the files in this directory to drive demand.  If
        ``None``, evaluation mode is disabled.
    rng_seed : Optional[int]
        Seed for random number generation.  Provides deterministic
        training demand when set.
    """

    def __init__(
        self,
        level_num: int = 3,
        lead_time: int = 2,
        episode_len: int = 200,
        action_dim: int = 21,
        init_inventory: int = 10,
        init_outstanding: int = 10,
        holding_cost: Optional[List[float]] = None,
        backlog_cost: Optional[List[float]] = None,
        fixed_cost: float = 0.0,
        price_discount: bool = False,
        discount_schedule: Optional[List[float]] = None,
        eval_data_dir: Optional[str] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.level_num = level_num
        self.lead_time = lead_time
        self.episode_len = episode_len
        self.action_dim = action_dim
        self.init_inventory = init_inventory
        self.init_outstanding = init_outstanding
        self.holding_cost = holding_cost if holding_cost is not None else [1.0] * level_num
        self.backlog_cost = backlog_cost if backlog_cost is not None else [1.0] * level_num
        self.fixed_cost = fixed_cost
        self.price_discount = price_discount
        if discount_schedule is None:
            # Default discount schedule: no discount for any bucket.
            self.discount_schedule = [1.0 for _ in range((action_dim + 4) // 5)]
        else:
            self.discount_schedule = discount_schedule
        self.eval_data_dir = eval_data_dir
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # Derived attributes
        self.agent_num = self.level_num
        self.obs_dim = self.lead_time + 3

        # Evaluation data
        if self.eval_data_dir:
            self.n_eval, self.eval_data = load_eval_data(self.eval_data_dir)
        else:
            self.n_eval, self.eval_data = 0, []
        self.eval_index: int = 0

        # Environment state (initialised in reset)
        self.inventory: List[int] = []
        self.backlog: List[int] = []
        self.backlog_history: List[List[int]] = []
        self.pipeline_orders: List[List[int]] = []
        self.demand_list: List[int] = []
        self.step_num: int = 0
        self.train: bool = True
        self.normalize: bool = True
        self.action_history: List[List[int]] = []
        self.record_act_sta: List[List[float]] = [[] for _ in range(self.level_num)]
        self.eval_bw_res: List[float] = []
        # For reward smoothing
        self.alpha: float = 0.5

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def reset(self, train: bool = True, normalize: bool = True) -> List[np.ndarray]:
        """Reset the environment for a new episode.

        Args:
            train: If True, generate new random demand for training.  If
                False, cycle through the evaluation data.
            normalize: Whether to normalise observation values to the
                range [0, 1].  Normalisation divides by ``action_dim-1``.

        Returns:
            A list of initial observations for each agent.
        """
        self.train = train
        self.normalize = normalize
        self.step_num = 0
        self.inventory = [self.init_inventory for _ in range(self.level_num)]
        self.backlog = [0 for _ in range(self.level_num)]
        self.backlog_history = [[] for _ in range(self.level_num)]
        self.pipeline_orders = [
            [self.init_outstanding for _ in range(self.lead_time)]
            for _ in range(self.level_num)
        ]
        self.action_history = [[] for _ in range(self.level_num)]
        # Load or generate demand sequence
        if not train:
            if not self.eval_data:
                raise RuntimeError(
                    "Evaluation data directory not provided or empty."
                )
            self.demand_list = self.eval_data[self.eval_index]
            self.eval_index = (self.eval_index + 1) % max(1, self.n_eval)
        else:
            # Generate random demand using the maximum possible action as the upper bound
            self.demand_list = generate_training_demand(
                self.episode_len, self.action_dim - 1, distribution="uniform", seed=self.rng_seed
            )
        # Reset metrics for bullwhip measurement in evaluation
        self.record_act_sta = [[] for _ in range(self.level_num)]
        self.eval_bw_res = []
        # Initial observation
        return self._get_reset_obs()

    def step(self, actions: List[int], one_hot: bool = True) -> Tuple[List[np.ndarray], List[List[float]], List[bool], List[Any]]:
        """Advance the environment by one time step.

        Args:
            actions: List of actions (order quantities) chosen by each agent.
                If ``one_hot`` is True, actions should be one-hot encoded and
                will be converted via ``argmax``.  Otherwise actions are
                assumed to be integer order quantities in the range
                [0, action_dim-1].
            one_hot: Indicates whether ``actions`` are one-hot vectors.

        Returns:
            A tuple ``(obs, rewards, done, info)`` similar to OpenAI Gym.
            ``obs``: list of observations per agent.
            ``rewards``: list of rewards per agent, each wrapped in a list
                for compatibility with certain MARL libraries.
            ``done``: list of booleans indicating episode termination for
                each agent (all elements equal once episode ends).
            ``info``: a list of optional diagnostic info (unused).
        """
        # Convert one-hot actions to integer indices
        if one_hot:
            # Each element of actions is expected to be a list or np array
            act_idxs = [int(np.argmax(a)) for a in actions]
        else:
            act_idxs = [int(a) for a in actions]
        # Map indices to actual order quantities (identity mapping here)
        order_quantities = self._action_map(act_idxs)
        # Update state and compute raw rewards
        reward = self._state_update(order_quantities)
        # Get observations for next step
        next_obs = self._get_step_obs(order_quantities)
        # Process rewards (smooth across agents if training)
        processed_rewards = self._get_processed_rewards(reward)
        # Determine done flag
        done_flag = self.step_num >= self.episode_len
        done = [done_flag for _ in range(self.agent_num)]
        info = [{} for _ in range(self.agent_num)]
        return next_obs, processed_rewards, done, info

    def get_eval_num(self) -> int:
        """Return the number of evaluation demand sequences loaded."""
        return self.n_eval

    def get_eval_bw_res(self) -> List[float]:
        """Get the bullwhip metric after evaluation episodes.

        The bullwhip metric is computed at the end of each evaluation
        episode and stored in ``self.eval_bw_res``.  This method returns
        the current stored values and resets them for the next evaluation.
        """
        return self.eval_bw_res

    def get_orders(self) -> List[int]:
        """Return the current actual orders placed by each agent.

        During training or evaluation the environment stores the last
        processed actions (order quantities) so that they can be printed
        or logged externally.  If called before any steps have been
        executed this will return an empty list.
        """
        return getattr(self, "current_orders", [])

    def get_inventory(self) -> List[int]:
        """Return the current inventory level for each agent."""
        return self.inventory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _action_map(self, actions: List[int]) -> List[int]:
        """Map discrete action indices to order quantities.

        In the current implementation the action index corresponds
        directly to the order quantity (e.g. index 0 -> 0 units, index 1 -> 1
        unit, ..., index N -> N units).  If you wish to use a coarser
        discretisation or a different mapping (e.g. scaling by a factor),
        modify this method.

        The mapped actions are also saved in ``self.current_orders`` for
        retrieval via ``get_orders()``.
        """
        mapped_actions = [min(max(int(a), 0), self.action_dim - 1) for a in actions]
        self.current_orders = mapped_actions
        return mapped_actions

    def _get_reset_obs(self) -> List[np.ndarray]:
        """Construct initial observations for each agent after reset."""
        obs_list: List[np.ndarray] = []
        for i in range(self.level_num):
            inv = self.inventory[i]
            back = self.backlog[i]
            # For reset observation, use a dummy demand value equal to
            # init_outstanding to indicate the outstanding order pipeline state
            down_info = self.init_outstanding
            pipeline = self.pipeline_orders[i]
            arr = np.array([inv, back, down_info] + pipeline, dtype=float)
            if self.normalize:
                arr = arr / (self.action_dim - 1)
            obs_list.append(arr.reshape(self.obs_dim))
        return obs_list

    def _get_step_obs(self, actions: List[int]) -> List[np.ndarray]:
        """Construct observations for each agent after a time step."""
        obs_list: List[np.ndarray] = []
        # Downstream agent (0) observes customer demand directly
        downstream_dem = self.demand_list[self.step_num - 1]
        inv = self.inventory[0]
        back = self.backlog[0]
        pipe = self.pipeline_orders[0]
        arr = np.array([inv, back, downstream_dem] + pipe, dtype=float)
        if self.normalize:
            arr = arr / (self.action_dim - 1)
        obs_list.append(arr.reshape(self.obs_dim))
        # Upstream agents observe previous agent's order instead of demand
        for i in range(1, self.level_num):
            inv_i = self.inventory[i]
            back_i = self.backlog[i]
            down_action = actions[i - 1]
            pipe_i = self.pipeline_orders[i]
            arr = np.array([inv_i, back_i, down_action] + pipe_i, dtype=float)
            if self.normalize:
                arr = arr / (self.action_dim - 1)
            obs_list.append(arr.reshape(self.obs_dim))
        return obs_list

    def _get_processed_rewards(self, rewards: List[float]) -> List[List[float]]:
        """Apply smoothing to raw rewards across agents if training.

        The smoothing uses a convex combination of each agent's own
        reward and the mean reward across all agents.  This encourages
        coordination by penalising large discrepancies in cost among
        agents.  During evaluation (``train=False``) the raw rewards are
        returned unchanged.

        Returns:
            A list of rewards where each element is wrapped in a list
            (shape ``[agent_num][1]``) for compatibility with some MARL
            frameworks.
        """
        if self.train:
            mean_r = float(np.mean(rewards))
            processed = [[self.alpha * r + (1.0 - self.alpha) * mean_r] for r in rewards]
            return processed
        else:
            return [[r] for r in rewards]

    def _state_update(self, actions: List[int]) -> List[float]:
        """Update system state based on actions and return raw rewards.

        The environment processes orders and demand as follows:

        1. Determine the effective demand faced by each agent.  The
           downstream agent (level 0) faces exogenous customer demand.
           Each upstream agent faces the order placed by the agent
           immediately downstream.
        2. Apply shipping losses (currently disabled by default).  If
           ``random_shipping_loss`` is enabled, a fraction of the
           goods shipped may be lost.
        3. Compute unmet demand (positive backlog or negative inventory)
           and update inventory and backlog for each agent accordingly.
        4. Generate new pipeline orders for each agent: downstream
           agents place orders equal to their action; upstream agents
           place orders equal to the incoming demand from downstream
           agents (bounded by available inventory and pipeline receipts).
        5. Compute reward (negative cost) as the sum of holding cost,
           backlog cost, variable order cost (currently zero unless
           ``price_discount`` is enabled), and fixed ordering cost if
           ``fixed_cost > 0``.
        6. Update bullwhip statistics during evaluation.

        Args:
            actions: A list of order quantities for each agent.

        Returns:
            A list of raw rewards (negative costs) for each agent.
        """
        # Save current actions for order logging
        self.action_history = [h + [a] for h, a in zip(self.action_history, actions)]
        # Compute effective demand for each agent: downstream demand + backlog
        # Downstream agent faces external customer demand
        downstream_demands: List[int] = [self.demand_list[self.step_num]] + actions[:-1]
        effective_demand = [d + self.backlog[i] for i, d in enumerate(downstream_demands)]
        # Shipping loss is disabled by default; if enabled, some shipped
        # items are lost randomly.  For determinism, the loss rate can
        # depend on a global constant or be drawn per agent per period.
        random_shipping_loss = False
        lost_rate = [1.0 for _ in range(self.level_num)]
        if random_shipping_loss:
            # Example: 10% maximum loss
            lost_rate = [1.0 - random.random() * 0.1 for _ in range(self.level_num)]
        # Advance simulation clock
        self.step_num += 1
        rewards: List[float] = []
        for i in range(self.level_num):
            # Unmet demand (positive means backlog, negative means inventory remains)
            # Goods received this period: pipeline_orders[i][0], adjusted for loss
            received = int(self.pipeline_orders[i][0] * lost_rate[i])
            unmet = effective_demand[i] - (self.inventory[i] + received)
            # Update inventory and backlog
            if unmet > 0:
                self.backlog[i] = unmet
                self.inventory[i] = 0
            else:
                self.backlog[i] = 0
                self.inventory[i] = -unmet
            self.backlog_history[i].append(self.backlog[i])
            # Append new order into pipeline
            if i == self.level_num - 1:
                # Last agent orders according to its action
                new_order = actions[i]
            else:
                # Upstream agents order enough to meet downstream demand
                # Bound by available inventory + incoming pipeline
                downstream_demand = effective_demand[i + 1]
                upstream_available = self.inventory[i + 1] + int(self.pipeline_orders[i + 1][0] * lost_rate[i + 1])
                new_order = int(min(downstream_demand, upstream_available))
            # Append to pipeline and pop the oldest item
            self.pipeline_orders[i].append(new_order)
            self.pipeline_orders[i].pop(0)
            # Compute ordering cost; price discount not currently active
            order_cost_var = 0.0
            if self.price_discount and actions[i] > 0:
                # Determine bucket index based on action (0..action_dim-1)
                bucket_idx = min(actions[i] // 5, len(self.discount_schedule) - 1)
                order_cost_var = self.discount_schedule[bucket_idx] * actions[i]
            # Fixed ordering cost if order quantity > 0
            order_cost_fix = self.fixed_cost if actions[i] > 0 else 0.0
            # Compute reward (negative of cost)
            cost = (
                self.inventory[i] * self.holding_cost[i]
                + self.backlog[i] * self.backlog_cost[i]
                + order_cost_var
                + order_cost_fix
            )
            rewards.append(-cost)
        # Update bullwhip metrics during evaluation
        if not self.train:
            if self.step_num == self.episode_len:
                # At the end of the evaluation episode compute the
                # coefficient of variation of order history for each
                # agent and store it in record_act_sta.
                for k in range(self.level_num):
                    hist = self.action_history[k]
                    if not hist or np.mean(hist) < 1e-6:
                        self.record_act_sta[k].append(0.0)
                    else:
                        cv = float(np.std(hist) / np.mean(hist))
                        self.record_act_sta[k].append(cv)
                # If this was the last evaluation sequence, compute
                # average bullwhip metric
                if self.eval_index == 0:
                    self.eval_bw_res = [float(np.mean(sta)) for sta in self.record_act_sta]
                    # Reset record_act_sta for next evaluation round
                    self.record_act_sta = [[] for _ in range(self.level_num)]
        return rewards
