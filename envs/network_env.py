"""
network_env.py
================

This module implements a generic network supply chain environment for
reinforcement learning.  It extends the serial supply chain model by
allowing arbitrary directed tree structures, where each node (agent)
represents a company in the network and may have multiple downstream
customers and a single upstream supplier.  The environment tracks
inventory, backlog, pipeline orders, and exogenous customer demand at the
leaf (retailer) nodes.

The model assumes a fixed shipping lead time across all edges.  At each
time step, all agents simultaneously decide their order quantity from
their immediate supplier.  Downstream demand is computed as the sum of
orders placed by children agents (for internal nodes) or as exogenous
customer demand for leaf nodes.  Inventory and backlog are updated
accordingly, and rewards are computed as the negative of holding,
backlog and fixed ordering costs.

This environment is intended for demonstration purposes and does not
attempt to replicate all complexities of a real supply chain network.
It is sufficient for exploring MARL algorithms on small networks.

Example usage::

    from inventory_management_RL_Lot.envs.network_env import NetworkInventoryEnv
    # Define a 2x3 network: two retailers (0,1), two distributors (2,3), one factory (4)
    children = {4: [2,3], 2: [0], 3: [1], 0: [], 1: []}
    parents = {0: 2, 1: 3, 2: 4, 3: 4, 4: None}
    env = NetworkInventoryEnv(
        children=children,
        parents=parents,
        lead_time=2,
        episode_len=100,
        action_dim=21,
        init_inventory=10,
        init_outstanding=10,
        holding_cost=[1.0]*5,
        backlog_cost=[1.0]*5,
        fixed_cost=0.0,
        external_demand_dist="uniform",
    )
    obs = env.reset(train=True)
    done = [False]*env.agent_num
    while not all(done):
        actions = [env.action_dim//2]*env.agent_num
        obs, rewards, done, info = env.step(actions, one_hot=False)

The implementation here is intentionally lightweight and does not
validate the supplied adjacency lists.  It assumes the supplied
``children`` and ``parents`` form a directed tree with a single root
(supplier) that has no parent and that each node has exactly one
parent (except the root).  Cycles in the graph will produce
undefined behaviour.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base_env import BaseInventoryEnv
from ..data_loader import generate_training_demand, load_eval_data


class NetworkInventoryEnv(BaseInventoryEnv):
    """Supply chain network environment for multi–agent RL.

    Parameters
    ----------
    children : Dict[int, List[int]]
        Mapping from each agent index to a list of downstream child indices.
        Nodes with no children are treated as retailers facing exogenous
        customer demand.
    parents : Dict[int, Optional[int]]
        Mapping from each agent index to its parent index.  The root
        node(s) should have value ``None``.
    lead_time : int
        Shipping lead time between connected nodes.  Applies uniformly.
    episode_len : int
        Number of time steps per episode.
    action_dim : int
        Size of the discrete action space for each agent.  Actions are
        interpreted as order quantities from 0 to ``action_dim-1``.
    init_inventory : int
        Initial inventory level for all agents at episode start.
    init_outstanding : int
        Initial outstanding orders in the pipeline for each agent and
        period.  Creates a pipeline with ``lead_time`` identical
        outstanding orders.
    holding_cost : List[float]
        Per-unit holding cost for each agent.
    backlog_cost : List[float]
        Per-unit backorder cost for each agent.
    fixed_cost : float
        Fixed cost incurred whenever an agent places a non-zero order.
    external_demand_dist : str
        Distribution used to generate random customer demand for
        retailers during training.  Supported values: ``"uniform"``,
        ``"poisson"``, ``"merton"``.
    external_max_demand : int
        Upper bound for the random demand distribution.  Defaults to
        ``action_dim - 1``.
    eval_data_dirs : Optional[List[str]]
        Optional list of directories containing evaluation demand
        sequences for each retailer.  If provided, the environment will
        cycle through these sequences in evaluation mode.  The list
        length must equal the number of retailers.  If ``None``,
        evaluation will use randomly generated demand.
    rng_seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        children: Dict[int, List[int]],
        parents: Dict[int, Optional[int]],
        lead_time: int = 2,
        episode_len: int = 200,
        action_dim: int = 21,
        init_inventory: int = 10,
        init_outstanding: int = 10,
        holding_cost: Optional[List[float]] = None,
        backlog_cost: Optional[List[float]] = None,
        fixed_cost: float = 0.0,
        external_demand_dist: str = "uniform",
        external_max_demand: Optional[int] = None,
        eval_data_dirs: Optional[List[str]] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        # Topology
        self.children: Dict[int, List[int]] = {k: list(v) for k, v in children.items()}
        self.parents: Dict[int, Optional[int]] = dict(parents)
        # Determine agent order from keys (deterministic ordering)
        nodes = sorted(self.children.keys())
        self.id_to_index = {node_id: idx for idx, node_id in enumerate(nodes)}
        self.index_to_id = {idx: node_id for node_id, idx in self.id_to_index.items()}
        self.agent_num = len(nodes)
        # Precompute inverse mapping for children/parents in index space
        self.child_ids: List[List[int]] = [[] for _ in range(self.agent_num)]
        self.parent_ids: List[Optional[int]] = [None for _ in range(self.agent_num)]
        for node_id, kids in self.children.items():
            idx = self.id_to_index[node_id]
            self.child_ids[idx] = [self.id_to_index[c] for c in kids]
        for node_id, parent in self.parents.items():
            idx = self.id_to_index[node_id]
            self.parent_ids[idx] = self.id_to_index[parent] if parent is not None else None
        self.leaf_indices = [idx for idx, kids in enumerate(self.child_ids) if not kids]
        # Environment parameters
        self.lead_time = lead_time
        self.episode_len = episode_len
        self.action_dim = action_dim
        self.init_inventory = init_inventory
        self.init_outstanding = init_outstanding
        self.holding_cost = holding_cost if holding_cost is not None else [1.0] * self.agent_num
        self.backlog_cost = backlog_cost if backlog_cost is not None else [1.0] * self.agent_num
        self.fixed_cost = fixed_cost
        self.external_demand_dist = external_demand_dist
        self.external_max_demand = external_max_demand if external_max_demand is not None else (action_dim - 1)
        self.eval_data_dirs = eval_data_dirs
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        # Observations: inventory, backlog, aggregated demand/downstream order, and pipeline
        # For each agent, obs_dim = 2 + 1 + lead_time = lead_time + 3
        self.obs_dim = self.lead_time + 3
        # Evaluation data
        if self.eval_data_dirs:
            if len(self.eval_data_dirs) != len(self.leaf_indices):
                raise ValueError(
                    "eval_data_dirs length must equal number of retailers"
                )
            self.eval_data = []
            self.n_eval = None
            # Load sequences for each retailer separately; evaluation will zip them
            self.eval_data_per_retailer = []
            self.n_eval = None
            for d in self.eval_data_dirs:
                n, data = load_eval_data(d)
                if self.n_eval is None:
                    self.n_eval = n
                elif self.n_eval != n:
                    raise ValueError(
                        "All eval_data_dirs must contain the same number of sequences"
                    )
                self.eval_data_per_retailer.append(data)
            # Transpose list of lists so that each element is a list of demands per retailer
            # for one episode
            self.eval_data = []
            for i in range(self.n_eval):
                episode_demands = []
                for data in self.eval_data_per_retailer:
                    episode_demands.append(data[i])
                self.eval_data.append(episode_demands)
        else:
            self.eval_data = []
            self.n_eval = 0
        self.eval_index: int = 0
        # State variables
        self.inventory: List[int] = []
        self.backlog: List[int] = []
        self.backlog_history: List[List[int]] = []
        self.pipeline_orders: List[List[int]] = []
        
        # NEW: Track demand and fulfilled for service level calculation
        self.demand_history: List[List[int]] = []
        self.fulfilled_history: List[List[int]] = []
        
        # Demand list holds list per retailer if eval_data_dirs provided; else generated
        self.external_demand_list: List[List[int]] = []
        self.step_num: int = 0
        self.train: bool = True
        self.normalize: bool = True
        self.action_history: List[List[int]] = []
        self.record_act_sta: List[List[float]] = []
        self.eval_bw_res: List[float] = []
        # Smoothing parameter for rewards
        self.alpha: float = 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, train: bool = True, normalize: bool = True) -> List[np.ndarray]:
        """Reset environment for a new episode and return initial observations."""
        self.train = train
        self.normalize = normalize
        self.step_num = 0
        # Initialise state
        self.inventory = [self.init_inventory for _ in range(self.agent_num)]
        self.backlog_history = [[] for _ in range(self.agent_num)]
        self.backlog = [0 for _ in range(self.agent_num)]
        self.pipeline_orders = [
            [self.init_outstanding for _ in range(self.lead_time)]
            for _ in range(self.agent_num)
        ]
        self.action_history = [[] for _ in range(self.agent_num)]
        
        # NEW: Reset demand and fulfilled history
        self.demand_history = [[] for _ in range(self.agent_num)]
        self.fulfilled_history = [[] for _ in range(self.agent_num)]
        
        # Reset bullwhip stats
        self.record_act_sta = [[] for _ in range(self.agent_num)]
        self.eval_bw_res = []
        # Load or generate external demand for retailers (leaf nodes)
        if not train and self.eval_data:
            # Each evaluation episode uses preloaded sequences
            self.external_demand_list = self.eval_data[self.eval_index]
            self.eval_index = (self.eval_index + 1) % max(1, self.n_eval)
        else:
            # Generate random demand for each retailer
            # Create list of demands per retailer (list of lists)
            self.external_demand_list = []
            for _ in self.leaf_indices:
                seq = generate_training_demand(
                    self.episode_len,
                    self.external_max_demand,
                    distribution=self.external_demand_dist,
                    seed=self.rng_seed,
                )
                self.external_demand_list.append(seq)
        # Construct initial observation
        return self._get_reset_obs()

    def step(self, actions: List[int], one_hot: bool = True) -> Tuple[List[np.ndarray], List[List[float]], List[bool], List[Any]]:
        """Run one time step of the network supply chain.

        Each agent supplies an action representing the order quantity from
        its parent.  The environment then computes downstream demand,
        updates inventories and backlogs, advances the pipeline, and
        computes per–agent rewards.

        Returns the next observations, rewards, done flags, and info
        dictionaries.
        """
        # Convert actions to indices if one_hot
        if one_hot:
            act_idxs = [int(np.argmax(a)) for a in actions]
        else:
            act_idxs = [int(a) for a in actions]
        # Bound actions
        act_idxs = [min(max(ai, 0), self.action_dim - 1) for ai in act_idxs]
        # Map actions directly to order quantities
        order_quantities = act_idxs
        # Update state and compute raw rewards
        rewards = self._state_update(order_quantities)
        # Generate next observation
        next_obs = self._get_step_obs(order_quantities)
        # Smooth rewards if training
        processed_rewards = self._get_processed_rewards(rewards)
        done_flag = self.step_num >= self.episode_len
        done = [done_flag for _ in range(self.agent_num)]
        info = [{} for _ in range(self.agent_num)]
        return next_obs, processed_rewards, done, info

    def get_eval_num(self) -> int:
        return self.n_eval

    def get_eval_bw_res(self) -> List[float]:
        return self.eval_bw_res

    def get_orders(self) -> List[int]:
        return getattr(self, "current_orders", [])

    def get_inventory(self) -> List[int]:
        return self.inventory

    # NEW: Getter methods for service level calculation
    def get_demand_history(self) -> List[List[int]]:
        """Return demand history for each agent."""
        return self.demand_history

    def get_fulfilled_history(self) -> List[List[int]]:
        """Return fulfilled demand history for each agent."""
        return self.fulfilled_history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_reset_obs(self) -> List[np.ndarray]:
        """Construct initial observations for each agent."""
        obs_list: List[np.ndarray] = []
        # External demand: initial demand for retailers at step 0
        initial_demands = [
            self.external_demand_list[j][0] if j < len(self.external_demand_list) else 0
            for j in range(len(self.leaf_indices))
        ]
        # Map leaf index to demand
        leaf_demand_map = {
            leaf: initial_demands[idx] for idx, leaf in enumerate(self.leaf_indices)
        }
        # For each agent, build obs: inventory, backlog, downstream demand (or child orders) and pipeline
        for i in range(self.agent_num):
            inv = self.inventory[i]
            back = self.backlog[i]
            # Determine down_info: if leaf, use external demand; else use sum of child orders
            if i in self.leaf_indices:
                down_info = leaf_demand_map[i]
            else:
                # No previous actions yet; use initial outstanding as proxy
                down_info = self.init_outstanding
            pipe = self.pipeline_orders[i]
            arr = np.array([inv, back, down_info] + pipe, dtype=float)
            if self.normalize:
                arr = arr / (self.action_dim - 1)
            obs_list.append(arr.reshape(self.obs_dim))
        return obs_list

    def _get_step_obs(self, actions: List[int]) -> List[np.ndarray]:
        """Construct observations after a step for each agent."""
        obs_list: List[np.ndarray] = []
        # Determine current demands for leaves at this step
        if self.train or not self.eval_data:
            # Use generated external demand list
            current_demands = [
                self.external_demand_list[idx][self.step_num - 1]
                if self.step_num - 1 < len(self.external_demand_list[idx])
                else 0
                for idx in range(len(self.leaf_indices))
            ]
        else:
            current_demands = [
                self.external_demand_list[idx][self.step_num - 1]
                for idx in range(len(self.leaf_indices))
            ]
        leaf_demand_map = {
            leaf: current_demands[idx] for idx, leaf in enumerate(self.leaf_indices)
        }
        # Downstream orders for internal nodes: use actions of children
        # Precompute previous actions mapping: children_action[i] = sum actions of children
        children_action_sum = [0 for _ in range(self.agent_num)]
        for parent_idx in range(self.agent_num):
            for child_idx in self.child_ids[parent_idx]:
                children_action_sum[parent_idx] += actions[child_idx]
        for i in range(self.agent_num):
            inv = self.inventory[i]
            back = self.backlog[i]
            if i in self.leaf_indices:
                down_info = leaf_demand_map[i]
            else:
                down_info = children_action_sum[i]
            pipe = self.pipeline_orders[i]
            arr = np.array([inv, back, down_info] + pipe, dtype=float)
            if self.normalize:
                arr = arr / (self.action_dim - 1)
            obs_list.append(arr.reshape(self.obs_dim))
        return obs_list

    def _get_processed_rewards(self, rewards: List[float]) -> List[List[float]]:
        """Optionally smooth rewards across agents to encourage coordination."""
        if self.train:
            mean_r = float(np.mean(rewards))
            return [[self.alpha * r + (1.0 - self.alpha) * mean_r] for r in rewards]
        else:
            return [[r] for r in rewards]

    def _state_update(self, actions: List[int]) -> List[float]:
        """Update environment state given order actions and return raw rewards."""
        # Save actions history
        self.action_history = [h + [a] for h, a in zip(self.action_history, actions)]
        
        # Determine downstream demand for each agent: sum of child actions or external demand
        downstream_demands: List[int] = [0 for _ in range(self.agent_num)]
        # Leaf nodes: external demand
        for idx, leaf in enumerate(self.leaf_indices):
            downstream_demands[leaf] = self.external_demand_list[idx][self.step_num]
        # Internal nodes: sum of children actions
        for parent_idx in range(self.agent_num):
            if parent_idx not in self.leaf_indices:
                s = 0
                for child in self.child_ids[parent_idx]:
                    s += actions[child]
                downstream_demands[parent_idx] = s
        
        # NEW: Track demand faced by each agent
        for i, d in enumerate(downstream_demands):
            self.demand_history[i].append(d)
        
        # Effective demand per agent
        effective_demand = [downstream_demands[i] + self.backlog[i] for i in range(self.agent_num)]
        
        # Advance time
        self.step_num += 1
        
        # Compute rewards and update state
        rewards: List[float] = []
        for i in range(self.agent_num):
            # Received goods from parent
            received = int(self.pipeline_orders[i][0])
            
            # Available inventory to fulfill demand
            available = self.inventory[i] + received
            
            # NEW: Calculate fulfilled demand for this period
            fulfilled = min(effective_demand[i], available)
            self.fulfilled_history[i].append(fulfilled)
            
            unmet = effective_demand[i] - available
            if unmet > 0:
                self.backlog[i] = unmet
                self.inventory[i] = 0
            else:
                self.backlog[i] = 0
                self.inventory[i] = -unmet
            self.backlog_history[i].append(self.backlog[i])
            
            # Determine new order to parent; each agent orders based on their action
            new_order = actions[i]
            
            # Append new order into pipeline and remove oldest
            self.pipeline_orders[i].append(new_order)
            self.pipeline_orders[i].pop(0)
            
            # Compute costs
            order_cost_fix = self.fixed_cost if new_order > 0 else 0.0
            cost = (
                self.inventory[i] * self.holding_cost[i]
                + self.backlog[i] * self.backlog_cost[i]
                + order_cost_fix
            )
            rewards.append(-cost)
        
        # Update bullwhip metrics if evaluation episode ended
        if not self.train and self.step_num == self.episode_len:
            for k in range(self.agent_num):
                hist = self.action_history[k]
                if not hist or np.mean(hist) < 1e-6:
                    self.record_act_sta[k].append(0.0)
                else:
                    cv = float(np.std(hist) / np.mean(hist))
                    self.record_act_sta[k].append(cv)
            if self.eval_index == 0:
                self.eval_bw_res = [float(np.mean(sta)) for sta in self.record_act_sta]
                self.record_act_sta = [[] for _ in range(self.agent_num)]
        return rewards
