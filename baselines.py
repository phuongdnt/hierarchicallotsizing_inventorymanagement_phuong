"""
baselines.py
============

Classical inventory control baselines for fair comparison with RL agents.

Implements:
1. Base Stock (Order-Up-To) Policy
2. (s, S) Policy (Reorder Point, Order-Up-To-Level)
3. (R, Q) Policy (Reorder Point, Fixed Quantity)
4. Moving Average Forecast Policy
5. Optimal Newsvendor (for single-period benchmark)

CRITICAL: All baselines use the EXACT SAME cost function as RL agents
to ensure apple-to-apple comparison.

Author: Cishi (Thesis Project)
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CostParameters:
    """Cost parameters matching the RL environment."""
    holding_cost: List[float]  # Per-agent holding cost
    backlog_cost: List[float]  # Per-agent backlog cost
    fixed_cost: float          # Fixed ordering cost (same for all)
    
    def compute_period_cost(
        self,
        inventory: List[int],
        backlog: List[int],
        orders: List[int]
    ) -> Tuple[float, List[float]]:
        """
        Compute cost using EXACT same formula as RL environment.
        
        Returns:
            total_cost: Sum across all agents
            costs_per_agent: List of costs for each agent
        """
        costs = []
        for i in range(len(inventory)):
            holding = max(0, inventory[i]) * self.holding_cost[i]
            backlog_cost = max(0, backlog[i]) * self.backlog_cost[i]
            order_cost = self.fixed_cost if orders[i] > 0 else 0.0
            costs.append(holding + backlog_cost + order_cost)
        return sum(costs), costs


class BaselinePolicy(ABC):
    """Abstract base class for inventory control policies."""
    
    def __init__(
        self,
        num_agents: int,
        lead_time: int,
        cost_params: CostParameters,
        action_dim: int = 41
    ):
        self.num_agents = num_agents
        self.lead_time = lead_time
        self.cost_params = cost_params
        self.action_dim = action_dim
        self.name = "BaselinePolicy"
    
    @abstractmethod
    def get_orders(
        self,
        inventory: List[int],
        backlog: List[int],
        pipeline: List[List[int]],
        demand_history: List[List[int]],
        current_demand: Optional[List[int]] = None
    ) -> List[int]:
        """
        Compute order quantities for all agents.
        
        Args:
            inventory: Current inventory levels
            backlog: Current backlog levels
            pipeline: Outstanding orders in pipeline [agent][time_to_arrival]
            demand_history: Historical demand [agent][time_step]
            current_demand: Current period demand (if known)
        
        Returns:
            List of order quantities, one per agent
        """
        raise NotImplementedError
    
    def clip_orders(self, orders: List[int]) -> List[int]:
        """Clip orders to valid action space."""
        return [max(0, min(o, self.action_dim - 1)) for o in orders]
    
    def compute_inventory_position(
        self,
        inventory: List[int],
        backlog: List[int],
        pipeline: List[List[int]]
    ) -> List[int]:
        """
        Compute inventory position = On-hand + Pipeline - Backlog
        
        This is the key metric for inventory control decisions.
        """
        positions = []
        for i in range(self.num_agents):
            on_hand = inventory[i]
            in_pipeline = sum(pipeline[i]) if pipeline[i] else 0
            back = backlog[i]
            position = on_hand + in_pipeline - back
            positions.append(position)
        return positions


class BaseStockPolicy(BaselinePolicy):
    """
    Base Stock (Order-Up-To) Policy.
    
    Order quantity = max(0, S - inventory_position)
    
    where S = base stock level, typically set to cover demand during
    lead time plus review period with safety stock.
    
    S = μ_L + z * σ_L
    
    where:
        μ_L = mean demand during lead time
        σ_L = std dev of demand during lead time
        z = safety factor (from service level target)
    """
    
    def __init__(
        self,
        num_agents: int,
        lead_time: int,
        cost_params: CostParameters,
        action_dim: int = 41,
        base_stock_levels: Optional[List[int]] = None,
        target_service_level: float = 0.95,
        demand_mean: Optional[float] = None,
        demand_std: Optional[float] = None
    ):
        super().__init__(num_agents, lead_time, cost_params, action_dim)
        self.name = "BaseStock"
        self.target_service_level = target_service_level
        
        if base_stock_levels is not None:
            self.base_stock_levels = base_stock_levels
        else:
            # Calculate optimal base stock levels
            self.base_stock_levels = self._calculate_base_stock_levels(
                demand_mean, demand_std
            )
    
    def _calculate_base_stock_levels(
        self,
        demand_mean: Optional[float],
        demand_std: Optional[float]
    ) -> List[int]:
        """Calculate base stock levels from demand statistics."""
        # Default demand parameters if not provided
        mean_d = demand_mean if demand_mean is not None else 10.0
        std_d = demand_std if demand_std is not None else 3.0
        
        # Safety factor from service level (normal approximation)
        from scipy import stats
        z = stats.norm.ppf(self.target_service_level)
        
        # Lead time demand statistics
        # Assuming iid demand: μ_L = L * μ, σ_L = sqrt(L) * σ
        mean_L = (self.lead_time + 1) * mean_d  # +1 for review period
        std_L = np.sqrt(self.lead_time + 1) * std_d
        
        # Base stock level
        S = int(np.ceil(mean_L + z * std_L))
        
        return [S] * self.num_agents
    
    def get_orders(
        self,
        inventory: List[int],
        backlog: List[int],
        pipeline: List[List[int]],
        demand_history: List[List[int]],
        current_demand: Optional[List[int]] = None
    ) -> List[int]:
        """Order up to base stock level."""
        positions = self.compute_inventory_position(inventory, backlog, pipeline)
        
        orders = []
        for i in range(self.num_agents):
            # Order up to S
            order = max(0, self.base_stock_levels[i] - positions[i])
            orders.append(order)
        
        return self.clip_orders(orders)
    
    def update_base_stock_from_history(self, demand_history: List[List[int]]):
        """Dynamically update base stock levels from observed demand."""
        if not demand_history or not demand_history[0]:
            return
        
        for i in range(self.num_agents):
            if demand_history[i]:
                mean_d = np.mean(demand_history[i])
                std_d = np.std(demand_history[i]) if len(demand_history[i]) > 1 else mean_d * 0.3
                
                from scipy import stats
                z = stats.norm.ppf(self.target_service_level)
                
                mean_L = (self.lead_time + 1) * mean_d
                std_L = np.sqrt(self.lead_time + 1) * std_d
                
                self.base_stock_levels[i] = int(np.ceil(mean_L + z * std_L))


class sS_Policy(BaselinePolicy):
    """
    (s, S) Policy - Reorder Point, Order-Up-To-Level.
    
    If inventory_position <= s: Order up to S
    Otherwise: Order nothing
    
    Parameters:
        s = reorder point (triggers ordering)
        S = order-up-to level
    
    Optimal (s, S) can be computed via dynamic programming, but we use
    approximations based on demand statistics.
    """
    
    def __init__(
        self,
        num_agents: int,
        lead_time: int,
        cost_params: CostParameters,
        action_dim: int = 41,
        reorder_points: Optional[List[int]] = None,
        order_up_to_levels: Optional[List[int]] = None,
        target_service_level: float = 0.95,
        demand_mean: Optional[float] = None,
        demand_std: Optional[float] = None
    ):
        super().__init__(num_agents, lead_time, cost_params, action_dim)
        self.name = "(s,S)"
        self.target_service_level = target_service_level
        
        if reorder_points is not None and order_up_to_levels is not None:
            self.s_levels = reorder_points
            self.S_levels = order_up_to_levels
        else:
            self.s_levels, self.S_levels = self._calculate_sS_levels(
                demand_mean, demand_std
            )
    
    def _calculate_sS_levels(
        self,
        demand_mean: Optional[float],
        demand_std: Optional[float]
    ) -> Tuple[List[int], List[int]]:
        """
        Calculate (s, S) levels using approximation.
        
        s ≈ Lead time demand + safety stock
        S ≈ s + EOQ (Economic Order Quantity)
        """
        mean_d = demand_mean if demand_mean is not None else 10.0
        std_d = demand_std if demand_std is not None else 3.0
        
        from scipy import stats
        z = stats.norm.ppf(self.target_service_level)
        
        s_levels = []
        S_levels = []
        
        for i in range(self.num_agents):
            # Reorder point: cover lead time demand + safety stock
            mean_L = self.lead_time * mean_d
            std_L = np.sqrt(self.lead_time) * std_d
            s = int(np.ceil(mean_L + z * std_L))
            
            # Order-up-to level: s + EOQ approximation
            # EOQ = sqrt(2 * D * K / h) where D=annual demand, K=fixed cost, h=holding
            # Simplified: use 2-3 periods of demand as order quantity
            h = self.cost_params.holding_cost[i]
            K = self.cost_params.fixed_cost
            
            if h > 0 and K > 0:
                # EOQ formula (simplified for period demand)
                D = mean_d * 100  # Approximate "annual" demand
                EOQ = int(np.sqrt(2 * D * K / h))
                EOQ = max(1, min(EOQ, self.action_dim - 1))
            else:
                EOQ = int(2 * mean_d)  # Default to ~2 periods
            
            S = s + EOQ
            
            s_levels.append(s)
            S_levels.append(S)
        
        return s_levels, S_levels
    
    def get_orders(
        self,
        inventory: List[int],
        backlog: List[int],
        pipeline: List[List[int]],
        demand_history: List[List[int]],
        current_demand: Optional[List[int]] = None
    ) -> List[int]:
        """Order up to S only if position <= s."""
        positions = self.compute_inventory_position(inventory, backlog, pipeline)
        
        orders = []
        for i in range(self.num_agents):
            if positions[i] <= self.s_levels[i]:
                # Trigger order: order up to S
                order = max(0, self.S_levels[i] - positions[i])
            else:
                # No order needed
                order = 0
            orders.append(order)
        
        return self.clip_orders(orders)


class RQ_Policy(BaselinePolicy):
    """
    (R, Q) Policy - Reorder Point, Fixed Quantity.
    
    If inventory_position <= R: Order exactly Q units
    Otherwise: Order nothing
    
    Simpler than (s, S) as order quantity is fixed.
    """
    
    def __init__(
        self,
        num_agents: int,
        lead_time: int,
        cost_params: CostParameters,
        action_dim: int = 41,
        reorder_points: Optional[List[int]] = None,
        order_quantities: Optional[List[int]] = None,
        target_service_level: float = 0.95,
        demand_mean: Optional[float] = None,
        demand_std: Optional[float] = None
    ):
        super().__init__(num_agents, lead_time, cost_params, action_dim)
        self.name = "(R,Q)"
        
        if reorder_points is not None and order_quantities is not None:
            self.R_levels = reorder_points
            self.Q_levels = order_quantities
        else:
            self.R_levels, self.Q_levels = self._calculate_RQ_levels(
                demand_mean, demand_std, target_service_level
            )
    
    def _calculate_RQ_levels(
        self,
        demand_mean: Optional[float],
        demand_std: Optional[float],
        target_service_level: float
    ) -> Tuple[List[int], List[int]]:
        """Calculate (R, Q) levels."""
        mean_d = demand_mean if demand_mean is not None else 10.0
        std_d = demand_std if demand_std is not None else 3.0
        
        from scipy import stats
        z = stats.norm.ppf(target_service_level)
        
        R_levels = []
        Q_levels = []
        
        for i in range(self.num_agents):
            # Reorder point
            mean_L = self.lead_time * mean_d
            std_L = np.sqrt(self.lead_time) * std_d
            R = int(np.ceil(mean_L + z * std_L))
            
            # Fixed order quantity (EOQ or similar)
            h = self.cost_params.holding_cost[i]
            K = self.cost_params.fixed_cost
            
            if h > 0 and K > 0:
                D = mean_d * 100
                Q = int(np.sqrt(2 * D * K / h))
                Q = max(1, min(Q, self.action_dim - 1))
            else:
                Q = int(3 * mean_d)
            
            R_levels.append(R)
            Q_levels.append(Q)
        
        return R_levels, Q_levels
    
    def get_orders(
        self,
        inventory: List[int],
        backlog: List[int],
        pipeline: List[List[int]],
        demand_history: List[List[int]],
        current_demand: Optional[List[int]] = None
    ) -> List[int]:
        """Order fixed Q if position <= R."""
        positions = self.compute_inventory_position(inventory, backlog, pipeline)
        
        orders = []
        for i in range(self.num_agents):
            if positions[i] <= self.R_levels[i]:
                order = self.Q_levels[i]
            else:
                order = 0
            orders.append(order)
        
        return self.clip_orders(orders)


class MovingAverageForecastPolicy(BaselinePolicy):
    """
    Moving Average Forecast Policy.
    
    Order quantity = Forecasted demand for lead time + safety stock - inventory position
    
    Uses simple moving average for demand forecasting.
    """
    
    def __init__(
        self,
        num_agents: int,
        lead_time: int,
        cost_params: CostParameters,
        action_dim: int = 41,
        window_size: int = 5,
        safety_factor: float = 1.5
    ):
        super().__init__(num_agents, lead_time, cost_params, action_dim)
        self.name = "MovingAvgForecast"
        self.window_size = window_size
        self.safety_factor = safety_factor
    
    def get_orders(
        self,
        inventory: List[int],
        backlog: List[int],
        pipeline: List[List[int]],
        demand_history: List[List[int]],
        current_demand: Optional[List[int]] = None
    ) -> List[int]:
        """Order based on moving average forecast."""
        positions = self.compute_inventory_position(inventory, backlog, pipeline)
        
        orders = []
        for i in range(self.num_agents):
            # Compute forecast
            if demand_history[i] and len(demand_history[i]) >= 2:
                recent = demand_history[i][-self.window_size:]
                mean_demand = np.mean(recent)
                std_demand = np.std(recent) if len(recent) > 1 else mean_demand * 0.3
            else:
                # Default if no history
                mean_demand = 10.0
                std_demand = 3.0
            
            # Forecast for lead time periods
            forecast = (self.lead_time + 1) * mean_demand
            safety_stock = self.safety_factor * np.sqrt(self.lead_time + 1) * std_demand
            
            # Target inventory position
            target = forecast + safety_stock
            
            # Order quantity
            order = max(0, int(np.ceil(target - positions[i])))
            orders.append(order)
        
        return self.clip_orders(orders)


# =============================================================================
# BASELINE RUNNER (Evaluation Interface)
# =============================================================================

class BaselineRunner:
    """
    Runs baseline policies on the same environment as RL agents.
    
    Ensures fair comparison by:
    1. Using exact same environment
    2. Using exact same demand sequences
    3. Using exact same cost function
    """
    
    def __init__(self, env, policy: BaselinePolicy):
        """
        Args:
            env: Inventory environment (SerialInventoryEnv or NetworkInventoryEnv)
            policy: Baseline policy to evaluate
        """
        self.env = env
        self.policy = policy
    
    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate baseline policy on evaluation episodes.
        
        Returns:
            Dictionary with same structure as RL evaluation for comparison
        """
        if num_episodes is None:
            num_episodes = self.env.get_eval_num() if hasattr(self.env, 'get_eval_num') else 1
        
        all_costs = []
        all_rewards = []
        all_service_levels = []
        all_cycle_sl = []
        all_bullwhip = []
        all_order_history = []
        
        # CRITICAL: Reset eval_index to ensure same sequences as RL
        if hasattr(self.env, 'eval_index'):
            self.env.eval_index = 0
        
        for ep in range(num_episodes):
            obs = self.env.reset(train=False)
            
            episode_costs = []
            episode_rewards = []
            order_history = [[] for _ in range(self.env.agent_num)]
            demand_history = [[] for _ in range(self.env.agent_num)]
            
            done = [False] * self.env.agent_num
            step = 0
            
            while not all(done):
                # Get current state
                inventory = self.env.get_inventory() if hasattr(self.env, 'get_inventory') else [0] * self.env.agent_num
                backlog = self.env.backlog if hasattr(self.env, 'backlog') else [0] * self.env.agent_num
                pipeline = self.env.pipeline_orders if hasattr(self.env, 'pipeline_orders') else [[0]] * self.env.agent_num
                
                # Get orders from policy
                orders = self.policy.get_orders(
                    inventory=inventory,
                    backlog=backlog,
                    pipeline=pipeline,
                    demand_history=demand_history,
                    current_demand=None
                )
                
                # Record orders
                for i, o in enumerate(orders):
                    order_history[i].append(o)
                
                # Step environment
                next_obs, rewards, done, info = self.env.step(orders, one_hot=False)
                
                # Record rewards (convert from nested list)
                flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                episode_rewards.extend(flat_rewards)
                
                # Record costs (negative of rewards)
                costs = [-r for r in flat_rewards]
                episode_costs.append(sum(costs))
                
                # Update demand history (for forecasting policies)
                if hasattr(self.env, 'demand_history'):
                    for i in range(self.env.agent_num):
                        if self.env.demand_history[i]:
                            demand_history[i] = self.env.demand_history[i].copy()
                
                obs = next_obs
                step += 1
            
            # Compute episode metrics
            total_cost = sum(episode_costs)
            all_costs.append(total_cost)
            all_rewards.append(sum(episode_rewards))
            all_order_history.append(order_history)
            
            # Service levels
            if hasattr(self.env, 'get_demand_history') and hasattr(self.env, 'get_fulfilled_history'):
                demand_hist = self.env.get_demand_history()
                fulfilled_hist = self.env.get_fulfilled_history()
                
                sl = []
                for i in range(self.env.agent_num):
                    total_demand = sum(demand_hist[i]) if demand_hist[i] else 1
                    total_fulfilled = sum(fulfilled_hist[i]) if fulfilled_hist[i] else 0
                    sl.append(total_fulfilled / max(1, total_demand))
                all_service_levels.append(sl)
            
            # Cycle service level
            if hasattr(self.env, 'backlog_history'):
                csl = []
                for i in range(self.env.agent_num):
                    bl_hist = self.env.backlog_history[i]
                    if bl_hist:
                        csl.append(sum(1 for b in bl_hist if b == 0) / len(bl_hist))
                    else:
                        csl.append(1.0)
                all_cycle_sl.append(csl)
            
            # Bullwhip effect (CORRECTED FORMULA)
            if demand_history[0]:  # If we have demand history
                bw = []
                for i in range(self.env.agent_num):
                    order_var = np.var(order_history[i]) if order_history[i] else 0
                    demand_var = np.var(demand_history[i]) if demand_history[i] else 1
                    bw.append(order_var / max(0.001, demand_var))  # Var(O)/Var(D)
                all_bullwhip.append(bw)
            
            if verbose:
                print(f"  Episode {ep+1}/{num_episodes}: Cost={total_cost:.2f}")
        
        # Aggregate results
        results = {
            "policy_name": self.policy.name,
            "num_episodes": num_episodes,
            "total_cost": {
                "mean": float(np.mean(all_costs)),
                "std": float(np.std(all_costs)),
                "min": float(np.min(all_costs)),
                "max": float(np.max(all_costs)),
                "all": all_costs,
            },
            "total_reward": {
                "mean": float(np.mean(all_rewards)),
                "std": float(np.std(all_rewards)),
            },
        }
        
        if all_service_levels:
            sl_array = np.array(all_service_levels)
            results["service_level"] = {
                "mean_per_agent": sl_array.mean(axis=0).tolist(),
                "overall_mean": float(sl_array.mean()),
                "overall_std": float(sl_array.std()),
            }
        
        if all_cycle_sl:
            csl_array = np.array(all_cycle_sl)
            results["cycle_service_level"] = {
                "mean_per_agent": csl_array.mean(axis=0).tolist(),
                "overall_mean": float(csl_array.mean()),
            }
        
        if all_bullwhip:
            bw_array = np.array(all_bullwhip)
            results["bullwhip_effect"] = {
                "mean_per_agent": bw_array.mean(axis=0).tolist(),
                "overall_mean": float(bw_array.mean()),
            }
        
        return results


def create_baselines_for_env(
    env,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None
) -> Dict[str, BaselinePolicy]:
    """
    Create all baseline policies configured for a specific environment.
    
    Args:
        env: The inventory environment
        demand_mean: Expected demand mean (estimated if not provided)
        demand_std: Expected demand std (estimated if not provided)
    
    Returns:
        Dictionary of policy_name -> BaselinePolicy
    """
    num_agents = env.agent_num
    lead_time = env.lead_time if hasattr(env, 'lead_time') else 2
    action_dim = env.action_dim if hasattr(env, 'action_dim') else 41
    
    # Extract cost parameters from environment
    holding_cost = env.holding_cost if hasattr(env, 'holding_cost') else [1.0] * num_agents
    backlog_cost = env.backlog_cost if hasattr(env, 'backlog_cost') else [1.0] * num_agents
    fixed_cost = env.fixed_cost if hasattr(env, 'fixed_cost') else 0.0
    
    cost_params = CostParameters(
        holding_cost=holding_cost,
        backlog_cost=backlog_cost,
        fixed_cost=fixed_cost
    )
    
    # Estimate demand stats if not provided
    if demand_mean is None:
        demand_mean = 10.0  # Default assumption
    if demand_std is None:
        demand_std = demand_mean * 0.3  # CV of 0.3
    
    baselines = {
        "BaseStock_95": BaseStockPolicy(
            num_agents=num_agents,
            lead_time=lead_time,
            cost_params=cost_params,
            action_dim=action_dim,
            target_service_level=0.95,
            demand_mean=demand_mean,
            demand_std=demand_std
        ),
        "BaseStock_99": BaseStockPolicy(
            num_agents=num_agents,
            lead_time=lead_time,
            cost_params=cost_params,
            action_dim=action_dim,
            target_service_level=0.99,
            demand_mean=demand_mean,
            demand_std=demand_std
        ),
        "(s,S)_95": sS_Policy(
            num_agents=num_agents,
            lead_time=lead_time,
            cost_params=cost_params,
            action_dim=action_dim,
            target_service_level=0.95,
            demand_mean=demand_mean,
            demand_std=demand_std
        ),
        "(R,Q)_95": RQ_Policy(
            num_agents=num_agents,
            lead_time=lead_time,
            cost_params=cost_params,
            action_dim=action_dim,
            target_service_level=0.95,
            demand_mean=demand_mean,
            demand_std=demand_std
        ),
        "MovingAvg": MovingAverageForecastPolicy(
            num_agents=num_agents,
            lead_time=lead_time,
            cost_params=cost_params,
            action_dim=action_dim,
            window_size=5,
            safety_factor=1.5
        ),
    }
    
    return baselines


def run_all_baselines(
    env,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run all baseline policies and return comparison results.
    
    Returns:
        Dictionary of policy_name -> evaluation_results
    """
    baselines = create_baselines_for_env(env, demand_mean, demand_std)
    
    all_results = {}
    
    for name, policy in baselines.items():
        if verbose:
            print(f"\nEvaluating {name}...")
            print(f"  Parameters: {policy.__dict__}")
        
        runner = BaselineRunner(env, policy)
        results = runner.evaluate(verbose=verbose)
        all_results[name] = results
        
        if verbose:
            print(f"  Mean Cost: {results['total_cost']['mean']:.2f} ± {results['total_cost']['std']:.2f}")
            if 'service_level' in results:
                print(f"  Service Level: {results['service_level']['overall_mean']:.2%}")
            if 'bullwhip_effect' in results:
                print(f"  Bullwhip: {results['bullwhip_effect']['overall_mean']:.3f}")
    
    return all_results
