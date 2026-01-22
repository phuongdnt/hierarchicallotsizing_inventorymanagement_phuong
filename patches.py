"""
patches.py
==========

Critical patches for the inventory management RL codebase.

This module provides monkey-patches and wrapper functions to fix
identified issues without modifying the original source files.

ISSUES FIXED:
1. Bullwhip calculation (CV -> Var(O)/Var(D))
2. Evaluation determinism (reset eval_index)
3. Reward/cost consistency check
4. Multi-objective claim verification

Usage:
    from patches import apply_all_patches
    apply_all_patches()

Author: Cishi (Thesis Project)
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Callable
import warnings
import functools


# =============================================================================
# PATCH 1: CORRECTED BULLWHIP EFFECT CALCULATION
# =============================================================================

def corrected_bullwhip_effect(
    order_history: List[List[int]],
    demand_history: List[List[int]]
) -> List[float]:
    """
    CORRECTED bullwhip effect calculation.
    
    Bullwhip Ratio = Var(Orders) / Var(Demand)
    
    The original implementation used CV of orders only, which is INCORRECT.
    True bullwhip measures demand amplification through the supply chain.
    
    Args:
        order_history: List of order sequences per agent [agent][time]
        demand_history: List of demand sequences per agent [agent][time]
    
    Returns:
        Bullwhip ratio for each agent (>1 means amplification)
    """
    bullwhip_ratios = []
    
    for i in range(len(order_history)):
        orders = order_history[i]
        demands = demand_history[i] if i < len(demand_history) else orders
        
        if len(orders) < 2 or len(demands) < 2:
            bullwhip_ratios.append(1.0)  # Default to no amplification
            continue
        
        order_variance = np.var(orders)
        demand_variance = np.var(demands)
        
        # Avoid division by zero
        if demand_variance < 1e-6:
            if order_variance < 1e-6:
                bullwhip_ratios.append(1.0)  # Both constant
            else:
                bullwhip_ratios.append(float('inf'))  # Infinite amplification
        else:
            bullwhip_ratios.append(order_variance / demand_variance)
    
    return bullwhip_ratios


def patch_bullwhip_function():
    """
    Monkey-patch the bullwhip_effect function in reward_functions module.
    """
    try:
        from envs import reward_functions
        
        # Store original for reference
        original_bullwhip = reward_functions.bullwhip_effect
        
        def patched_bullwhip(order_history: List[List[int]], 
                            demand_history: List[List[int]] = None) -> List[float]:
            """Patched bullwhip that warns about incorrect usage."""
            if demand_history is None:
                warnings.warn(
                    "bullwhip_effect called without demand_history. "
                    "Using CV approximation (INCORRECT for true bullwhip ratio). "
                    "Provide demand_history for correct Var(O)/Var(D) calculation.",
                    DeprecationWarning
                )
                return original_bullwhip(order_history)
            return corrected_bullwhip_effect(order_history, demand_history)
        
        reward_functions.bullwhip_effect = patched_bullwhip
        print("[PATCH] Bullwhip calculation patched to use Var(O)/Var(D)")
        return True
        
    except ImportError:
        warnings.warn("Could not patch bullwhip_effect: module not found")
        return False


# =============================================================================
# PATCH 2: EVALUATION DETERMINISM
# =============================================================================

def ensure_eval_determinism(env) -> None:
    """
    Ensure evaluation uses deterministic demand sequences.
    
    CRITICAL: Must be called before each model's evaluation to ensure
    all models see the EXACT same demand sequences.
    """
    if hasattr(env, 'eval_index'):
        env.eval_index = 0
    
    # Also reset any internal RNG state
    if hasattr(env, 'rng_seed') and env.rng_seed is not None:
        env.rng = np.random.default_rng(env.rng_seed)


def create_deterministic_evaluator(eval_func: Callable) -> Callable:
    """
    Decorator to ensure evaluation function resets eval_index.
    """
    @functools.wraps(eval_func)
    def wrapper(env, *args, **kwargs):
        ensure_eval_determinism(env)
        return eval_func(env, *args, **kwargs)
    return wrapper


# =============================================================================
# PATCH 3: REWARD/COST CONSISTENCY VERIFICATION
# =============================================================================

def verify_cost_calculation(env, rewards: List[float], verbose: bool = False) -> Dict[str, Any]:
    """
    Verify that reward calculation is consistent with cost calculation.
    
    This function helps identify the "168k vs 69.7" type discrepancies.
    
    Returns diagnostic information about the calculation.
    """
    diagnostics = {
        "reward_sum": sum(rewards),
        "cost_from_reward": -sum(rewards),
        "num_agents": env.agent_num,
        "issues": []
    }
    
    # Check if rewards are nested lists (potential bug)
    if rewards and isinstance(rewards[0], list):
        diagnostics["issues"].append(
            "Rewards are nested lists - potential double-wrapping issue"
        )
        flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
        diagnostics["flat_reward_sum"] = sum(flat_rewards)
    
    # Verify cost components if available
    if hasattr(env, 'inventory') and hasattr(env, 'backlog'):
        holding_cost = sum(
            env.inventory[i] * env.holding_cost[i] 
            for i in range(env.agent_num)
        )
        backlog_cost = sum(
            env.backlog[i] * env.backlog_cost[i] 
            for i in range(env.agent_num)
        )
        
        diagnostics["computed_holding_cost"] = holding_cost
        diagnostics["computed_backlog_cost"] = backlog_cost
        diagnostics["computed_total_cost"] = holding_cost + backlog_cost
    
    # Check for magnitude issues
    if abs(diagnostics["cost_from_reward"]) > 10000:
        diagnostics["issues"].append(
            f"Very high cost magnitude ({diagnostics['cost_from_reward']:.2f}). "
            "Check for unit/scaling issues."
        )
    
    if verbose:
        print("Cost Calculation Diagnostics:")
        for key, value in diagnostics.items():
            print(f"  {key}: {value}")
    
    return diagnostics


# =============================================================================
# PATCH 4: MULTI-OBJECTIVE VERIFICATION
# =============================================================================

def verify_multi_objective_claim(reward_function: Callable) -> Dict[str, Any]:
    """
    Analyze reward function to determine if it's truly multi-objective.
    
    TRUE multi-objective:
    - Returns vector of rewards (one per objective)
    - Uses Pareto dominance for selection
    - No linear scalarization
    
    LINEAR SCALARIZATION (what the code actually does):
    - Returns scalar reward
    - Objectives combined via weighted sum
    - NOT truly multi-objective (approximation only)
    """
    import inspect
    source = inspect.getsource(reward_function)
    
    analysis = {
        "is_scalarized": False,
        "weighted_sum_detected": False,
        "separate_objectives": False,
        "recommendation": ""
    }
    
    # Check for weighted sum patterns
    scalarization_patterns = [
        "holding_cost" in source and "backlog_cost" in source and "+" in source,
        "sum(" in source,
        "np.sum(" in source,
        "total" in source.lower(),
    ]
    
    if any(scalarization_patterns):
        analysis["is_scalarized"] = True
        analysis["weighted_sum_detected"] = True
        analysis["recommendation"] = (
            "The reward function uses LINEAR SCALARIZATION (weighted sum of objectives). "
            "This is NOT true multi-objective optimization. "
            "For academic accuracy, describe as 'multi-criteria' or 'weighted sum' approach, "
            "not 'multi-objective' unless implementing Pareto-based selection."
        )
    
    return analysis


# =============================================================================
# PATCH 5: OBSERVATION SPACE ENHANCEMENT (Optional)
# =============================================================================

def compute_enhanced_observation(
    inventory: int,
    backlog: int,
    pipeline: List[int],
    demand_history: List[int],
    max_val: int = 40
) -> np.ndarray:
    """
    Compute enhanced observation with additional features.
    
    Original: [inventory, backlog, downstream_demand, pipeline...]
    
    Enhanced adds:
    - Inventory position (inv + pipeline - backlog)
    - Demand statistics (mean, std of recent history)
    - Normalized values
    """
    # Base features
    inv_pos = inventory + sum(pipeline) - backlog
    
    # Demand statistics (if available)
    if demand_history and len(demand_history) >= 3:
        demand_mean = np.mean(demand_history[-10:])
        demand_std = np.std(demand_history[-10:]) if len(demand_history) > 1 else 0
    else:
        demand_mean = 10.0
        demand_std = 3.0
    
    # Construct enhanced observation
    obs = np.array([
        inventory / max_val,
        backlog / max_val,
        inv_pos / max_val,
        demand_mean / max_val,
        demand_std / max_val,
    ] + [p / max_val for p in pipeline], dtype=np.float32)
    
    return obs


# =============================================================================
# MASTER PATCH APPLICATION
# =============================================================================

_patches_applied = False

def apply_all_patches(verbose: bool = True) -> Dict[str, bool]:
    """
    Apply all critical patches to the codebase.
    
    Returns dictionary of patch_name -> success_status
    """
    global _patches_applied
    
    if _patches_applied:
        if verbose:
            print("[INFO] Patches already applied")
        return {}
    
    results = {}
    
    if verbose:
        print("=" * 50)
        print("APPLYING CRITICAL PATCHES")
        print("=" * 50)
    
    # Patch 1: Bullwhip calculation
    results["bullwhip"] = patch_bullwhip_function()
    
    # Patch 2: Add warning to reward_functions
    try:
        from envs import reward_functions
        reward_functions.corrected_bullwhip_effect = corrected_bullwhip_effect
        if verbose:
            print("[PATCH] Added corrected_bullwhip_effect to reward_functions")
        results["corrected_bullwhip_added"] = True
    except ImportError:
        results["corrected_bullwhip_added"] = False
    
    _patches_applied = True
    
    if verbose:
        print("=" * 50)
        print(f"Patches applied: {sum(results.values())}/{len(results)}")
        print("=" * 50)
    
    return results


# =============================================================================
# DIAGNOSTIC REPORT
# =============================================================================

def generate_diagnostic_report(env, verbose: bool = True) -> Dict[str, Any]:
    """
    Generate comprehensive diagnostic report for the environment.
    
    Useful for debugging the "168k vs 69.7" type discrepancies.
    """
    report = {
        "environment": type(env).__name__,
        "num_agents": env.agent_num,
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim,
        "lead_time": getattr(env, 'lead_time', 'N/A'),
        "holding_costs": getattr(env, 'holding_cost', 'N/A'),
        "backlog_costs": getattr(env, 'backlog_cost', 'N/A'),
        "fixed_cost": getattr(env, 'fixed_cost', 'N/A'),
        "reward_smoothing_alpha": getattr(env, 'alpha', 'N/A'),
    }
    
    # Check for common issues
    issues = []
    
    # Issue 1: Reward smoothing during training
    if hasattr(env, 'alpha') and env.alpha != 1.0:
        issues.append(
            f"Reward smoothing enabled (alpha={env.alpha}). "
            "Training rewards differ from evaluation rewards!"
        )
    
    # Issue 2: Cost magnitude
    if hasattr(env, 'backlog_cost'):
        max_cost = max(env.backlog_cost) if env.backlog_cost else 0
        if max_cost > 10:
            issues.append(
                f"High backlog cost ({max_cost}). "
                "May cause large cost values."
            )
    
    # Issue 3: Fixed cost
    if hasattr(env, 'fixed_cost') and env.fixed_cost > 0:
        issues.append(
            f"Fixed ordering cost enabled ({env.fixed_cost}). "
            "Adds to total cost for every non-zero order."
        )
    
    report["potential_issues"] = issues
    
    if verbose:
        print("\n" + "=" * 50)
        print("ENVIRONMENT DIAGNOSTIC REPORT")
        print("=" * 50)
        for key, value in report.items():
            if key != "potential_issues":
                print(f"  {key}: {value}")
        
        if issues:
            print("\nPOTENTIAL ISSUES:")
            for issue in issues:
                print(f"  ⚠ {issue}")
        print("=" * 50)
    
    return report


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Patches module loaded. Run apply_all_patches() to apply fixes.")
    
    # Test corrected bullwhip
    orders = [[10, 15, 8, 20, 12]]
    demands = [[10, 12, 11, 10, 13]]
    
    old_style = [np.std(orders[0]) / np.mean(orders[0])]  # CV only
    new_style = corrected_bullwhip_effect(orders, demands)
    
    print(f"\nBullwhip test:")
    print(f"  Old (CV only): {old_style[0]:.4f}")
    print(f"  New (Var(O)/Var(D)): {new_style[0]:.4f}")
    print(f"  Order variance: {np.var(orders[0]):.4f}")
    print(f"  Demand variance: {np.var(demands[0]):.4f}")
