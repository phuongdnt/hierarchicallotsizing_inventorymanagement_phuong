"""
run_comprehensive_eval.py
=========================

Comprehensive evaluation script that:
1. Evaluates RL models (HAPPO, Hybrid)
2. Evaluates classical baselines (Base Stock, (s,S), etc.)
3. Ensures FAIR comparison (same demand sequences, same cost function)
4. Generates all thesis visualizations
5. Produces structured JSON results

FIXES APPLIED:
- Proper seed management
- Reset eval_index before each model evaluation
- Corrected bullwhip formula (Var(O)/Var(D))
- Consistent cost calculation across all methods

Usage:
    python run_comprehensive_eval.py --config configs/train_serial.yaml --models results/

Author: Cishi (Thesis Project)
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.happo_agent import HAPPOAgent
from envs.serial_env import SerialInventoryEnv
from envs.network_env import NetworkInventoryEnv
from baselines import (
    create_baselines_for_env,
    BaselineRunner,
    CostParameters
)
from utils.experiment_utils import (
    set_seed,
    setup_directories,
    ArtifactSaver,
    create_eval_result
)
from utils.visualize import (
    generate_all_figures,
    create_summary_table
)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and resolve configuration paths."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Resolve relative paths
    base_dir = os.path.dirname(os.path.abspath(config_path))
    env_cfg = cfg.get('env', {})
    
    for key in ['eval_data_dir', 'eval_data_dirs']:
        if key in env_cfg:
            val = env_cfg[key]
            if isinstance(val, str) and not os.path.isabs(val):
                env_cfg[key] = os.path.join(base_dir, val)
            elif isinstance(val, list):
                env_cfg[key] = [
                    os.path.join(base_dir, v) if not os.path.isabs(v) else v
                    for v in val
                ]
    
    cfg['env'] = env_cfg
    return cfg


def build_env(cfg: Dict[str, Any], seed: Optional[int] = None):
    """Build environment from configuration."""
    env_cfg = cfg.get('env', {}).copy()
    
    if seed is not None:
        env_cfg['rng_seed'] = seed
    
    env_type = env_cfg.pop('env_type', 'serial')
    
    if env_type == 'serial':
        # Remove network-specific keys
        env_cfg.pop('children', None)
        env_cfg.pop('parents', None)
        env_cfg.pop('eval_data_dirs', None)
        return SerialInventoryEnv(**env_cfg)
    else:
        children = {int(k): [int(x) for x in v] 
                   for k, v in env_cfg.pop('children', {}).items()}
        parents = {int(k): (int(v) if v is not None else None)
                  for k, v in env_cfg.pop('parents', {}).items()}
        return NetworkInventoryEnv(children=children, parents=parents, **env_cfg)


# =============================================================================
# RL MODEL EVALUATION
# =============================================================================

def load_rl_agent(cfg: Dict[str, Any], model_path: str, env) -> HAPPOAgent:
    """Load trained RL agent from checkpoint."""
    obs_dim = env.obs_dim
    action_dim = cfg.get('env', {}).get('action_dim', 21)
    num_agents = env.agent_num
    agent_cfg = cfg.get('agent', {})
    
    agent = HAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=agent_cfg.get('hidden_dim', 64),
        critic_hidden_dim=agent_cfg.get('critic_hidden_dim', 128),
        actor_lr=agent_cfg.get('actor_lr', 3e-4),
        critic_lr=agent_cfg.get('critic_lr', 3e-4),
        gamma=agent_cfg.get('gamma', 0.99),
        gae_lambda=agent_cfg.get('gae_lambda', 0.95),
        eps_clip=agent_cfg.get('eps_clip', 0.2),
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    for actor, state_dict in zip(agent.actors, checkpoint['actor_state_dicts']):
        actor.load_state_dict(state_dict)
    agent.critic_net.load_state_dict(checkpoint['critic_state_dict'])
    
    return agent


def evaluate_rl_model(
    env,
    agent: HAPPOAgent,
    model_name: str,
    num_episodes: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate RL agent with comprehensive metrics.
    
    CRITICAL FIX: Reset eval_index to ensure deterministic evaluation.
    """
    # CRITICAL: Reset evaluation index for fair comparison
    if hasattr(env, 'eval_index'):
        env.eval_index = 0
    
    if num_episodes is None:
        num_episodes = env.get_eval_num() if hasattr(env, 'get_eval_num') else 30
    
    all_costs = []
    all_rewards = []
    all_service_levels = []
    all_cycle_sl = []
    all_bullwhip = []
    
    for ep in range(num_episodes):
        obs = env.reset(train=False)
        
        episode_rewards = []
        order_history = [[] for _ in range(env.agent_num)]
        
        done = [False] * env.agent_num
        
        while not all(done):
            actions, _ = agent.select_actions(obs)
            
            # Record orders
            for i, a in enumerate(actions):
                order_history[i].append(a)
            
            obs, rewards, done, _ = env.step(actions, one_hot=False)
            
            flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
            episode_rewards.extend(flat_rewards)
        
        # Compute metrics
        total_reward = sum(episode_rewards)
        total_cost = -total_reward  # Cost = negative reward
        
        all_rewards.append(total_reward)
        all_costs.append(total_cost)
        
        # Service levels
        if hasattr(env, 'get_demand_history') and hasattr(env, 'get_fulfilled_history'):
            demand_hist = env.get_demand_history()
            fulfilled_hist = env.get_fulfilled_history()
            
            sl = []
            for i in range(env.agent_num):
                total_demand = sum(demand_hist[i]) if demand_hist[i] else 1
                total_fulfilled = sum(fulfilled_hist[i]) if fulfilled_hist[i] else 0
                sl.append(min(1.0, total_fulfilled / max(1, total_demand)))
            all_service_levels.append(sl)
            
            # CORRECTED BULLWHIP: Var(Orders) / Var(Demand)
            bw = []
            for i in range(env.agent_num):
                order_var = np.var(order_history[i]) if len(order_history[i]) > 1 else 0
                demand_var = np.var(demand_hist[i]) if len(demand_hist[i]) > 1 else 1
                bw.append(order_var / max(0.001, demand_var))
            all_bullwhip.append(bw)
        
        # Cycle service level
        if hasattr(env, 'backlog_history'):
            csl = []
            for i in range(env.agent_num):
                bl_hist = env.backlog_history[i]
                if bl_hist:
                    csl.append(sum(1 for b in bl_hist if b == 0) / len(bl_hist))
                else:
                    csl.append(1.0)
            all_cycle_sl.append(csl)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes}: Cost={total_cost:.2f}")
    
    # Aggregate results
    results = {
        "model_name": model_name,
        "model_type": "RL",
        "num_episodes": num_episodes,
        "total_cost": {
            "mean": float(np.mean(all_costs)),
            "std": float(np.std(all_costs)),
            "min": float(np.min(all_costs)),
            "max": float(np.max(all_costs)),
            "all": [float(c) for c in all_costs],
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
            "overall_std": float(bw_array.std()),
        }
    
    return results


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_comprehensive_evaluation(
    config_path: str,
    model_paths: List[str],
    output_dir: str = "eval_results",
    run_baselines: bool = True,
    generate_figures: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of all models.
    
    Args:
        config_path: Path to configuration file
        model_paths: List of paths to trained RL models
        output_dir: Directory for output files
        run_baselines: Whether to evaluate classical baselines
        generate_figures: Whether to generate visualization figures
        verbose: Print progress
    
    Returns:
        Dictionary with all evaluation results
    """
    # Setup
    cfg = load_config(config_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Set global seed for reproducibility
    set_seed(42)
    
    # Build environment
    env = build_env(cfg)
    
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 60)
    print(f"Environment: {type(env).__name__}")
    print(f"Agents: {env.agent_num}")
    print(f"Action dim: {env.action_dim}")
    print(f"Lead time: {env.lead_time}")
    print(f"Eval episodes: {env.get_eval_num() if hasattr(env, 'get_eval_num') else 'N/A'}")
    print("=" * 60)
    
    # =================================
    # 1. EVALUATE RL MODELS
    # =================================
    print("\n[1] Evaluating RL Models...")
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"  WARNING: Model not found: {model_path}")
            continue
        
        model_name = Path(model_path).stem
        
        # Determine model type from name
        if 'hybrid' in model_name.lower():
            model_type = "Hybrid_MARL"
        else:
            model_type = "HAPPO"
        
        print(f"\n  Evaluating: {model_name}")
        
        try:
            # Rebuild environment to reset state
            env = build_env(cfg)
            agent = load_rl_agent(cfg, model_path, env)
            
            results = evaluate_rl_model(
                env, agent, model_name, verbose=verbose
            )
            results['model_path'] = model_path
            results['model_type'] = model_type
            
            all_results[model_name] = results
            
            print(f"    Cost: {results['total_cost']['mean']:.2f} ± {results['total_cost']['std']:.2f}")
            if 'service_level' in results:
                print(f"    Service Level: {results['service_level']['overall_mean']:.2%}")
            if 'bullwhip_effect' in results:
                print(f"    Bullwhip: {results['bullwhip_effect']['overall_mean']:.3f}")
        
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # =================================
    # 2. EVALUATE BASELINES
    # =================================
    if run_baselines:
        print("\n[2] Evaluating Classical Baselines...")
        
        # Estimate demand statistics from evaluation data
        env = build_env(cfg)
        env.reset(train=False)
        
        demand_mean = 10.0
        demand_std = 5.0
        
        if hasattr(env, 'eval_data') and env.eval_data:
            all_demands = []
            for seq in env.eval_data:
                if isinstance(seq, list) and all(isinstance(x, int) for x in seq):
                    all_demands.extend(seq)
                elif isinstance(seq, list):
                    for subseq in seq:
                        if isinstance(subseq, list):
                            all_demands.extend(subseq)
            
            if all_demands:
                demand_mean = float(np.mean(all_demands))
                demand_std = float(np.std(all_demands))
                print(f"  Demand stats: mean={demand_mean:.2f}, std={demand_std:.2f}")
        
        # Create and evaluate baselines
        baselines = create_baselines_for_env(env, demand_mean, demand_std)
        
        for name, policy in baselines.items():
            print(f"\n  Evaluating: {name}")
            
            # Rebuild environment for each baseline
            env = build_env(cfg)
            
            runner = BaselineRunner(env, policy)
            results = runner.evaluate(verbose=False)
            results['model_type'] = 'Heuristic'
            
            all_results[name] = results
            
            print(f"    Cost: {results['total_cost']['mean']:.2f} ± {results['total_cost']['std']:.2f}")
            if 'service_level' in results:
                print(f"    Service Level: {results['service_level']['overall_mean']:.2%}")
            if 'bullwhip_effect' in results:
                print(f"    Bullwhip: {results['bullwhip_effect']['overall_mean']:.3f}")
    
    # =================================
    # 3. SAVE RESULTS
    # =================================
    print("\n[3] Saving Results...")
    
    # Save comprehensive JSON
    results_file = output_path / "comprehensive_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config_path": config_path,
            "results": all_results,
        }, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"  Results saved to: {results_file}")
    
    # Save summary table (Markdown)
    table = create_summary_table(all_results)
    table_file = output_path / "summary_table.md"
    with open(table_file, 'w') as f:
        f.write("# Evaluation Results Summary\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(table)
    print(f"  Summary table saved to: {table_file}")
    
    # =================================
    # 4. GENERATE FIGURES
    # =================================
    if generate_figures:
        print("\n[4] Generating Figures...")
        
        figures_dir = output_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        try:
            saved_figures = generate_all_figures(
                all_results,
                training_histories=None,  # Would need to load separately
                output_dir=str(figures_dir),
                prefix="thesis"
            )
            print(f"  Figures saved to: {figures_dir}")
        except Exception as e:
            print(f"  WARNING: Figure generation failed: {e}")
    
    # =================================
    # 5. PRINT FINAL SUMMARY
    # =================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    # Sort by cost
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get('total_cost', {}).get('mean', float('inf'))
    )
    
    print(f"\n{'Rank':<5} {'Model':<25} {'Cost':<20} {'Service Level':<15} {'Bullwhip':<10}")
    print("-" * 75)
    
    for rank, (name, data) in enumerate(sorted_models, 1):
        cost = data.get('total_cost', {}).get('mean', 'N/A')
        cost_std = data.get('total_cost', {}).get('std', 0)
        sl = data.get('service_level', {}).get('overall_mean', 'N/A')
        bw = data.get('bullwhip_effect', {}).get('overall_mean', 'N/A')
        
        cost_str = f"{cost:.2f}±{cost_std:.2f}" if isinstance(cost, float) else str(cost)
        sl_str = f"{sl:.1%}" if isinstance(sl, float) else str(sl)
        bw_str = f"{bw:.3f}" if isinstance(bw, float) else str(bw)
        
        print(f"{rank:<5} {name:<25} {cost_str:<20} {sl_str:<15} {bw_str:<10}")
    
    print("=" * 60)
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of inventory management models"
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--models', type=str, nargs='+',
        help='Paths to trained model files or directory'
    )
    parser.add_argument(
        '--output', type=str, default='eval_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-baselines', action='store_true',
        help='Skip baseline evaluation'
    )
    parser.add_argument(
        '--no-figures', action='store_true',
        help='Skip figure generation'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Resolve model paths
    model_paths = []
    if args.models:
        for path in args.models:
            if os.path.isdir(path):
                # Find all .pth files in directory
                model_paths.extend(glob.glob(os.path.join(path, '*.pth')))
                model_paths.extend(glob.glob(os.path.join(path, '**/*.pth'), recursive=True))
            elif '*' in path:
                model_paths.extend(glob.glob(path))
            elif os.path.exists(path):
                model_paths.append(path)
    
    model_paths = sorted(list(set(model_paths)))
    
    if not model_paths:
        print("No model files found. Running baselines only.")
    else:
        print(f"Found {len(model_paths)} model(s) to evaluate")
    
    # Run evaluation
    run_comprehensive_evaluation(
        config_path=args.config,
        model_paths=model_paths,
        output_dir=args.output,
        run_baselines=not args.no_baselines,
        generate_figures=not args.no_figures,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
