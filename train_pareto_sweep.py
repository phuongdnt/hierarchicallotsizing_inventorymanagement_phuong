"""
train_pareto_sweep.py (STANDALONE VERSION)
==========================================

Train multiple models with different cost weight combinations
to explore the Pareto frontier of the multi-objective problem.

✅ STANDALONE: Không phụ thuộc vào train_main.py
✅ COMPATIBLE: Works với existing code structure
✅ SUPPORTS: Cả Serial (1-1-1) và Network (2x3) topologies

Usage:
    # Serial topology
    python train_pareto_sweep.py --config configs/train_serial.yaml --output results/pareto_serial
    
    # Network topology
    python train_pareto_sweep.py --config configs/train_network.yaml --output results/pareto_network

Author: Cishi (Thesis Project)
"""

from __future__ import annotations

import argparse
import os
import sys
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import yaml

# =============================================================================
# SEED CONTROL (Comprehensive)
# =============================================================================

def set_seed(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# WEIGHT CONFIGURATIONS FOR PARETO SWEEP
# =============================================================================

# Mỗi config định nghĩa tỉ lệ holding:backlog
# Base costs sẽ được nhân với các weights này
PARETO_WEIGHTS = [
    # Balanced (baseline)
    {
        "name": "balanced",
        "holding_multiplier": 1.0,
        "backlog_multiplier": 1.0,
        "description": "Equal weight on holding and backlog"
    },
    
    # Favor low inventory (higher holding penalty)
    {
        "name": "low_inventory",
        "holding_multiplier": 2.0,
        "backlog_multiplier": 1.0,
        "description": "Penalize holding more -> lean inventory"
    },
    {
        "name": "very_low_inventory",
        "holding_multiplier": 3.0,
        "backlog_multiplier": 1.0,
        "description": "Strong penalty on holding"
    },
    
    # Favor high service (higher backlog penalty)
    {
        "name": "high_service",
        "holding_multiplier": 1.0,
        "backlog_multiplier": 2.0,
        "description": "Penalize backlog more -> better service"
    },
    {
        "name": "max_service",
        "holding_multiplier": 1.0,
        "backlog_multiplier": 5.0,
        "description": "Strong penalty on stockout"
    },
]

# Minimal set for quick testing
PARETO_WEIGHTS_MINIMAL = [
    {"name": "balanced", "holding_multiplier": 1.0, "backlog_multiplier": 1.0},
    {"name": "low_inv", "holding_multiplier": 2.0, "backlog_multiplier": 1.0},
    {"name": "high_svc", "holding_multiplier": 1.0, "backlog_multiplier": 3.0},
]


# =============================================================================
# ENVIRONMENT BUILDER (STANDALONE)
# =============================================================================

def build_environment(config: Dict[str, Any], seed: int = None):
    """
    Build environment from config dict.
    
    Supports both Serial and Network topologies based on config.
    
    Args:
        config: Config dict loaded from YAML
        seed: Random seed for environment
    
    Returns:
        Environment instance (SerialInventoryEnv or NetworkInventoryEnv)
    """
    # Import environments
    from envs.serial_env import SerialInventoryEnv
    from envs.network_env import NetworkInventoryEnv
    
    env_cfg = config.get('env', {}).copy()
    
    # Override seed if provided
    if seed is not None:
        env_cfg['rng_seed'] = seed
    
    # Determine environment type
    env_type = env_cfg.pop('env_type', None)
    
    # Auto-detect from config if not specified
    if env_type is None:
        if 'children' in env_cfg or 'parents' in env_cfg:
            env_type = 'network'
        else:
            env_type = 'serial'
    
    print(f"  [ENV] Building {env_type} environment")
    
    if env_type == 'serial':
        # Remove network-specific params
        env_cfg.pop('children', None)
        env_cfg.pop('parents', None)
        env_cfg.pop('eval_data_dirs', None)
        
        return SerialInventoryEnv(**env_cfg)
    
    elif env_type == 'network':
        # Parse children/parents from config (may be string keys in YAML)
        children = env_cfg.pop('children', {})
        parents = env_cfg.pop('parents', {})
        
        # Convert string keys to int
        children = {int(k): [int(x) for x in v] for k, v in children.items()}
        parents = {int(k): (int(v) if v is not None else None) for k, v in parents.items()}
        
        return NetworkInventoryEnv(children=children, parents=parents, **env_cfg)
    
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def build_agent(env, config: Dict[str, Any]):
    """
    Build HAPPO agent from config.
    
    Args:
        env: Environment instance
        config: Config dict
    
    Returns:
        HAPPOAgent instance
    """
    from agents.happo_agent import HAPPOAgent
    
    agent_cfg = config.get('agent', {})
    env_cfg = config.get('env', {})
    
    agent = HAPPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env_cfg.get('action_dim', 41),
        num_agents=env.agent_num,
        hidden_dim=agent_cfg.get('hidden_dim', 128),
        critic_hidden_dim=agent_cfg.get('critic_hidden_dim', 256),
        actor_lr=agent_cfg.get('actor_lr', 1e-4),
        critic_lr=agent_cfg.get('critic_lr', 1e-4),
        gamma=agent_cfg.get('gamma', 0.99),
        gae_lambda=agent_cfg.get('gae_lambda', 0.95),
        eps_clip=agent_cfg.get('eps_clip', 0.2),
    )
    
    return agent


# =============================================================================
# CONFIG LOADING & MODIFICATION
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths
    config_dir = os.path.dirname(os.path.abspath(config_path))
    env_cfg = config.get('env', {})
    
    # Handle eval_data_dir
    if 'eval_data_dir' in env_cfg and env_cfg['eval_data_dir']:
        if not os.path.isabs(env_cfg['eval_data_dir']):
            env_cfg['eval_data_dir'] = os.path.join(config_dir, env_cfg['eval_data_dir'])
    
    # Handle eval_data_dirs (for network)
    if 'eval_data_dirs' in env_cfg and env_cfg['eval_data_dirs']:
        resolved_dirs = []
        for d in env_cfg['eval_data_dirs']:
            if not os.path.isabs(d):
                d = os.path.join(config_dir, d)
            resolved_dirs.append(d)
        env_cfg['eval_data_dirs'] = resolved_dirs
    
    config['env'] = env_cfg
    return config


def apply_weight_config(
    base_config: Dict[str, Any],
    weight_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply weight multipliers to base config.
    
    Args:
        base_config: Original config from YAML
        weight_config: Dict with holding_multiplier and backlog_multiplier
    
    Returns:
        Modified config with adjusted costs
    """
    config = copy.deepcopy(base_config)
    env_cfg = config.get('env', {})
    
    # Get base costs from config
    base_holding = env_cfg.get('holding_cost', [1.0, 1.0, 1.0])
    base_backlog = env_cfg.get('backlog_cost', [5.0, 3.0, 2.0])
    
    # Ensure lists
    if not isinstance(base_holding, list):
        base_holding = [base_holding]
    if not isinstance(base_backlog, list):
        base_backlog = [base_backlog]
    
    # Apply multipliers
    h_mult = weight_config.get('holding_multiplier', 1.0)
    b_mult = weight_config.get('backlog_multiplier', 1.0)
    
    env_cfg['holding_cost'] = [h * h_mult for h in base_holding]
    env_cfg['backlog_cost'] = [b * b_mult for b in base_backlog]
    
    config['env'] = env_cfg
    return config


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_single_run(
    config: Dict[str, Any],
    weight_name: str,
    seed: int,
    output_dir: Path,
    episodes: int = 5000,
    log_every: int = 500,
    save_checkpoints: bool = True
) -> Dict[str, Any]:
    """
    Train a single model with given config and seed.
    
    Args:
        config: Full config dict
        weight_name: Name of weight configuration
        seed: Random seed
        output_dir: Directory to save model
        episodes: Number of training episodes
        log_every: Log frequency
        save_checkpoints: Whether to save intermediate checkpoints
    
    Returns:
        Dict with training results
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {weight_name} | Seed: {seed}")
    print(f"{'='*60}")
    print(f"  Holding costs: {config['env']['holding_cost']}")
    print(f"  Backlog costs: {config['env']['backlog_cost']}")
    
    # Set seed
    set_seed(seed)
    
    # Build environment and agent
    env = build_environment(config, seed=seed)
    agent = build_agent(env, config)
    
    print(f"  Env type: {type(env).__name__}")
    print(f"  Num agents: {env.agent_num}")
    print(f"  Obs dim: {env.obs_dim}")
    print(f"{'='*60}")
    
    # Training
    reward_history = []
    best_reward = float('-inf')
    
    for ep in range(1, episodes + 1):
        obs = env.reset(train=True)
        ep_reward = 0.0
        done = [False] * env.agent_num
        
        step = 0
        max_steps = config.get('env', {}).get('episode_length', 200)
        
        while not all(done) and step < max_steps:
            actions, log_probs = agent.select_actions(obs)
            next_obs, rewards, done, info = env.step(actions, one_hot=False)
            
            # Flatten rewards if nested
            flat_rewards = []
            for r in rewards:
                if isinstance(r, (list, np.ndarray)):
                    flat_rewards.append(float(r[0]) if len(r) > 0 else 0.0)
                else:
                    flat_rewards.append(float(r))
            
            # Store transition
            agent.store_transition(
                obs, actions, log_probs, flat_rewards, next_obs, 
                done[0] if isinstance(done, list) else done
            )
            
            ep_reward += sum(flat_rewards)
            obs = next_obs
            step += 1
        
        # Update agent
        agent.update()
        reward_history.append(ep_reward)
        
        # Logging
        if ep % log_every == 0:
            recent_avg = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            print(f"  Episode {ep:5d}/{episodes} | Reward: {ep_reward:8.2f} | Avg(100): {recent_avg:8.2f}")
        
        # Track best
        if ep_reward > best_reward:
            best_reward = ep_reward
    
    # Save final model
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{weight_name}_seed{seed}.pth"
    
    torch.save({
        'actor_state_dicts': [a.state_dict() for a in agent.actors],
        'critic_state_dict': agent.critic_net.state_dict(),
        'weight_name': weight_name,
        'weight_config': {
            'holding_cost': config['env']['holding_cost'],
            'backlog_cost': config['env']['backlog_cost'],
        },
        'seed': seed,
        'episodes': episodes,
        'final_reward': reward_history[-1] if reward_history else 0,
        'best_reward': best_reward,
    }, model_path)
    
    print(f"  ✅ Model saved: {model_path}")
    
    # Return results
    return {
        'weight_name': weight_name,
        'seed': seed,
        'model_path': str(model_path),
        'episodes': episodes,
        'final_reward': float(reward_history[-1]) if reward_history else 0,
        'best_reward': float(best_reward),
        'mean_reward_last100': float(np.mean(reward_history[-100:])) if len(reward_history) >= 100 else float(np.mean(reward_history)),
        'holding_cost': config['env']['holding_cost'],
        'backlog_cost': config['env']['backlog_cost'],
    }


# =============================================================================
# PARETO SWEEP
# =============================================================================

def run_pareto_sweep(
    config_path: str,
    output_dir: str,
    seeds: List[int] = [1, 2, 3],
    episodes: int = 5000,
    weight_indices: List[int] = None,
    minimal: bool = False
) -> Dict[str, Any]:
    """
    Run complete Pareto weight sweep.
    
    Args:
        config_path: Path to base config YAML
        output_dir: Output directory
        seeds: List of random seeds
        episodes: Episodes per run
        weight_indices: Specific weight configs to run (None = all)
        minimal: Use minimal weight set (3 configs)
    
    Returns:
        Summary dict with all results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load base config
    base_config = load_config(config_path)
    
    # Select weights
    weights = PARETO_WEIGHTS_MINIMAL if minimal else PARETO_WEIGHTS
    if weight_indices is not None:
        weights = [weights[i] for i in weight_indices]
    
    # Summary
    total_runs = len(weights) * len(seeds)
    print(f"\n{'#'*60}")
    print(f"PARETO WEIGHT SWEEP")
    print(f"{'#'*60}")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print(f"Weight configs: {len(weights)}")
    print(f"Seeds: {seeds}")
    print(f"Episodes per run: {episodes}")
    print(f"Total training runs: {total_runs}")
    print(f"{'#'*60}\n")
    
    # Save config
    config_save_path = output_path / "sweep_config.json"
    with open(config_save_path, 'w') as f:
        json.dump({
            'base_config_path': config_path,
            'seeds': seeds,
            'episodes': episodes,
            'weights': weights,
        }, f, indent=2)
    
    # Run all configurations
    all_results = []
    run_idx = 0
    
    for weight_cfg in weights:
        for seed in seeds:
            run_idx += 1
            print(f"\n[RUN {run_idx}/{total_runs}]")
            
            # Apply weights to config
            modified_config = apply_weight_config(base_config, weight_cfg)
            
            # Train
            result = train_single_run(
                config=modified_config,
                weight_name=weight_cfg['name'],
                seed=seed,
                output_dir=output_path,
                episodes=episodes
            )
            
            all_results.append(result)
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config_path': config_path,
        'output_dir': str(output_path),
        'seeds': seeds,
        'episodes': episodes,
        'num_weight_configs': len(weights),
        'total_runs': total_runs,
        'results': all_results
    }
    
    summary_path = output_path / "pareto_sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Summary: {summary_path}")
    print(f"Models: {output_path / 'models'}")
    print(f"{'='*60}")
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pareto Weight Sweep Training for Multi-Objective Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train serial topology with all weights
  python train_pareto_sweep.py --config configs/train_serial.yaml --output results/pareto_serial
  
  # Train network topology with minimal weights (faster)
  python train_pareto_sweep.py --config configs/train_network.yaml --output results/pareto_network --minimal
  
  # Train specific weight configs (indices 0,2,4)
  python train_pareto_sweep.py --config configs/train_serial.yaml --weights 0 2 4
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base config YAML (train_serial.yaml or train_network.yaml)')
    parser.add_argument('--output', type=str, default='results/pareto',
                       help='Output directory (default: results/pareto)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3],
                       help='Random seeds (default: 1 2 3)')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Episodes per run (default: 5000)')
    parser.add_argument('--weights', type=int, nargs='+', default=None,
                       help='Indices of weight configs to run (default: all)')
    parser.add_argument('--minimal', action='store_true',
                       help='Use minimal weight set (3 configs instead of 5)')
    
    args = parser.parse_args()
    
    run_pareto_sweep(
        config_path=args.config,
        output_dir=args.output,
        seeds=args.seeds,
        episodes=args.episodes,
        weight_indices=args.weights,
        minimal=args.minimal
    )


if __name__ == '__main__':
    main()
