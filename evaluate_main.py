"""
evaluate_main.py
================

Script for evaluating a trained RL policy on the inventory management
environments.  This script loads a saved model checkpoint, constructs
the environment according to a configuration file, and runs all
evaluation sequences to compute performance metrics such as total cost,
bullwhip effect, and service level.

Usage::

    python evaluate_main.py --config configs/train_serial.yaml --models results/serial_model_seed0.pth results/serial_model_seed1.pth

The evaluation results are printed to the console.  For more advanced
analysis, consider using the functions in :mod:`inventory_management_RL_Lot.utils.metrics`.
"""

from __future__ import annotations

import argparse
import logging
import os
import glob
import yaml
import torch
import numpy as np
from typing import Any, Dict, List

from .agents.happo_agent import HAPPOAgent
from .envs.serial_env import SerialInventoryEnv
from .envs.network_env import NetworkInventoryEnv
from .utils.logger import setup_logger
from .utils.metrics import (
    compute_episode_costs, 
    compute_bullwhip, 
    compute_service_levels,
    compute_cycle_service_level
)


def _resolve_config_path(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), path)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _resolve_env_paths(cfg: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    env_cfg = cfg.get("env", {})
    base_dir = os.path.dirname(os.path.abspath(config_path))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    def resolve_path(path: str) -> str:
        candidates = [
            os.path.abspath(os.path.join(cwd, path)),
            os.path.abspath(os.path.join(base_dir, path)),
            os.path.abspath(os.path.join(module_dir, path)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return os.path.abspath(os.path.join(base_dir, path))

    eval_data_dir = env_cfg.get("eval_data_dir")
    if isinstance(eval_data_dir, str) and not os.path.isabs(eval_data_dir):
        env_cfg["eval_data_dir"] = os.path.normpath(resolve_path(eval_data_dir))
    eval_data_dirs = env_cfg.get("eval_data_dirs")
    if isinstance(eval_data_dirs, list):
        resolved_dirs = []
        for d in eval_data_dirs:
            if isinstance(d, str) and not os.path.isabs(d):
                resolved_dirs.append(os.path.normpath(resolve_path(d)))
            else:
                resolved_dirs.append(d)
        env_cfg["eval_data_dirs"] = resolved_dirs
    cfg["env"] = env_cfg
    return cfg


def parse_config(path: str) -> Dict[str, Any]:
    resolved_path = _resolve_config_path(path)
    with open(resolved_path, "r") as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_paths(cfg, resolved_path)


def build_environment(cfg: Dict[str, Any]):
    env_cfg = cfg.get("env", {})
    env_type = env_cfg.get("env_type", "serial")
    if env_type == "serial":
        env = SerialInventoryEnv(**env_cfg)
    else:
        children = {int(k): [int(x) for x in v] for k, v in env_cfg.get("children", {}).items()}
        parents = {int(k): (int(v) if v is not None else None) for k, v in env_cfg.get("parents", {}).items()}
        other_params = {k: v for k, v in env_cfg.items() if k not in ["env_type", "children", "parents"]}
        env = NetworkInventoryEnv(children=children, parents=parents, **other_params)
    return env


def load_agent(cfg: Dict[str, Any], model_path: str, env: Any) -> HAPPOAgent:
    # Create agent with same architecture as during training
    obs_dim = env.obs_dim
    action_dim = cfg.get("env", {}).get("action_dim", 21)
    num_agents = env.agent_num
    agent_cfg = cfg.get("agent", {})
    agent = HAPPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=agent_cfg.get("hidden_dim", 64),
        critic_hidden_dim=agent_cfg.get("critic_hidden_dim", 128),
        actor_lr=agent_cfg.get("actor_lr", 3e-4),
        critic_lr=agent_cfg.get("critic_lr", 3e-4),
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        eps_clip=agent_cfg.get("eps_clip", 0.2),
        value_coef=agent_cfg.get("value_coef", 0.5),
        entropy_coef=agent_cfg.get("entropy_coef", 0.01),
    )
    # Load state dict
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    for actor, state_dict in zip(agent.actors, checkpoint["actor_state_dicts"]):
        actor.load_state_dict(state_dict)
    agent.critic_net.load_state_dict(checkpoint["critic_state_dict"])
    return agent


def evaluate_model(config_path: str, model_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate a trained model and return metrics.
    
    Args:
        config_path: Path to configuration file.
        model_path: Path to saved model weights.
        logger: Logger instance for output.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    cfg = parse_config(config_path)

    env = build_environment(cfg)
    n_eval = env.get_eval_num() if hasattr(env, "get_eval_num") else 0
    if n_eval == 0:
        logger.error("No evaluation data available in environment")
        return {}
    
    agent = load_agent(cfg, model_path, env)
    
    # Run evaluation
    total_costs: List[np.ndarray] = []
    bullwhip_metrics: List[List[float]] = []
    fill_rate_service_levels: List[List[float]] = []
    cycle_service_levels: List[List[float]] = []
    
    for ep in range(n_eval):
        obs = env.reset(train=False)
        reward_hist: List[List[float]] = []
        order_hist: List[List[int]] = [[] for _ in range(env.agent_num)]
        
        while True:
            actions, _ = agent.select_actions(obs)
            next_obs, rewards, done, _ = env.step(actions, one_hot=False)
            
            # Record rewards and actions
            reward_hist.append(rewards)
            for i, a in enumerate(actions):
                order_hist[i].append(a)
            
            obs = next_obs
            if all(done):
                break
        
        # Compute cost per agent
        costs = compute_episode_costs(reward_hist)
        total_costs.append(np.array(costs))
        
        # Compute bullwhip effect for this episode
        bw = compute_bullwhip(order_hist)
        bullwhip_metrics.append(bw)
        
        # FIXED: Compute service level using demand and fulfilled history
        demand_hist = env.get_demand_history()
        fulfilled_hist = env.get_fulfilled_history()
        
        # Fill Rate Service Level (Type 1)
        sl_fill_rate = compute_service_levels(demand_hist, fulfilled_hist)
        fill_rate_service_levels.append(sl_fill_rate)
        
        # Cycle Service Level (Type 2) - based on backlog history
        sl_cycle = compute_cycle_service_level(env.backlog_history)
        cycle_service_levels.append(sl_cycle)
    
    # Compute averages
    avg_costs = np.mean(total_costs, axis=0)
    avg_bw = np.mean(bullwhip_metrics, axis=0)
    avg_fill_rate_sl = np.mean(fill_rate_service_levels, axis=0)
    avg_cycle_sl = np.mean(cycle_service_levels, axis=0)
    
    # Log results
    logger.info(f"=" * 60)
    logger.info(f"Model: {os.path.basename(model_path)}")
    logger.info(f"=" * 60)
    logger.info(f"Average Cost per Agent: {avg_costs}")
    logger.info(f"  - Agent breakdown: {['Agent {}: {:.2f}'.format(i, c) for i, c in enumerate(avg_costs)]}")
    logger.info(f"Average Bullwhip Effect per Agent: {avg_bw}")
    logger.info(f"  - Agent breakdown: {['Agent {}: {:.4f}'.format(i, b) for i, b in enumerate(avg_bw)]}")
    logger.info(f"Average Fill Rate Service Level (Type 1): {avg_fill_rate_sl}")
    logger.info(f"  - Agent breakdown: {['Agent {}: {:.2%}'.format(i, s) for i, s in enumerate(avg_fill_rate_sl)]}")
    logger.info(f"Average Cycle Service Level (Type 2): {avg_cycle_sl}")
    logger.info(f"  - Agent breakdown: {['Agent {}: {:.2%}'.format(i, s) for i, s in enumerate(avg_cycle_sl)]}")
    logger.info(f"=" * 60)
    
    return {
        "costs": avg_costs.tolist(),
        "bullwhip": avg_bw.tolist(),
        "fill_rate_service_level": avg_fill_rate_sl.tolist(),
        "cycle_service_level": avg_cycle_sl.tolist(),
    }


def main(config_path: str, model_paths: list[str]) -> None:
    logger = setup_logger()
    
    all_results = []
    for model_path in model_paths:
        logger.info(f"Evaluating model: {model_path}")
        results = evaluate_model(config_path, model_path, logger)
        if results:
            results["model"] = model_path
            all_results.append(results)
    
    # Summary across all models
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY ACROSS ALL MODELS")
        logger.info("=" * 60)
        
        avg_costs = np.mean([r["costs"] for r in all_results], axis=0)
        std_costs = np.std([r["costs"] for r in all_results], axis=0)
        avg_bw = np.mean([r["bullwhip"] for r in all_results], axis=0)
        avg_sl = np.mean([r["fill_rate_service_level"] for r in all_results], axis=0)
        
        logger.info(f"Average Cost (mean ± std): {avg_costs} ± {std_costs}")
        logger.info(f"Average Bullwhip: {avg_bw}")
        logger.info(f"Average Fill Rate Service Level: {avg_sl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained inventory RL agent")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Path to saved model file")
    parser.add_argument("--models", nargs="+", help="One or more model paths to evaluate")
    args = parser.parse_args()
    raw_paths = []
    if args.models:
        raw_paths.extend(args.models)
    if args.model:
        raw_paths.append(args.model)

    if not raw_paths:
        parser.error("one of --model or --models is required")

    # 2. Xử lý logic tìm file thông minh
    final_model_paths = []
    for path in raw_paths:
        if os.path.isdir(path):
            # CASE A: Nếu là thư mục -> Lấy hết file .pth bên trong
            found_files = glob.glob(os.path.join(path, "*.pth"))
            final_model_paths.extend(found_files)
            print(f"[Auto-Discovery] Found {len(found_files)} models in directory: {path}")
        elif "*" in path or "?" in path:
            # CASE B: Nếu dùng ký tự đại diện (wildcard)
            found_files = glob.glob(path)
            final_model_paths.extend(found_files)
        else:
            # CASE C: File lẻ bình thường
            if os.path.exists(path):
                final_model_paths.append(path)
            else:
                print(f"Warning: File not found: {path}")


    final_model_paths = sorted(list(set(final_model_paths)))

    if not final_model_paths:
        print("Error: No valid model files found!")
        exit(1)

    print(f"Starting evaluation list ({len(final_model_paths)} models):")
    # In ra danh sách file sẽ chạy để kiểm tra
    for p in final_model_paths:
        print(f" - {p}")


    main(args.config, final_model_paths)
