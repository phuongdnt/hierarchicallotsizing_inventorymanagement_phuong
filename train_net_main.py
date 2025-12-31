  """
train_net_main.py
Script huấn luyện riêng cho Network, hỗ trợ chạy nhiều seed và sửa lỗi Unicode/Import.
"""
from __future__ import annotations

import argparse
import os
import yaml
import random
from typing import Any, Dict, List

import numpy as np
import torch

# --- KHÁC BIỆT 1: Import tuyệt đối (xóa dấu chấm) ---
from utils.logger import setup_logger
from agents.happo_agent import HAPPOAgent
from envs.serial_env import SerialInventoryEnv
from envs.network_env import NetworkInventoryEnv
from envs.vec_env import SubprocVecEnv, DummyVecEnv
from lot_sizing.hybrid_planner import HybridPlanner
from utils.metrics import compute_episode_costs

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
    # --- KHÁC BIỆT 2: Thêm encoding="utf-8" để chạy trên Windows ---
    with open(resolved_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_paths(cfg, resolved_path)

def set_seed(seed: int):
    """Thiết lập seed cho tất cả các thư viện để đảm bảo tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_environment(cfg: Dict[str, Any], seed: int = None):
    env_cfg = cfg.get("env", {})
    if seed is not None:
        env_cfg["rng_seed"] = seed
        
    env_type = env_cfg.get("env_type", "serial")
    if env_type == "serial":
        env = SerialInventoryEnv(**env_cfg)
    elif env_type == "network":
        children = {int(k): [int(x) for x in v] for k, v in env_cfg.get("children", {}).items()}
        parents = {int(k): (int(v) if v is not None else None) for k, v in env_cfg.get("parents", {}).items()}
        other_params = {k: v for k, v in env_cfg.items() if k not in ["env_type", "children", "parents"]}
        env = NetworkInventoryEnv(children=children, parents=parents, **other_params)
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")
    return env

def main(config_path: str, seed: int = None) -> None:
    cfg = parse_config(config_path)
    
    # --- KHÁC BIỆT 3: Xử lý seed và đổi tên file save ---
    if seed is not None:
        set_seed(seed)
        print(f"Running with Seed: {seed}")
        training_cfg = cfg.get("training", {})
        save_path = training_cfg.get("save_path")
        if save_path:
            # Tự động thêm _seed_X vào tên file
            name, ext = os.path.splitext(save_path)
            new_save_path = f"{name}_seed_{seed}{ext}"
            training_cfg["save_path"] = new_save_path
            print(f"Model will be saved to: {new_save_path}")
        cfg["training"] = training_cfg

    logger = setup_logger()
    base_env = build_environment(cfg, seed)
    
    obs_dim = base_env.obs_dim
    action_dim = cfg.get("env", {}).get("action_dim", 21)
    num_agents = base_env.agent_num

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

    training_cfg = cfg.get("training", {})
    use_ga = training_cfg.get("use_ga", False)
    ga_horizon = training_cfg.get("ga_horizon", 5)
    ga_params = cfg.get("heuristic", {}).get("ga", {})
    
    planner = None
    planners = None
    eval_planner = None
    
    n_rollout_threads = training_cfg.get("n_rollout_threads", 1)

    if n_rollout_threads > 1:
        def make_env_fn() -> Any:
            return build_environment(cfg, seed) 
        env = SubprocVecEnv(make_env_fn, n_rollout_threads)
        is_vec = True
    else:
        env = base_env
        is_vec = False

    if use_ga:
        if is_vec:
            planners = [
                HybridPlanner(env=env_i, horizon=ga_horizon, use_ga=True, ga_params=ga_params)
                for env_i in env.env_list
            ]
            eval_planner = planners[0]
        else:
            planner = HybridPlanner(env=env, horizon=ga_horizon, use_ga=True, ga_params=ga_params)
            eval_planner = planner
        logger.info("GA-based planner enabled for action refinement")

    episodes = training_cfg.get("episodes", 10)
    evaluate_every = training_cfg.get("evaluate_every", 0)
    early_stop = training_cfg.get("early_stop", False)
    n_warmup_evals = training_cfg.get("n_warmup_evaluations", 0)
    n_no_improve_thres = training_cfg.get("n_no_improvement_thres", 10)
    best_eval_reward: float | None = None
    no_improve_count = 0
    save_path = training_cfg.get("save_path", None)
    
    stop_training = False

    for ep in range(1, episodes + 1):
        ep_reward_sum = 0.0
        if is_vec:
            obs_batch = env.reset(train=True)
            done_batch = np.zeros((n_rollout_threads, num_agents), dtype=bool)
            while not np.all(done_batch):
                actions_batch: List[List[int]] = []
                log_probs_batch: List[List[float]] = []
                for i in range(n_rollout_threads):
                    if not np.all(done_batch[i]):
                        obs_list = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        acts, log_ps = agent.select_actions(obs_list)
                        if planners is not None:
                            acts = planners[i].refine_actions(acts)
                        actions_batch.append(acts)
                        log_probs_batch.append(log_ps)
                    else:
                        actions_batch.append([0] * num_agents)
                        log_probs_batch.append([0.0] * num_agents)
                
                next_obs_batch, rewards_batch, done_batch2, _ = env.step(actions_batch, one_hot=False)
                
                for i in range(n_rollout_threads):
                    if not np.all(done_batch[i]):
                        obs_i = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        next_obs_i = next_obs_batch[i].tolist() if isinstance(next_obs_batch[i], np.ndarray) else next_obs_batch[i]
                        rewards_i = rewards_batch[i].tolist() if isinstance(rewards_batch[i], np.ndarray) else rewards_batch[i]
                        done_i = done_batch2[i].tolist() if isinstance(done_batch2[i], np.ndarray) else done_batch2[i]
                        agent.store_transition(obs_i, actions_batch[i], log_probs_batch[i], rewards_i, next_obs_i, done_i)
                        ep_reward_sum += float(np.sum(rewards_i))
                obs_batch = next_obs_batch
                done_batch = done_batch2
        else:
            obs_list = env.reset(train=True)
            done_flags = [False] * num_agents
            while True:
                actions, log_probs = agent.select_actions(obs_list)
                refined_actions = actions
                if planner is not None:
                    refined_actions = planner.refine_actions(actions)
                next_obs_list, rewards, done, _ = env.step(refined_actions, one_hot=False)
                flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                agent.store_transition(obs_list, refined_actions, log_probs, flat_rewards, next_obs_list, done[0])
                ep_reward_sum += float(np.sum(flat_rewards))
                obs_list = next_obs_list
                if all(done):
                    break
        
        agent.update()
        logger.info(f"Episode {ep}/{episodes}: total reward = {ep_reward_sum:.2f}")

        if evaluate_every and ep % evaluate_every == 0:
            eval_env = env.env_list[0] if is_vec else env
            if hasattr(eval_env, "get_eval_num") and eval_env.get_eval_num() > 0:
                logger.info("Running evaluation episodes...")
                n_eval = eval_env.get_eval_num()
                eval_reward_sum = 0.0
                for _ in range(n_eval):
                    obs = eval_env.reset(train=False)
                    while True:
                        actions, _ = agent.select_actions(obs)
                        if eval_planner is not None:
                            actions = eval_planner.refine_actions(actions)
                        obs, rewards, done, _ = eval_env.step(actions, one_hot=False)
                        flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                        eval_reward_sum += float(np.sum(flat_rewards))
                        if all(done):
                            break
                
                bw_res = eval_env.get_eval_bw_res()
                if bw_res:
                    logger.info(f"Evaluation bullwhip per agent: {bw_res}")
                
                avg_eval_reward = eval_reward_sum / float(n_eval)
                logger.info(f"Average evaluation reward: {avg_eval_reward:.2f}")

                if early_stop:
                    if best_eval_reward is None or avg_eval_reward > best_eval_reward:
                        best_eval_reward = avg_eval_reward
                        no_improve_count = 0
                    else:
                        if ep // evaluate_every > n_warmup_evals:
                            no_improve_count += 1
                            logger.info(f"No improvement for {no_improve_count} evals (best={best_eval_reward:.2f})")
                            if no_improve_count >= n_no_improve_thres:
                                logger.info("Early stopping triggered.")
                                stop_training = True

        if save_path and (ep == episodes or stop_training):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "actor_state_dicts": [actor.state_dict() for actor in agent.actors],
                    "critic_state_dict": agent.critic_net.state_dict(),
                },
                save_path,
            )
            logger.info(f"Saved trained model to {save_path}")
        
        if stop_training:
            break
    
    logger.info(f"Training complete for seed {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train network inventory management")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    # Tham số --seed mới
    parser.add_argument("--seed", type=int, default=None, help="Random seed for training")
    args = parser.parse_args()
    main(args.config, args.seed)
