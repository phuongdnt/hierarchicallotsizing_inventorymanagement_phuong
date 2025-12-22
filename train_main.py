"""
train_main.py
==============

Entry point for training reinforcement learning agents on the
inventory management environments.  This script parses a YAML
configuration file, constructs the environment, agent and optional
lot–sizing planner, and then runs a simple episodic training loop.

Usage::

    python train_main.py --config configs/train_serial.yaml

The script logs progress to the console and optionally saves the
trained model parameters.  Evaluation during training (on separate
evaluation demand sequences) can be enabled by specifying an
``evaluate_every`` interval in the configuration.  The evaluation
routine uses the environment's built-in evaluation mode to compute
bullwhip metrics.
"""

from __future__ import annotations

import argparse
import os
import yaml
from typing import Any, Dict, List

import numpy as np
import torch

from .utils.logger import setup_logger
from .agents.happo_agent import HAPPOAgent
from .envs.serial_env import SerialInventoryEnv
from .envs.network_env import NetworkInventoryEnv
# Import vectorised environment wrappers for multi‑threaded training
from .envs.vec_env import SubprocVecEnv, DummyVecEnv
from .lot_sizing.hybrid_planner import HybridPlanner
from .utils.metrics import compute_episode_costs


def parse_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_environment(cfg: Dict[str, Any]):
    """Construct an inventory management environment from configuration.

    Parameters
    ----------
    cfg : dict
        Full configuration loaded from a YAML file.  Only the ``env``
        section is inspected here.

    Returns
    -------
    env : BaseInventoryEnv
        A new environment instance.  The type (serial vs. network) and
        all associated parameters are inferred from ``cfg['env']``.

    Notes
    -----
    This helper does *not* perform any vectorisation; that is done
    separately in the training loop when ``n_rollout_threads > 1``.
    """
    env_cfg = cfg.get("env", {})
    env_type = env_cfg.get("env_type", "serial")
    if env_type == "serial":
        env = SerialInventoryEnv(**env_cfg)
    elif env_type == "network":
        # Extract children and parents separately as they are strings in YAML
        children = {int(k): [int(x) for x in v] for k, v in env_cfg.get("children", {}).items()}
        parents = {int(k): (int(v) if v is not None else None) for k, v in env_cfg.get("parents", {}).items()}
        # Remove keys not needed for NetworkInventoryEnv
        other_params = {k: v for k, v in env_cfg.items() if k not in ["env_type", "children", "parents"]}
        env = NetworkInventoryEnv(children=children, parents=parents, **other_params)
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")
    return env


def main(config_path: str) -> None:
    cfg = parse_config(config_path)
    logger = setup_logger()
    # Build a single base environment; vectorisation is handled below
    base_env = build_environment(cfg)
    # Unpack environment parameters
    obs_dim = base_env.obs_dim
    action_dim = cfg.get("env", {}).get("action_dim", 21)
    num_agents = base_env.agent_num
    # Construct the agent (shared across all environment copies)
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
    # Planner for lot‑sizing (optional)
    training_cfg = cfg.get("training", {})
    use_ga = training_cfg.get("use_ga", False)
    ga_horizon = training_cfg.get("ga_horizon", 5)
    ga_params = cfg.get("heuristic", {}).get("ga", {})
    planner = None
    # Placeholder for vectorised environment (training) and evaluation environment
    n_rollout_threads = training_cfg.get("n_rollout_threads", 1)
    n_eval_rollout_threads = training_cfg.get("n_eval_rollout_threads", 1)
    if use_ga:
        # Planner is bound to the base environment; if using vectorised env the
        # planner will refine actions per environment instance on the fly.
        planner = HybridPlanner(env=base_env, horizon=ga_horizon, use_ga=True, ga_params=ga_params)
        logger.info("GA-based planner enabled for action refinement")
    # Create vectorised training environment if n_rollout_threads > 1
    if n_rollout_threads > 1:
        def make_env_fn() -> Any:
            # Closure capturing cfg to construct identical environments
            return build_environment(cfg)
        env = SubprocVecEnv(make_env_fn, n_rollout_threads)
        is_vec = True
    else:
        env = base_env
        is_vec = False
    # Determine number of episodes and evaluation settings
    episodes = training_cfg.get("episodes", 10)
    evaluate_every = training_cfg.get("evaluate_every", 0)
    # Early stopping parameters
    early_stop = training_cfg.get("early_stop", False)
    n_warmup_evals = training_cfg.get("n_warmup_evaluations", 0)
    n_no_improve_thres = training_cfg.get("n_no_improvement_thres", 10)
    best_eval_reward: float | None = None
    no_improve_count = 0
    # Save path for final model
    save_path = training_cfg.get("save_path", None)
    reward_history: List[float] = []
    # Flag to signal early termination
    stop_training = False
    # Main training loop
    for ep in range(1, episodes + 1):
        ep_reward_sum = 0.0
        if is_vec:
            # Vectorised rollout: reset all env copies
            obs_batch = env.reset(train=True)
            # Maintain done flags per environment and per agent
            done_batch = np.zeros((n_rollout_threads, num_agents), dtype=bool)
            # Loop until all env copies are done
            while not np.all(done_batch):
                actions_batch: List[List[int]] = []
                log_probs_batch: List[List[float]] = []
                # Compute actions for each env copy
                for i in range(n_rollout_threads):
                    if not np.all(done_batch[i]):
                        obs_list = obs_batch[i].tolist() if isinstance(obs_batch[i], np.ndarray) else obs_batch[i]
                        acts, log_ps = agent.select_actions(obs_list)
                        # Refine actions using planner if enabled
                        if planner is not None:
                            acts = planner.refine_actions(acts)
                        actions_batch.append(acts)
                        log_probs_batch.append(log_ps)
                    else:
                        # Dummy actions for completed env; values won't be used
                        actions_batch.append([0] * num_agents)
                        log_probs_batch.append([0.0] * num_agents)
                # Step all environments
                next_obs_batch, rewards_batch, done_batch2, _ = env.step(actions_batch, one_hot=False)
                # Store transitions for each environment separately
                for i in range(n_rollout_threads):
                    # Only store transitions for unfinished environments
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
            # Single environment rollout
            obs_list = env.reset(train=True)
            done_flags = [False] * num_agents
            while True:
                actions, log_probs = agent.select_actions(obs_list)
                refined_actions = actions
                if planner is not None:
                    refined_actions = planner.refine_actions(actions)
                next_obs_list, rewards, done, _ = env.step(refined_actions, one_hot=False)
                # Convert rewards from [[r], [r], ...] to flat list
                flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                agent.store_transition(obs_list, refined_actions, log_probs, flat_rewards, next_obs_list, done[0])
                ep_reward_sum += float(np.sum(flat_rewards))
                obs_list = next_obs_list
                if all(done):
                    break
        # Update agent after each full batch of episodes
        agent.update()
        reward_history.append(ep_reward_sum)
        logger.info(f"Episode {ep}/{episodes}: total reward = {ep_reward_sum:.2f}")
        # Periodic evaluation and early stopping
        if evaluate_every and ep % evaluate_every == 0 and hasattr(base_env, "get_eval_num") and base_env.get_eval_num() > 0:
            # Run evaluation on a single environment (the first one) to compute reward and bullwhip
            logger.info("Running evaluation episodes...")
            # Determine which environment to use for evaluation: if vectorised, use the first env in the list
            eval_env = env.env_list[0] if is_vec else env
            n_eval = eval_env.get_eval_num()
            eval_reward_sum = 0.0
            for _ in range(n_eval):
                obs = eval_env.reset(train=False)
                while True:
                    actions, _ = agent.select_actions(obs)
                    if planner is not None:
                        actions = planner.refine_actions(actions)
                    obs, rewards, done, _ = eval_env.step(actions, one_hot=False)
                    # Sum reward across agents
                    flat_rewards = [r[0] if isinstance(r, list) else r for r in rewards]
                    eval_reward_sum += float(np.sum(flat_rewards))
                    if all(done):
                        break
            # Retrieve aggregated bullwhip results
            bw_res = eval_env.get_eval_bw_res()
            if bw_res:
                logger.info(f"Evaluation bullwhip per agent: {bw_res}")
            # Compute average reward per episode
            avg_eval_reward = eval_reward_sum / float(n_eval)
            logger.info(f"Average evaluation reward: {avg_eval_reward:.2f}")
            # Early stopping logic: update best reward and check for no improvement
            if early_stop:
                if best_eval_reward is None or avg_eval_reward > best_eval_reward:
                    best_eval_reward = avg_eval_reward
                    no_improve_count = 0
                else:
                    # Only count after warm‑up evaluations
                    if ep // evaluate_every > n_warmup_evals:
                        no_improve_count += 1
                        logger.info(
                            f"No improvement in evaluation reward for {no_improve_count} evaluations (best={best_eval_reward:.2f})"
                        )
                        if no_improve_count >= n_no_improve_thres:
                            logger.info(
                                "Early stopping triggered: no improvement threshold reached. Training will end early."
                            )
                            # Set stop flag and exit both loops
                            stop_training = True
                            break
        # Save model at the end or on early stop
        if save_path and (ep == episodes or (early_stop and stop_training)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "actor_state_dicts": [actor.state_dict() for actor in agent.actors],
                    "critic_state_dict": agent.critic_net.state_dict(),
                },
                save_path,
            )
            logger.info(f"Saved trained model to {save_path}")
        # Break outer loop if early stopping was triggered
        if stop_training:
            break
    logger.info("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train inventory management RL agent")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    main(args.config)
