"""
happo_agent.py
===============

Implementation of a simple multi–agent policy gradient algorithm inspired
by Heterogeneous Agent Proximal Policy Optimisation (HAPPO).  This
version supports a team of agents with individual policy networks and a
centralised critic.  The training procedure collects full trajectories
from the environment, computes returns and advantages using
Generalised Advantage Estimation (GAE), and performs policy and value
updates using the PPO objective with clipping.

This implementation is intentionally lightweight: it avoids many
optimisations present in mature RL libraries in order to remain
accessible and easy to modify.  It should nonetheless serve as a
reasonable baseline for experimentation in multi–agent supply chain
control.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .policy_networks import MLPActor
from .centralized_critic import CentralizedCritic
from .replay_buffer import ReplayBuffer


class HAPPOAgent:
    """Multi–agent PPO with heterogeneous policies and a central critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Create actor networks for each agent
        self.actors: List[MLPActor] = []
        self.actor_optimisers: List[Adam] = []
        for _ in range(num_agents):
            actor = MLPActor(obs_dim, action_dim, hidden_dim).to(self.device)
            self.actors.append(actor)
            self.actor_optimisers.append(Adam(actor.parameters(), lr=actor_lr))
        # Centralised critic
        state_dim = num_agents * obs_dim
        self.critic = CentralizedCritic(state_dim, critic_hidden_dim)
        self.critic_net = self.critic.net.to(self.device)
        self.critic_optimizer = Adam(self.critic_net.parameters(), lr=critic_lr)
        # Experience buffer
        self.buffer = ReplayBuffer()

    def select_actions(self, obs_list: List[np.ndarray]) -> Tuple[List[int], List[torch.Tensor]]:
        """Given a list of observations, sample actions and return log probs.

        Args:
            obs_list: List of per–agent observations (numpy arrays).

        Returns:
            actions: List of integer actions chosen by each agent.
            log_probs: List of log probability tensors (one per agent).
        """
        actions: List[int] = []
        log_probs: List[torch.Tensor] = []
        for i, obs in enumerate(obs_list):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            act, log_prob = self.actors[i].get_action(obs_tensor)
            actions.append(act)
            log_probs.append(log_prob)
        return actions, log_probs

    def store_transition(
        self,
        obs: List[np.ndarray],
        actions: List[int],
        log_probs: List[torch.Tensor],
        rewards: List[float],
        next_obs: List[np.ndarray],
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            obs: List of observations for each agent at time t.
            actions: List of actions taken by each agent.
            log_probs: List of log probabilities under the old policy.
            rewards: List of rewards received by each agent.
            next_obs: List of observations at time t+1.
            done: Episode termination flag (same for all agents).
        """
        # Detach log_probs to avoid retaining computation graph.  Convert each
        # log probability to a Python float to ensure consistent tensor shapes
        # when converting the replay buffer to tensors.  Storing NumPy 0‑d
        # arrays here can lead to unsized objects that break torch.tensor().
        log_probs_detached = [float(lp.detach().cpu().item()) for lp in log_probs]
        transition = {
            "obs": obs,
            "actions": actions,
            "log_probs": log_probs_detached,
            "rewards": rewards,
            "next_obs": next_obs,
            "done": done,
        }
        self.buffer.add(transition)

    def update(self, batch_size: Optional[int] = None) -> None:
        """Update policy and critic using the collected trajectories.

        This method processes the entire buffer as a single batch.  It
        computes returns and advantages via GAE and performs one gradient
        update for each actor and the critic.
        """
        if len(self.buffer) == 0:
            return
        # Convert buffer to dict of lists
        data = self.buffer.as_dict()
        T = len(self.buffer)
        # Prepare tensors
        # obs: shape (T, num_agents, obs_dim)
        obs_tensor = torch.tensor(data["obs"], dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(data["actions"], dtype=torch.int64, device=self.device)
        old_log_probs_tensor = torch.tensor(data["log_probs"], dtype=torch.float32, device=self.device)
        rewards_tensor = torch.tensor(data["rewards"], dtype=torch.float32, device=self.device)
        next_obs_tensor = torch.tensor(data["next_obs"], dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(data["done"], dtype=torch.float32, device=self.device)
        # Flatten observations to global state
        states = obs_tensor.view(T, -1)
        next_states = next_obs_tensor.view(T, -1)
        # Compute state values
        with torch.no_grad():
            values = self.critic_net(states).squeeze(-1)  # (T,)
            next_values = self.critic_net(next_states).squeeze(-1)
        # Compute global rewards as sum across agents
        global_rewards = rewards_tensor.sum(dim=1)  # (T,)
        # Compute GAE advantages and returns
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones_tensor[t]
            delta = global_rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # Update critic (value function) using MSE loss
        self.critic_optimizer.zero_grad()
        value_preds = self.critic_net(states).squeeze(-1)
        critic_loss = torch.mean((returns - value_preds) ** 2)
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update each actor independently
        for i in range(self.num_agents):
            actor = self.actors[i]
            actor_opt = self.actor_optimisers[i]
            # Select data for agent i
            obs_i = obs_tensor[:, i, :]
            actions_i = actions_tensor[:, i]
            old_log_probs_i = old_log_probs_tensor[:, i]
            # Evaluate new log probs and entropy
            new_log_probs, entropy = actor.evaluate_actions(obs_i, actions_i)
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs_i.to(self.device))
            # Compute surrogate loss
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2)) - self.entropy_coef * torch.mean(entropy)
            # Update actor
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
        # Clear buffer after update
        self.buffer.clear()
