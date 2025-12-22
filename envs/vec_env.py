"""
vec_env.py
===========

This module implements simple vectorized wrappers for the inventory
management environments.  The primary motivation is to enable
multi‑environment rollout during training, similar to the
`SubprocVecEnv` used in the original Liu‑MARL repository.  The
wrappers here create multiple independent instances of a base
environment and provide batched ``reset`` and ``step`` methods that
return stacked observations, rewards and done flags.

Two classes are provided:

``MultiDiscrete``
    A helper space used when aggregating multiple discrete action
    spaces into one.  It mirrors the interface from OpenAI Gym and is
    included for completeness although our environments typically
    expose a single discrete action space.

``SubprocVecEnv``
    A lightweight vectorised environment that holds ``n`` copies of
    the underlying environment.  It does **not** launch separate
    processes; instead it runs each environment sequentially when
    ``step`` is called.  This design keeps the implementation simple
    while still allowing the agent to generate multiple episodes in
    parallel from a logical perspective.

``DummyVecEnv``
    A thin wrapper around a single environment used for evaluation.
    It delegates evaluation bookkeeping (bullwhip metrics and number
    of evaluation sequences) to the underlying environment.

These wrappers are intentionally minimal and do not implement the
full multiprocessing and shared memory behaviour of the original
``SubprocVecEnv``.  They provide the required interface for
multi‑agent RL algorithms that expect a batched environment.  If you
need true parallel execution, consider integrating with libraries
such as ``gymnasium.vector``.
"""

from __future__ import annotations

from typing import List, Tuple, Any

import numpy as np

from gym import spaces


class MultiDiscrete(spaces.Space):
    """A simple multi‑discrete action space.

    It accepts a list of (min, max) pairs and represents the Cartesian
    product of the corresponding discrete spaces.  This class mirrors
    the Gym implementation and is used when an agent has multiple
    discrete sub‑actions.
    """

    def __init__(self, array_of_param_array: List[Tuple[int, int]]):
        # Lower and upper bounds for each discrete component
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        # Number of discrete components
        self.num_discrete_space = self.low.shape[0]
        # Total number of actions (not used here)
        self.n = np.sum(self.high - self.low + 1)
        super().__init__(shape=(self.num_discrete_space,), dtype=np.int64)

    def sample(self) -> List[int]:
        # Uniformly sample each discrete component
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor((self.high - self.low + 1.) * random_array + self.low)]

    def contains(self, x: Any) -> bool:
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self) -> Tuple[int, ...]:  # type: ignore[override]
        return (self.num_discrete_space,)

    def __repr__(self) -> str:
        return f"MultiDiscrete{self.num_discrete_space}"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MultiDiscrete) and np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv:
    """A simple vectorised environment running ``n`` copies sequentially.

    Given a constructor function and configuration for a base
    environment, this wrapper instantiates multiple copies and
    orchestrates batched resets and steps.  Each call to ``step``
    iterates over all environment instances, passing in the
    corresponding action for each.  Observations, rewards and done
    flags are stacked along the first dimension.

    Parameters
    ----------
    env_fn : callable
        A function with no arguments that returns a new instance of the
        environment when called.
    n_envs : int
        Number of parallel environment instances.
    """

    def __init__(self, env_fn, n_envs: int):
        self.env_list = [env_fn() for _ in range(n_envs)]
        self.num_envs = n_envs
        # Assume all envs have the same agent count and observation/action dims
        env0 = self.env_list[0]
        self.num_agent = env0.agent_num
        self.signal_obs_dim = env0.obs_dim
        self.signal_action_dim = env0.action_dim
        # Build per‑agent observation and action spaces
        self.observation_space: List[spaces.Space] = []
        self.action_space: List[spaces.Space] = []
        share_obs_dim = 0
        for _ in range(self.num_agent):
            # Single discrete action space per agent
            self.action_space.append(spaces.Discrete(self.signal_action_dim))
            # Observation space for each agent
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            )
            share_obs_dim += self.signal_obs_dim
        # Shared observation space (concatenation of all agents)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def reset(self, train: bool = True):
        """Reset all environments.

        If ``train`` is True, use the training mode of the environment;
        otherwise use evaluation mode.  Returns a stacked array of
        observations with shape ``(n_envs, num_agents, obs_dim)``.
        """
        obs_batch = []
        for env in self.env_list:
            if train:
                obs = env.reset(train=True)
            else:
                obs = env.reset(train=False)
            obs_batch.append(obs)
        return np.array(obs_batch)

    def step(self, actions_batch: List[List[int]], one_hot: bool = False):
        """Step all environments with a batch of actions.

        ``actions_batch`` should be a list of length ``n_envs``, where
        each element is a list of actions (one per agent) for the
        corresponding environment.

        Returns
        -------
        next_obs_batch : ndarray
            Array of next observations with shape ``(n_envs, num_agents, obs_dim)``.
        rewards_batch : ndarray
            Array of rewards with shape ``(n_envs, num_agents)``.
        dones_batch : ndarray
            Boolean array of done flags with shape ``(n_envs, num_agents)``.
        infos : list
            List of info dicts (one per environment).  Not used here.
        """
        next_obs_batch: List[Any] = []
        rewards_batch: List[Any] = []
        dones_batch: List[Any] = []
        infos: List[Any] = []
        for env, acts in zip(self.env_list, actions_batch):
            next_obs, rewards, done, info = env.step(acts, one_hot=one_hot)
            # Flatten reward list to plain numbers
            flat_rew = [r[0] if isinstance(r, list) else r for r in rewards]
            next_obs_batch.append(next_obs)
            rewards_batch.append(flat_rew)
            dones_batch.append(done)
            infos.append(info)
        return np.array(next_obs_batch), np.array(rewards_batch), np.array(dones_batch), infos

    def get_eval_bw_res(self) -> List[float]:
        """Retrieve bullwhip results from the first environment.

        Only the first environment accumulates evaluation statistics.  All
        other environments ignore evaluation metrics.  This mirrors the
        behaviour of ``DummyVecEnv`` from the original code where
        evaluation mode uses a single environment.
        """
        return self.env_list[0].get_eval_bw_res() if hasattr(self.env_list[0], "get_eval_bw_res") else []

    def get_eval_num(self) -> int:
        """Retrieve number of evaluation sequences from the first environment."""
        return self.env_list[0].get_eval_num() if hasattr(self.env_list[0], "get_eval_num") else 0


class DummyVecEnv:
    """A dummy vectorised environment wrapping a single environment.

    This class provides the same interface as ``SubprocVecEnv`` but
    contains only one environment instance.  It is primarily used for
    evaluation where batching is not required.
    """

    def __init__(self, env_fn):
        self.env = env_fn()
        self.num_envs = 1
        self.num_agent = self.env.agent_num
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim
        # Per‑agent observation and action spaces
        self.observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]
        self.action_space = [spaces.Discrete(self.signal_action_dim) for _ in range(self.num_agent)]
        share_obs_dim = self.signal_obs_dim * self.num_agent
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.num_agent)
        ]

    def reset(self, train: bool = False):
        if train:
            obs = self.env.reset(train=True)
        else:
            obs = self.env.reset(train=False)
        return np.array([obs])

    def step(self, actions_batch: List[List[int]], one_hot: bool = False):
        # ``actions_batch`` is expected to be a list of length 1 containing a list of actions
        next_obs, rewards, done, info = self.env.step(actions_batch[0], one_hot=one_hot)
        flat_rew = [r[0] if isinstance(r, list) else r for r in rewards]
        return np.array([next_obs]), np.array([flat_rew]), np.array([done]), [info]

    def get_eval_bw_res(self) -> List[float]:
        return self.env.get_eval_bw_res() if hasattr(self.env, "get_eval_bw_res") else []

    def get_eval_num(self) -> int:
        return self.env.get_eval_num() if hasattr(self.env, "get_eval_num") else 0
