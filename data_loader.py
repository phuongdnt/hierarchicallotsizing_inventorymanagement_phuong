"""
data_loader.py
=================

Utility functions for loading evaluation and training demand data for the
inventory management RL environment.  These functions are intentionally
simple so that the rest of the project can focus on reinforcement learning
and supply chain logic rather than file I/O.

The evaluation data is expected to live under a directory such as
``test_data/test_demand_merton/`` where each file contains one demand
sequence.  The loader will read each file line by line and convert the
lines to integers.  When generating training data on the fly, a simple
stochastic process is used to simulate customer demand.

Note: In a real application you may wish to replace the random demand
generation with more realistic time‐series models or import actual
historical data.  This file provides a minimal starting point.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np


def load_eval_data(eval_dir: str) -> Tuple[int, List[List[int]]]:
    """Load evaluation demand sequences from a directory.

    Each file in ``eval_dir`` is treated as one episode of demand.  The
    file should contain one integer per line representing demand in that
    period.  Files are sorted alphabetically to provide a deterministic
    order.

    Args:
        eval_dir: Directory containing demand files (e.g. ``test_data/test_demand_merton``).

    Returns:
        A tuple ``(n, data)`` where ``n`` is the number of demand
        sequences and ``data`` is a list of integer lists.
    """
    eval_path = Path(eval_dir)
    if not eval_path.exists() or not eval_path.is_dir():
        raise FileNotFoundError(f"Evaluation directory {eval_dir} does not exist")
    demand_sequences: List[List[int]] = []
    for demand_file in sorted(eval_path.iterdir()):
        if not demand_file.is_file():
            continue
        with demand_file.open("r") as f:
            seq = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seq.append(int(line))
                    except ValueError:
                        raise ValueError(
                            f"Invalid demand value in {demand_file}: {line}")
            if seq:
                demand_sequences.append(seq)
    return len(demand_sequences), demand_sequences


def generate_training_demand(
    episode_len: int,
    max_demand: int,
    distribution: str = "uniform",
    seed: int | None = None,
) -> List[int]:
    """Generate a random sequence of customer demand for training.

    By default this uses a simple uniform distribution over the range
    [0, max_demand].  The function can easily be extended to support
    other distributions, such as Poisson or a mixture model, by adding
    additional branches on the ``distribution`` parameter.

    Args:
        episode_len: Length of the episode (number of periods).
        max_demand: Maximum possible demand value (inclusive).
        distribution: Type of random distribution to use (default ``"uniform"``).
        seed: Optional random seed for reproducibility.

    Returns:
        A list of integers representing the demand at each time period.
    """
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.integers(low=0, high=max_demand + 1, size=episode_len).tolist()
    elif distribution == "poisson":
        # Use the max_demand as the lambda parameter.  The values are
        # truncated at ``max_demand`` to avoid extremely large orders.
        samples = rng.poisson(lam=max_demand, size=episode_len)
        return np.clip(samples, 0, max_demand).tolist()
    elif distribution == "merton":
        # Merton jump diffusion model: mixture of normal and Poisson jumps.
        # This is an illustrative implementation; feel free to tweak
        # parameters as needed to match your desired demand variability.
        mu, sigma, jump_lambda, jump_mu, jump_sigma = 0.5 * max_demand, 0.2 * max_demand, 0.05, 1.0, 0.5
        demand = []
        for _ in range(episode_len):
            base = rng.normal(loc=mu, scale=sigma)
            jumps = rng.poisson(lam=jump_lambda)
            for _ in range(jumps):
                base += rng.normal(loc=jump_mu * max_demand, scale=jump_sigma * max_demand)
            demand.append(int(max(0, min(base, max_demand))))
        return demand
    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")