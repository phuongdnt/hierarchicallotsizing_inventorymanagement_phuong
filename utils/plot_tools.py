"""
plot_tools.py
==============

Collection of helper functions for visualising results from the
inventory management RL experiments.  Uses Matplotlib to generate
plots of inventory levels, order quantities, reward curves, and
training progress.  Functions in this module follow the guidelines
described in the project README: each chart is plotted on its own
axes and colours are left to Matplotlib defaults.

Note: These functions are optional and may require additional
dependencies (e.g. Matplotlib).  They are not used during training by
default but are provided for post-hoc analysis.
"""

from __future__ import annotations

from typing import List, Iterable
import matplotlib.pyplot as plt


def plot_inventory_levels(inventory_history: List[List[int]], title: str = "Inventory Levels") -> None:
    """Plot inventory levels for each agent over an episode.

    Args:
        inventory_history: List of inventory states over time; each
            element is a list of inventory levels for each agent.
        title: Title for the plot.
    """
    num_agents = len(inventory_history[0])
    time_steps = range(len(inventory_history))
    for i in range(num_agents):
        series = [inv[i] for inv in inventory_history]
        plt.plot(time_steps, series, label=f"Agent {i}")
    plt.xlabel("Time step")
    plt.ylabel("Inventory level")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_order_quantities(order_history: List[List[int]], title: str = "Order Quantities") -> None:
    """Plot order quantities for each agent over time."""
    num_agents = len(order_history)
    time_steps = range(len(order_history[0]))
    for i, hist in enumerate(order_history):
        plt.plot(time_steps, hist, label=f"Agent {i}")
    plt.xlabel("Time step")
    plt.ylabel("Order quantity")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_reward_curve(reward_sums: Iterable[float], title: str = "Episode Reward") -> None:
    """Plot the sum of rewards per episode across training."""
    plt.plot(list(reward_sums))
    plt.xlabel("Episode")
    plt.ylabel("Total reward (sum across agents)")
    plt.title(title)
    plt.show()
