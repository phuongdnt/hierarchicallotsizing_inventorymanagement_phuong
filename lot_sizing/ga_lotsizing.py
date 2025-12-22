"""
ga_lotsizing.py
================

Genetic algorithm for the lot–sizing problem with fixed order costs and
linear holding/backorder penalties.  This module provides a simple
implementation of a GA that searches for a near–optimal order plan
over a short horizon given a deterministic demand forecast.  It is
intended as a plug–in heuristic for use alongside learned RL policies,
allowing an agent to refine its ordering quantity to better satisfy
lot–sizing constraints.

The GA operates on individuals represented as integer arrays of order
quantities for each period in the horizon.  Fitness is defined as the
negative total cost over the horizon.  Crossover and mutation are
implemented as single–point crossover and uniform random mutation.

Example usage::

    from inventory_management_RL_Lot.lot_sizing.ga_lotsizing import optimise_order
    best_order = optimise_order(
        current_inventory=10,
        demand_forecast=[5,7,4,9],
        holding_cost=1.0,
        backlog_cost=2.0,
        fixed_cost=5.0,
        max_order=20,
        horizon=4,
        pop_size=30,
        generations=50,
        mutation_rate=0.1,
    )
    # Returns the quantity to order now (an integer)

"""

from __future__ import annotations

import random
from typing import List


def evaluate_plan(
    plan: List[int],
    init_inventory: int,
    demands: List[int],
    holding_cost: float,
    backlog_cost: float,
    fixed_cost: float,
) -> float:
    """Compute total cost for a candidate order plan over the horizon.

    Each element of ``plan`` corresponds to the order quantity in that
    period (period 0, 1, ..., H-1).  Inventory and backlog evolve
    deterministically given the demand sequence ``demands``.  Only
    fixed order cost and linear holding/backorder costs are considered.

    Returns the total cost (lower is better).  Note: this function does
    not include variable cost for the goods themselves, as that does not
    affect the optimal schedule when prices are constant.
    """
    inventory = init_inventory
    backlog = 0
    cost = 0.0
    for t, demand in enumerate(demands):
        # Receive order placed at time t (assuming zero lead time for planning)
        received = plan[t]
        # Effective demand
        effective = demand + backlog
        unmet = effective - (inventory + received)
        if unmet > 0:
            backlog = unmet
            inventory = 0
        else:
            backlog = 0
            inventory = -unmet
        # Accumulate cost
        cost += inventory * holding_cost + backlog * backlog_cost
        if received > 0:
            cost += fixed_cost
    return cost


def optimise_order(
    current_inventory: int,
    demand_forecast: List[int],
    holding_cost: float,
    backlog_cost: float,
    fixed_cost: float,
    max_order: int,
    horizon: int,
    pop_size: int = 20,
    generations: int = 30,
    mutation_rate: float = 0.1,
) -> int:
    """Optimise the order quantity for period 0 using a genetic algorithm.

    This function searches for a near–optimal plan of length ``horizon``
    that minimises total cost.  It returns the order quantity in the
    first period of the best plan found.

    Args:
        current_inventory: Inventory available at the start of period 0.
        demand_forecast: List of deterministic demand values for each
            period in the horizon.  Length must equal ``horizon``.
        holding_cost: Per-unit holding cost.
        backlog_cost: Per-unit backlog cost.
        fixed_cost: Fixed cost per order (applied whenever order > 0).
        max_order: Upper bound on order quantity considered in search.
        horizon: Number of periods to optimise over (should match length
            of ``demand_forecast``).
        pop_size: Population size for the GA.
        generations: Number of generations to run.
        mutation_rate: Probability of mutating an element in the plan.

    Returns:
        The order quantity for the first period of the best plan found.
    """
    if horizon != len(demand_forecast):
        raise ValueError("horizon must equal length of demand_forecast")
    # Initialise population of random plans
    def random_plan() -> List[int]:
        return [random.randint(0, max_order) for _ in range(horizon)]
    population: List[List[int]] = [random_plan() for _ in range(pop_size)]
    # Evaluate fitness (lower cost is better)
    def fitness(plan: List[int]) -> float:
        return -evaluate_plan(plan, current_inventory, demand_forecast, holding_cost, backlog_cost, fixed_cost)
    # Main GA loop
    for _ in range(generations):
        # Compute fitnesses and probabilities for selection
        fitnesses = [fitness(p) for p in population]
        total_fit = sum(fitnesses)
        if total_fit == 0:
            probs = [1.0 / pop_size] * pop_size
        else:
            probs = [f / total_fit for f in fitnesses]
        # Create next population
        new_pop: List[List[int]] = []
        # Elitism: carry forward the best individual
        best_idx = max(range(pop_size), key=lambda i: fitnesses[i])
        new_pop.append(population[best_idx][:])
        while len(new_pop) < pop_size:
            # Select parents using roulette selection
            parent1 = population[random.choices(range(pop_size), weights=probs)[0]]
            parent2 = population[random.choices(range(pop_size), weights=probs)[0]]
            # Crossover: single point
            cx_point = random.randint(1, horizon - 1)
            child = parent1[:cx_point] + parent2[cx_point:]
            # Mutation
            for i in range(horizon):
                if random.random() < mutation_rate:
                    child[i] = random.randint(0, max_order)
            new_pop.append(child)
        population = new_pop
    # Select best plan from final population
    best_plan = max(population, key=fitness)
    return best_plan[0]
