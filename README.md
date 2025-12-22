# Inventory Management RL Lot

`inventory_management_RL_Lot` is a research framework for studying
multi‑echelon inventory management under uncertainty using a combination of
multi‑agent deep reinforcement learning (MADRL) and classic lot‑sizing
heuristics.  The goal of the project is to reproduce, in a self‑contained
way, many of the ideas from Xiaotian Liu et al.'s work on multi‑agent deep
reinforcement learning for multi‑echelon inventory management while also
providing hooks for hybrid approaches that incorporate heuristic lot sizing
(e.g. genetic algorithms or DALSA).  This repository is structured in a
modular fashion so that each component (environment, agent, heuristics,
meta‑learning, etc.) can be developed and tested independently.

The top level directory includes the following:

* **README.md** – This file.  Describes the purpose of the repository,
  project layout and high‑level usage instructions.
* **requirements.txt** – A list of Python packages required to run
  the experiments.  Install them with `pip install -r requirements.txt`.
* **test_data/** – Data used for training and evaluation.  This
  directory contains subdirectories such as
  `test_demand_merton/` and `test_demand_stationary/`, each holding
  dozens of text files where every file represents one demand
  sequence (one integer per line).  A YAML configuration file
  (`config.yaml`) describes environment parameters used by the data
  generator.  You can modify or extend these demand sequences to
  suit your own experiments.  In the original Liu et al. repository
  there are many evaluation episodes of length 200; you should
  populate `test_demand_merton/` and `test_demand_stationary/` with
  similar numbers of files if you want to replicate their scale.
* **envs/** – Definitions of the supply chain simulation environments.
  The core implementation lives in `serial_env.py` for serial
  multi‑echelon chains and `network_env.py` for more general network
  structures.  A base class in `base_env.py` defines the common
  interface and shared utilities.  Reward functions are factored into
  `reward_functions.py` for clarity and re‑use.
* **agents/** – Implementation of the multi‑agent RL algorithms.  The
  default is a simplified HAPPO trainer (`happo_agent.py`) built on
  top of PyTorch.  Policy and value network definitions live in
  `policy_networks.py`, a simple rollout buffer in `replay_buffer.py`
  and a centralized critic helper in `centralized_critic.py`.
* **lot_sizing/** – Heuristic modules for solving lot‑sizing subproblems.
  `ga_lotsizing.py` contains an adaptive genetic algorithm for finding
  good order quantities over short horizons, `dalsa_module.py` contains
  a basic implementation of the DALSA heuristic from Nguyen Van Hop
  and colleagues, and `hybrid_planner.py` exposes a simple interface
  for combining RL decisions with heuristics.
* **meta/** – Skeleton implementations of meta‑learning techniques.  For
  example, `hierarchical_metalearning.py` outlines how to structure
  hierarchical meta‑learners on top of the RL agents, while
  `maml.py` contains a toy Model‑Agnostic Meta‑Learning (MAML)
  algorithm.  These are optional and can be developed independently.
* **utils/** – Helper modules.  Includes a simple logger (`logger.py`),
  a data loader (`data_loader.py`) for reading demand sequences and
  configuration files, metrics utilities (`metrics.py`) for measuring
  bullwhip effect, service level and cost, and plotting helpers
  (`plot_tools.py`) for visualizing training progress and inventory
  trajectories.
* **configs/** – YAML configuration files used to specify hyper‑
  parameters for different experiments.  There are separate
  configurations for training on serial and network environments and
  for configuring the heuristic solvers.
* **train_main.py** – The entry point for training experiments.  Reads
  a configuration, constructs the environment and agents, optionally
  instantiates heuristic modules, runs the training loop and saves
  results.
* **evaluate_main.py** – Runs a trained policy on evaluation demand
  sequences and reports metrics such as total cost, bullwhip effect and
  service level.  Useful for benchmarking different approaches.

## Getting started

1. **Install dependencies**
   
   Ensure you have Python 3.8 or later installed.  Create a virtual
   environment if desired and install requirements:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Prepare data**

   This repository includes a small set of example demand sequences
   under `test_data/test_demand_merton/` and `test_data/test_demand_stationary/` as
   well as a configuration file in `test_data/config.yaml` that describes
   environment parameters (number of levels, lead time, costs, etc.).
   The data loader will automatically generate random training demand
   using the Merton jump diffusion or Poisson processes described in
   the original paper.  Feel free to extend the evaluation folders
   with your own sequences or increase the number of episodes.  For
   large‑scale experiments similar to the original Liu et al. repository
   you should create dozens of evaluation sequences of length 200.

3. **Train a policy**

   To train on a serial supply chain with three echelons using the
   default configuration, run:

   ```bash
   python train_main.py --config configs/train_serial.yaml
   ```

   Training progress will be logged to the console.  The trained
   models and run metadata will be saved under `results/`.

4. **Evaluate a policy**

   After training, evaluate the policy on the test sequences:

   ```bash
   python evaluate_main.py --config configs/train_serial.yaml --model_path results/serial_policy.pt
   ```

   This script reports the average total cost per echelon, bullwhip
   measurements and service levels over the evaluation set.

## Extending the framework

* **Changing environment parameters** – Modify `test_data/config.yaml`
  to adjust the number of levels, lead times, holding/backorder/fixed
  costs, price discount schemes, or random shipping loss.  The
  environment will pick up these values at runtime.
* **Adding network structures** – `network_env.py` demonstrates how to
  simulate a network supply chain (e.g. two retailers feeding into
  a central warehouse and multiple factories).  You can adjust the
  adjacency list and agent ordering to match any directed acyclic
  network.
* **Incorporating heuristics** – See `lot_sizing/ga_lotsizing.py` and
  `lot_sizing/hybrid_planner.py` for how to refine RL actions with
  short‑horizon heuristics.  You can plug in alternative lot‑sizing
  algorithms by implementing a similar interface.
* **Meta‑learning experiments** – The `meta/` directory contains
  skeletons for hierarchical meta‑learning and MAML.  These are
  intentionally light weight and meant as starting points for
  experimentation.

We encourage you to read through the code, modify it for your own
projects and contribute improvements.  Good luck with your research!
