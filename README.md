# BPSO Optimizer

A hybrid optimization algorithm combining Bayesian Optimization and Particle Swarm Optimization (BPSO).

## Features

- Bayesian Optimization for global search
- Particle Swarm Optimization for local search
- Region management mechanism
- Candidate point management

## Project Structure
BPSO-Optimizer/
├── BPSO_optimizer.py # Main optimizer class
├── Bayesian.py # Bayesian optimization implementation
├── Particle.py # Particle class for PSO
├── Swarm.py # Swarm class for PSO
├── utils.py # Utility functions
└── verify_BPSO_localoptimum.py # Test script

## Usage
python
from BPSO_optimizer import BPSOOptimizer

##Configure parameters
config = {
"n_dimensions": 2,
"xbounds": [-5.12, 5.12],
"ybounds": [-5.12, 5.12],
# ... other parameters
}

##Create and run optimizer
optimizer = BPSOOptimizer(config)
best_position, best_cost, local_records = optimizer.optimize()

## Requirements
- numpy
- scipy
- scikit-learn
- matplotlib