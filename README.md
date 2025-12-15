# SKQD Donor Calibration

Calibration framework for Sample-Based Quantum Krylov Diagonalization (SKQD) applied to donor systems in silicon (Si:P and Si:Bi).

## Overview

This project implements and compares quantum diagonalization algorithms for finding ground state energies of donor Hamiltonians:

- **Classical Krylov**: Standard Krylov subspace method (baseline)
- **KQD**: Krylov Quantum Diagonalization using quantum time evolution
- **SQD**: Sample-based Quantum Diagonalization using CI subspace from sampled configurations
- **SKQD**: Sample-based Krylov QD combining Krylov subspace with shot-based estimation

## Physical Systems

The calibration experiments target two donor systems:

| System | Valley-Orbit Splitting | Description |
|--------|----------------------|-------------|
| Si:P | 12 meV | Phosphorus donor (small splitting) |
| Si:Bi | 60 meV | Bismuth donor (large splitting) |

Each system is tested in two active space configurations:
- **Isolated** (2 qubits): A1-only orbital with spin
- **Full** (12 qubits): 6-valley manifold (±kx, ±ky, ±kz) with spin

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd skqd-donor-calibration

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r environment/requirements.txt
```

## Quick Start

```bash
# Run all calibration experiments
python scripts/run_experiment.py

# Run single experiment
python scripts/run_experiment.py --config sip_isolated

# Compare all quantum algorithms
python scripts/compare_algorithms.py
```

## Project Structure

```
src/
  hamiltonians/        # Hamiltonian construction
    donor_valley.py    # Valley-basis Hamiltonians for Si:P, Si:Bi
  krylov/              # Classical Krylov implementation
    krylov_loop.py     # Main Krylov iteration loop
  quantum_algorithms/  # Quantum algorithm implementations
    kqd.py             # Krylov Quantum Diagonalization
    sqd.py             # Sample-based Quantum Diagonalization
    skqd.py            # Sample-based Krylov QD
  io/                  # Input/output utilities
    config_loader.py   # Configuration loading
    run_logger.py      # Run log generation

scripts/
  run_experiment.py    # Main experiment runner
  compare_algorithms.py # Algorithm comparison

configs/               # Experiment configurations
results/               # Output results and logs
```

## Key Metrics

The primary metrics tracked are:
- **N_iter**: Iterations to converge (residual < 1e-6)
- **ΔN**: Convergence penalty = N_full - N_isolated
- **Log residual slope**: Convergence rate

## Documentation

See the `contracts/` and `specs/` directories for detailed specifications:
- `METRICS_SPEC.md`: Metric definitions and requirements
- `RUN_LOG_SCHEMA.md`: Run log format specification
- `FIGURE_CONTRACT.md`: Figure generation requirements
- `TECHNICAL_SPECS.md`: Technical implementation details
