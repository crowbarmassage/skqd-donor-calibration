# Scripts

## Available Scripts

### run_experiment.py
Main experiment runner for SKQD donor calibration. Runs all 4 calibration experiments:
- Si:P isolated (2 qubits)
- Si:P full (12 qubits)
- Si:Bi isolated (2 qubits)
- Si:Bi full (12 qubits)

```bash
python scripts/run_experiment.py              # Run all experiments
python scripts/run_experiment.py --config sip_isolated  # Run single experiment
```

### compare_algorithms.py
Compares different quantum diagonalization algorithms on the same Hamiltonians:
- **Classical**: Standard Krylov subspace method (baseline)
- **KQD**: Krylov Quantum Diagonalization - quantum time evolution for Krylov vectors
- **SQD**: Sample-based Quantum Diagonalization - CI subspace from sampled configurations
- **SKQD**: Sample-based Krylov QD - combines Krylov subspace with shot-based estimation

```bash
python scripts/compare_algorithms.py          # Run comparison
python scripts/compare_algorithms.py --max-iter 20 --seed 42
```

Outputs comparison results to `results/comparisons/`
