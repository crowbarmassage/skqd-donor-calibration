#!/usr/bin/env python3
"""
Main experiment runner for SKQD donor calibration.

Runs all 4 calibration experiments:
- Si:P isolated (2 qubits)
- Si:P full (12 qubits)
- Si:Bi isolated (2 qubits)
- Si:Bi full (12 qubits)

Produces RUN_LOG_SCHEMA compliant JSON logs and metrics tables.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.config_loader import load_and_validate_config, load_global_contract
from src.io.run_logger import create_run_log, save_run_log, compute_convergence_penalty
from src.hamiltonians.donor_valley import build_hamiltonian_from_config, hamiltonian_to_metadata
from src.krylov.krylov_loop import run_krylov_loop, normalize


def run_single_experiment(config_name: str, seed: int = 42, verbose: bool = False) -> dict:
    """
    Run a single calibration experiment.

    Args:
        config_name: Name of config file (without .json)
        seed: Random seed for reproducibility
        verbose: Print iteration-by-iteration progress

    Returns:
        Run log dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*60}")

    # Load and validate config
    config = load_and_validate_config(config_name, verbose=False)

    # Build Hamiltonian
    print("Building Hamiltonian...")
    hamiltonian = build_hamiltonian_from_config(config)
    h_metadata = hamiltonian_to_metadata(hamiltonian, config)

    print(f"  Qubits: {hamiltonian.num_qubits}")
    print(f"  Terms: {len(hamiltonian)}")
    print(f"  Spectral gap: {h_metadata['spectral_properties']['spectral_gap']:.6e} eV")

    # Prepare initial state
    dim = 2 ** hamiltonian.num_qubits
    np.random.seed(seed)
    init_state = np.random.randn(dim) + 1j * np.random.randn(dim)
    init_state, _ = normalize(init_state)

    # Get Krylov parameters
    krylov_config = config.get("krylov", {})
    max_iter = krylov_config.get("max_iterations", 50)
    tolerance = krylov_config.get("residual_tolerance", 1e-6)

    # Auto-enable verbose for full (12-qubit) experiments
    is_full = "full" in config_name
    use_verbose = verbose or is_full

    # Run Krylov loop
    print(f"Running Krylov loop (max_iter={max_iter}, tol={tolerance})...")
    if use_verbose:
        print("  Iteration progress:")
    start_time = time.time()

    result = run_krylov_loop(
        hamiltonian,
        initial_state=init_state,
        max_iterations=max_iter,
        residual_tolerance=tolerance,
        verbose=use_verbose
    )

    total_time = time.time() - start_time

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations_to_converge}")
    print(f"  Final energy: {result.final_ritz_energy:.6f} eV")
    print(f"  Final residual: {result.final_residual_norm:.2e}")
    print(f"  Total time: {total_time:.2f}s")

    # Create timing info
    timing_info = {
        "total_time": total_time,
        "per_iteration": [d.time_sec for d in result.iteration_data]
    }

    # Create run log
    run_log = create_run_log(
        config=config,
        krylov_result=result,
        hamiltonian_info={"term_count": len(hamiltonian)},
        timing_info=timing_info
    )

    return run_log


def run_all_experiments(seed: int = 42) -> dict:
    """
    Run all 4 calibration experiments.

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*60)
    print("SKQD DONOR CALIBRATION - FULL EXPERIMENT SUITE")
    print("="*60)

    # Print global contracts
    contract = load_global_contract()
    print(f"\nGlobal contracts:")
    print(f"  Residual tolerance: {contract['numerical_contracts']['residual_tolerance']}")
    print(f"  Basis: {contract['numerical_contracts']['basis_type']}")

    experiments = [
        "sip_isolated",
        "sibi_isolated",
        "sip_full",
        "sibi_full"
    ]

    results = {}
    run_logs = {}

    for config_name in experiments:
        run_log = run_single_experiment(config_name, seed)
        run_logs[config_name] = run_log
        results[config_name] = {
            "n_iter": run_log["convergence_results"]["iterations_to_converge"],
            "converged": run_log["convergence_results"]["converged"],
            "final_energy": run_log["convergence_results"]["final_ritz_energy"],
            "log_slope": run_log["convergence_results"]["log_residual_slope"]
        }

        # Save run log
        filepath = save_run_log(run_log)
        print(f"  Saved: {filepath}")

    # Compute convergence penalties
    print("\n" + "="*60)
    print("CONVERGENCE PENALTY ANALYSIS")
    print("="*60)

    for material in ["sip", "sibi"]:
        n_isolated = results[f"{material}_isolated"]["n_iter"]
        n_full = results[f"{material}_full"]["n_iter"]
        penalty = compute_convergence_penalty(n_full, n_isolated)

        system_name = "Si:P" if material == "sip" else "Si:Bi"
        print(f"\n{system_name}:")
        print(f"  N_iter (isolated): {n_isolated}")
        print(f"  N_iter (full): {n_full}")
        print(f"  ΔN (penalty): {penalty}")

        # Update run logs with penalty
        run_logs[f"{material}_full"]["convergence_results"]["convergence_penalty"] = penalty

        # Re-save with penalty
        filepath = save_run_log(run_logs[f"{material}_full"])
        print(f"  Updated: {filepath}")

    # Verify expected ordering
    print("\n" + "="*60)
    print("VERIFICATION: Expected Orderings")
    print("="*60)

    penalty_P = results["sip_full"]["n_iter"] - results["sip_isolated"]["n_iter"]
    penalty_Bi = results["sibi_full"]["n_iter"] - results["sibi_isolated"]["n_iter"]

    print(f"\nConvergence penalties:")
    print(f"  Si:P: {penalty_P}")
    print(f"  Si:Bi: {penalty_Bi}")
    print(f"  Expected: Si:P > Si:Bi")
    print(f"  Result: {'PASS' if penalty_P > penalty_Bi else 'NEEDS ATTENTION'}")

    return {
        "results": results,
        "run_logs": run_logs,
        "penalties": {
            "Si:P": penalty_P,
            "Si:Bi": penalty_Bi
        }
    }


def generate_metrics_table(results: dict) -> str:
    """Generate markdown metrics table."""
    lines = [
        "# Donor Calibration Metrics",
        "",
        "| System | Active Space | Qubits | N_iter | ΔN | Log Slope |",
        "|--------|--------------|--------|--------|----|-----------| "
    ]

    for config_name, data in results["results"].items():
        parts = config_name.split("_")
        system = "Si:P" if parts[0] == "sip" else "Si:Bi"
        space = parts[1]
        qubits = 2 if space == "isolated" else 12
        n_iter = data["n_iter"]

        if space == "full":
            penalty = results["penalties"][system]
            penalty_str = str(penalty)
        else:
            penalty_str = "-"

        slope = data.get("log_slope")
        slope_str = f"{slope:.2f}" if slope else "-"

        lines.append(f"| {system} | {space} | {qubits} | {n_iter} | {penalty_str} | {slope_str} |")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SKQD donor calibration experiments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, help="Run single config (e.g., sip_isolated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show iteration progress (auto-enabled for full experiments)")
    args = parser.parse_args()

    if args.config:
        # Run single experiment
        run_log = run_single_experiment(args.config, args.seed, verbose=args.verbose)
        filepath = save_run_log(run_log)
        print(f"\nSaved: {filepath}")
    else:
        # Run all experiments
        results = run_all_experiments(args.seed)

        # Generate and save metrics table
        table = generate_metrics_table(results)
        table_path = PROJECT_ROOT / "results" / "tables" / "donor_calibration_metrics.md"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with open(table_path, 'w') as f:
            f.write(table)
        print(f"\nMetrics table saved: {table_path}")

        print("\n" + "="*60)
        print("EXPERIMENT SUITE COMPLETE")
        print("="*60)
