#!/usr/bin/env python3
"""
Algorithm Comparison Script.

Compares the performance of different quantum diagonalization algorithms:
- Classical: Classical Krylov (baseline)
- KQD: Krylov Quantum Diagonalization
- SQD: Sample-based Quantum Diagonalization
- SKQD: Sample-based Krylov Quantum Diagonalization

Outputs a comparison table and convergence plots.
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

from src.hamiltonians.donor_valley import build_isolated_hamiltonian, build_full_valley_hamiltonian
from src.krylov.krylov_loop import run_krylov_loop, normalize
from src.quantum_algorithms.kqd import run_kqd
from src.quantum_algorithms.sqd import run_sqd
from src.quantum_algorithms.skqd import run_skqd
from src.quantum_algorithms.classical_sbd import run_classical_sbd


def run_comparison(
    hamiltonian,
    system_name: str,
    seed: int = 42,
    max_iter: int = 20,
    verbose: bool = False
) -> dict:
    """
    Run all algorithms on a given Hamiltonian and compare results.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        system_name: Name for logging
        seed: Random seed
        max_iter: Maximum iterations
        verbose: Print iteration-by-iteration progress

    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*60}")
    print(f"Comparing algorithms: {system_name}")
    print(f"Qubits: {hamiltonian.num_qubits}, Terms: {len(hamiltonian)}")
    print(f"{'='*60}")

    # Prepare initial state
    dim = 2 ** hamiltonian.num_qubits
    np.random.seed(seed)
    init_state = np.random.randn(dim) + 1j * np.random.randn(dim)
    init_state, _ = normalize(init_state)

    results = {}

    # 1. Classical Krylov (baseline)
    print("\n[1/5] Running Classical Krylov...")
    start = time.time()
    classical_result = run_krylov_loop(
        hamiltonian,
        initial_state=init_state,
        max_iterations=max_iter,
        residual_tolerance=1e-6,
        verbose=verbose
    )
    classical_time = time.time() - start

    results['Classical'] = {
        'method': 'Classical Krylov',
        'converged': classical_result.converged,
        'iterations': classical_result.iterations_to_converge,
        'final_energy': float(classical_result.final_ritz_energy),
        'final_residual': float(classical_result.final_residual_norm),
        'time_sec': classical_time,
        'total_shots': 0,
        'energy_history': classical_result.ritz_energy_history,
        'residual_history': classical_result.residual_norm_history
    }
    print(f"   Converged: {classical_result.converged}, "
          f"Iter: {classical_result.iterations_to_converge}, "
          f"E={classical_result.final_ritz_energy:.6f} eV")

    # 2. Classical SBD (sample-based baseline)
    print("\n[2/5] Running Classical SBD...")
    start = time.time()
    sbd_result = run_classical_sbd(
        hamiltonian,
        max_iterations=max_iter,
        samples_per_iteration=1000,
        max_configs=20,
        sampling_method="importance",
        temperature=0.01,
        seed=seed,
        verbose=verbose
    )
    sbd_time = time.time() - start

    results['Classical SBD'] = {
        'method': 'Classical SBD',
        'converged': sbd_result.converged,
        'iterations': sbd_result.iterations_to_converge,
        'final_energy': float(sbd_result.final_energy),
        'final_residual': float(sbd_result.final_residual_norm),
        'time_sec': sbd_time,
        'total_shots': sbd_result.total_samples,
        'energy_history': sbd_result.energy_history,
        'residual_history': sbd_result.residual_history
    }
    print(f"   Converged: {sbd_result.converged}, "
          f"Iter: {sbd_result.iterations_to_converge}, "
          f"E={sbd_result.final_energy:.6f} eV, "
          f"Samples: {sbd_result.total_samples}")

    # 3. KQD
    print("\n[3/5] Running KQD...")
    start = time.time()
    kqd_result = run_kqd(
        hamiltonian,
        initial_state=init_state,
        max_krylov_dim=max_iter,
        evolution_time=0.5,
        trotter_steps=2,
        seed=seed
    )
    kqd_time = time.time() - start

    results['KQD'] = {
        'method': 'KQD',
        'converged': kqd_result.converged,
        'iterations': kqd_result.iterations_to_converge,
        'final_energy': float(kqd_result.final_energy),
        'final_residual': float(kqd_result.final_residual_norm),
        'time_sec': kqd_time,
        'total_shots': kqd_result.total_shots,
        'energy_history': kqd_result.energy_history,
        'residual_history': kqd_result.residual_history
    }
    print(f"   Converged: {kqd_result.converged}, "
          f"Iter: {kqd_result.iterations_to_converge}, "
          f"E={kqd_result.final_energy:.6f} eV")

    # 4. SQD
    print("\n[4/5] Running SQD...")
    start = time.time()
    sqd_result = run_sqd(
        hamiltonian,
        max_iterations=max_iter,
        shots_per_iteration=5000,
        max_configs=20,
        seed=seed
    )
    sqd_time = time.time() - start

    results['SQD'] = {
        'method': 'SQD',
        'converged': sqd_result.converged,
        'iterations': sqd_result.iterations_to_converge,
        'final_energy': float(sqd_result.final_energy),
        'final_residual': float(sqd_result.final_residual_norm),
        'time_sec': sqd_time,
        'total_shots': sqd_result.total_shots,
        'energy_history': sqd_result.energy_history,
        'residual_history': sqd_result.residual_history
    }
    print(f"   Converged: {sqd_result.converged}, "
          f"Iter: {sqd_result.iterations_to_converge}, "
          f"E={sqd_result.final_energy:.6f} eV")

    # 5. SKQD
    print("\n[5/5] Running SKQD...")
    start = time.time()
    skqd_result = run_skqd(
        hamiltonian,
        initial_state=init_state,
        max_krylov_dim=max_iter,
        evolution_time=0.5,
        shots=4096,
        seed=seed
    )
    skqd_time = time.time() - start

    results['SKQD'] = {
        'method': 'SKQD',
        'converged': skqd_result.converged,
        'iterations': skqd_result.iterations_to_converge,
        'final_energy': float(skqd_result.final_energy),
        'final_residual': float(skqd_result.final_residual_norm),
        'time_sec': skqd_time,
        'total_shots': skqd_result.total_shots,
        'energy_history': skqd_result.energy_history,
        'residual_history': skqd_result.residual_history
    }
    print(f"   Converged: {skqd_result.converged}, "
          f"Iter: {skqd_result.iterations_to_converge}, "
          f"E={skqd_result.final_energy:.6f} eV")

    return results


def print_comparison_table(all_results: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'System':<20} {'Method':<12} {'Conv':<6} {'Iter':<6} {'Energy (eV)':<14} {'Residual':<12} {'Shots':<10}"
    print(header)
    print("-" * 80)

    for system, results in all_results.items():
        for method, data in results.items():
            conv = "Yes" if data['converged'] else "No"
            energy = f"{data['final_energy']:.6f}"
            residual = f"{data['final_residual']:.2e}"
            shots = str(data['total_shots']) if data['total_shots'] > 0 else "-"

            row = f"{system:<20} {method:<12} {conv:<6} {data['iterations']:<6} {energy:<14} {residual:<12} {shots:<10}"
            print(row)

        print("-" * 80)


def save_comparison_results(all_results: dict, output_dir: Path):
    """Save comparison results to JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / "algorithm_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved JSON: {json_path}")

    # Markdown table
    md_path = output_dir / "algorithm_comparison.md"
    with open(md_path, 'w') as f:
        f.write("# Algorithm Comparison Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| System | Method | Converged | Iterations | Energy (eV) | Residual | Shots |\n")
        f.write("|--------|--------|-----------|------------|-------------|----------|-------|\n")

        for system, results in all_results.items():
            for method, data in results.items():
                conv = "✓" if data['converged'] else "✗"
                shots = str(data['total_shots']) if data['total_shots'] > 0 else "-"
                f.write(f"| {system} | {method} | {conv} | {data['iterations']} | "
                       f"{data['final_energy']:.6f} | {data['final_residual']:.2e} | {shots} |\n")

        f.write("\n## Algorithm Descriptions\n\n")
        f.write("- **Classical**: Standard Krylov subspace method using numpy\n")
        f.write("- **KQD**: Krylov Quantum Diagonalization - quantum time evolution for Krylov vectors\n")
        f.write("- **SQD**: Sample-based Quantum Diagonalization - CI subspace from sampled configurations\n")
        f.write("- **SKQD**: Sample-based Krylov QD - combines Krylov subspace with shot-based estimation\n")

    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare quantum diagonalization algorithms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iter", type=int, default=15, help="Maximum iterations")
    parser.add_argument("--system", choices=['isolated', 'full', 'all'], default='isolated',
                       help="Which systems to test (isolated=2q, full=12q, all=both)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show iteration progress")
    args = parser.parse_args()

    all_results = {}

    # Test on isolated (2-qubit) systems
    if args.system in ['isolated', 'all']:
        print("\n" + "#" * 60)
        print("# Testing 2-qubit (isolated) systems")
        print("#" * 60)

        for system, delta in [("Si:P", 12.0), ("Si:Bi", 60.0)]:
            H = build_isolated_hamiltonian(delta)
            name = f"{system} (2q)"
            results = run_comparison(H, name, seed=args.seed, max_iter=args.max_iter, verbose=args.verbose)
            all_results[name] = results

    # Test on full (12-qubit) systems
    if args.system in ['full', 'all']:
        print("\n" + "#" * 60)
        print("# Testing 12-qubit (full) systems")
        print("# NOTE: This takes much longer due to larger Hilbert space")
        print("#" * 60)

        for system, delta in [("Si:P", 12.0), ("Si:Bi", 60.0)]:
            H = build_full_valley_hamiltonian(delta)
            name = f"{system} (12q)"
            # Use verbose=True by default for full systems
            results = run_comparison(H, name, seed=args.seed, max_iter=args.max_iter, verbose=True)
            all_results[name] = results

    # Print summary
    print_comparison_table(all_results)

    # Save results
    output_dir = PROJECT_ROOT / "results" / "comparisons"
    save_comparison_results(all_results, output_dir)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
