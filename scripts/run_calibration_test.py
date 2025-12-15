#!/usr/bin/env python3
"""
Calibration Test Script.

Tests that diagonalization algorithms converge to experimental
binding energies for all 4 donor systems:
- Si:P isolated (2 qubits)
- Si:P full (12 qubits)
- Si:Bi isolated (2 qubits)
- Si:Bi full (12 qubits)

Supported algorithms:
- classical: Classical Krylov (default)
- classical-sbd: Classical Sample-Based Diagonalization
- kqd: Krylov Quantum Diagonalization
- sqd: Sample-based Quantum Diagonalization
- skqd: Sample-based Krylov Quantum Diagonalization
- all: Run all algorithms

Experimental binding energies from Ramdas & Rodriguez (1981):
- Si:P: 45.59 meV
- Si:Bi: 70.98 meV
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hamiltonians.donor_valley import (
    build_calibrated_hamiltonian,
    validate_calibration,
    exact_diagonalize,
    EXPERIMENTAL_BINDING_ENERGIES,
    EXPERIMENTAL_VALLEY_ORBIT_SPLITTING,
    MEV_TO_EV
)
from src.krylov.krylov_loop import run_krylov_loop, normalize
from src.quantum_algorithms.classical_sbd import run_classical_sbd
from src.quantum_algorithms.kqd import run_kqd
from src.quantum_algorithms.sqd import run_sqd
from src.quantum_algorithms.skqd import run_skqd

# Available algorithms
ALGORITHMS = ['classical', 'classical-sbd', 'kqd', 'sqd', 'skqd']


def run_algorithm(
    algorithm: str,
    H,
    init_state: np.ndarray,
    max_iter: int,
    tolerance: float,
    verbose: bool,
    seed: int = 42
) -> dict:
    """
    Run a single algorithm and return results.

    Returns dict with: converged, iterations, final_energy, final_residual
    """
    if algorithm == 'classical':
        result = run_krylov_loop(
            H,
            initial_state=init_state,
            max_iterations=max_iter,
            residual_tolerance=tolerance,
            verbose=verbose
        )
        return {
            'converged': result.converged,
            'iterations': result.iterations_to_converge,
            'final_energy': result.final_ritz_energy,
            'final_residual': result.final_residual_norm,
        }

    elif algorithm == 'classical-sbd':
        result = run_classical_sbd(
            H,
            max_iterations=max_iter,
            samples_per_iteration=1000,
            max_configs=50,
            residual_tolerance=tolerance,
            sampling_method="importance",
            temperature=0.01,
            seed=seed,
            verbose=verbose
        )
        return {
            'converged': result.converged,
            'iterations': result.iterations_to_converge,
            'final_energy': result.final_energy,
            'final_residual': result.final_residual_norm,
        }

    elif algorithm == 'kqd':
        result = run_kqd(
            H,
            initial_state=init_state,
            max_krylov_dim=max_iter,
            evolution_time=0.5,
            trotter_steps=2,
            seed=seed,
            verbose=verbose
        )
        return {
            'converged': result.converged,
            'iterations': result.iterations_to_converge,
            'final_energy': result.final_energy,
            'final_residual': result.final_residual_norm,
        }

    elif algorithm == 'sqd':
        result = run_sqd(
            H,
            max_iterations=max_iter,
            shots_per_iteration=5000,
            max_configs=50,
            seed=seed,
            verbose=verbose
        )
        return {
            'converged': result.converged,
            'iterations': result.iterations_to_converge,
            'final_energy': result.final_energy,
            'final_residual': result.final_residual_norm,
        }

    elif algorithm == 'skqd':
        result = run_skqd(
            H,
            initial_state=init_state,
            max_krylov_dim=max_iter,
            evolution_time=0.5,
            shots=4096,
            seed=seed,
            verbose=verbose
        )
        return {
            'converged': result.converged,
            'iterations': result.iterations_to_converge,
            'final_energy': result.final_energy,
            'final_residual': result.final_residual_norm,
        }

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_calibration_test(
    donor_system: str,
    active_space: str,
    algorithms: List[str] = None,
    verbose: bool = True,
    max_iter_override: int = None,
    tolerance_override: float = None,
    seed: int = 42
) -> dict:
    """
    Run calibration test for a single system.

    Args:
        donor_system: "Si:P" or "Si:Bi"
        active_space: "isolated" or "full"
        algorithms: List of algorithms to run (default: ['classical'])
        verbose: Print iteration progress
        max_iter_override: Override default max iterations
        tolerance_override: Override default residual tolerance
        seed: Random seed

    Returns:
        Dictionary with test results
    """
    if algorithms is None:
        algorithms = ['classical']
    print(f"\n{'='*60}")
    print(f"Testing: {donor_system} ({active_space})")
    print(f"{'='*60}")

    # Get experimental values
    binding_meV = EXPERIMENTAL_BINDING_ENERGIES[donor_system]
    valley_orbit_meV = EXPERIMENTAL_VALLEY_ORBIT_SPLITTING[donor_system]
    expected_E0 = -binding_meV * MEV_TO_EV  # Convert to eV

    print(f"Experimental binding energy: {binding_meV:.2f} meV")
    print(f"Valley-orbit splitting: {valley_orbit_meV:.1f} meV")
    print(f"Expected ground state: {expected_E0:.6f} eV")

    # Build calibrated Hamiltonian
    print("\nBuilding calibrated Hamiltonian...")
    H = build_calibrated_hamiltonian(donor_system, active_space)
    n_qubits = H.num_qubits
    n_terms = len(H)
    print(f"  Qubits: {n_qubits}")
    print(f"  Terms: {n_terms}")

    # Get exact solution for reference
    print("\nComputing exact diagonalization...")
    exact_evals, _ = exact_diagonalize(H)
    exact_E0 = exact_evals[0]
    print(f"  Exact ground state: {exact_E0:.6f} eV")
    print(f"  Calibration error: {abs(exact_E0 - expected_E0)*1000:.4f} meV")

    # Prepare initial state
    dim = 2 ** n_qubits
    np.random.seed(42)
    init_state = np.random.randn(dim) + 1j * np.random.randn(dim)
    init_state, _ = normalize(init_state)

    # Set parameters based on system size (with optional overrides)
    if active_space == "isolated":
        max_iter = max_iter_override if max_iter_override else 20
        tolerance = tolerance_override if tolerance_override else 1e-10
    else:
        max_iter = max_iter_override if max_iter_override else 50
        tolerance = tolerance_override if tolerance_override else 1e-6

    # Run selected algorithms
    all_results = {}
    all_passed = True

    for algo in algorithms:
        print(f"\n--- Running {algo.upper()} (max_iter={max_iter}, tol={tolerance}) ---")
        start_time = time.time()

        result = run_algorithm(
            algo, H, init_state, max_iter, tolerance, verbose, seed
        )

        elapsed = time.time() - start_time

        # Analyze results
        algo_E0 = result['final_energy']
        error_vs_exact = abs(algo_E0 - exact_E0)
        error_vs_exp = abs(algo_E0 - expected_E0)

        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Final energy: {algo_E0:.6f} eV")
        print(f"Final residual: {result['final_residual']:.2e}")
        print(f"Error vs exact: {error_vs_exact:.2e} eV ({error_vs_exact*1000:.4f} meV)")
        print(f"Error vs experimental: {error_vs_exp:.2e} eV ({error_vs_exp*1000:.4f} meV)")

        # Determine pass/fail
        if active_space == "isolated":
            passed = error_vs_exact < 1e-8 and error_vs_exp < 0.01 * MEV_TO_EV
        else:
            # For full systems, use looser tolerance due to numerical limits
            passed = error_vs_exact < 1e-3 and error_vs_exp < 0.1 * MEV_TO_EV

        print(f"Test: {'PASS' if passed else 'FAIL'}")

        all_results[algo] = {
            'converged': result['converged'],
            'iterations': result['iterations'],
            'final_energy': algo_E0,
            'final_residual': result['final_residual'],
            'error_vs_exact_eV': error_vs_exact,
            'error_vs_exp_eV': error_vs_exp,
            'time_sec': elapsed,
            'passed': passed
        }

        if not passed:
            all_passed = False

    return {
        "donor_system": donor_system,
        "active_space": active_space,
        "n_qubits": n_qubits,
        "experimental_binding_meV": binding_meV,
        "expected_E0_eV": expected_E0,
        "exact_E0_eV": exact_E0,
        "algorithms": all_results,
        "passed": all_passed
    }


def run_all_calibration_tests(
    algorithms: List[str] = None,
    verbose: bool = False,
    max_iter_override: int = None,
    tolerance_override: float = None,
    seed: int = 42
) -> dict:
    """
    Run all 4 calibration tests.

    Args:
        algorithms: List of algorithms to run (default: ['classical'])
        verbose: Print iteration-by-iteration progress
        max_iter_override: Override default max iterations
        tolerance_override: Override default residual tolerance
        seed: Random seed

    Returns:
        Dictionary with all results
    """
    if algorithms is None:
        algorithms = ['classical']

    algo_str = ', '.join(algorithms)
    print("\n" + "#"*60)
    print("# SKQD DONOR CALIBRATION TEST SUITE")
    print(f"# Algorithms: {algo_str}")
    print("#"*60)

    systems = [
        ("Si:P", "isolated"),
        ("Si:P", "full"),
        ("Si:Bi", "isolated"),
        ("Si:Bi", "full"),
    ]

    results = {}
    all_passed = True

    for donor_system, active_space in systems:
        key = f"{donor_system}_{active_space}"
        # Auto-enable verbose for full systems (they take longer)
        use_verbose = verbose or (active_space == "full")
        results[key] = run_calibration_test(
            donor_system, active_space,
            algorithms=algorithms,
            verbose=use_verbose,
            max_iter_override=max_iter_override,
            tolerance_override=tolerance_override,
            seed=seed
        )
        if not results[key]["passed"]:
            all_passed = False

    # Print summary
    print("\n" + "="*70)
    print("CALIBRATION TEST SUMMARY")
    print("="*70)

    header = f"{'System':<20} {'Algorithm':<15} {'Conv':<6} {'Iter':<6} {'Error (meV)':<12} {'Status':<8}"
    print(header)
    print("-"*70)

    for key, data in results.items():
        for algo, algo_data in data["algorithms"].items():
            error_meV = algo_data["error_vs_exp_eV"] * 1000
            status = "PASS" if algo_data["passed"] else "FAIL"
            conv = "Yes" if algo_data["converged"] else "No"
            print(f"{key:<20} {algo:<15} {conv:<6} {algo_data['iterations']:<6} {error_meV:<12.4f} {status:<8}")

    print("-"*70)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return {
        "results": results,
        "all_passed": all_passed
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run calibration tests for SKQD donor systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_calibration_test.py                           # Classical Krylov on all systems
  python scripts/run_calibration_test.py --algorithm all           # All algorithms on all systems
  python scripts/run_calibration_test.py --algorithm kqd sqd       # KQD and SQD only
  python scripts/run_calibration_test.py --space isolated -a all   # All algorithms on isolated systems
  python scripts/run_calibration_test.py --max-iter 100 -t 1e-4    # Override hyperparameters
        """
    )
    parser.add_argument("--system", choices=["Si:P", "Si:Bi"], help="Test specific donor system")
    parser.add_argument("--space", choices=["isolated", "full"], help="Test specific active space")
    parser.add_argument("--algorithm", "-a", nargs='+', default=['classical'],
                       choices=ALGORITHMS + ['all'],
                       help="Algorithm(s) to run (default: classical). Use 'all' for all algorithms")
    parser.add_argument("--max-iter", type=int, help="Override max iterations (default: 20 isolated, 50 full)")
    parser.add_argument("--tolerance", "-t", type=float, help="Override residual tolerance (default: 1e-10 isolated, 1e-6 full)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show iteration progress")
    args = parser.parse_args()

    # Expand 'all' to list of all algorithms
    if 'all' in args.algorithm:
        algorithms = ALGORITHMS
    else:
        algorithms = args.algorithm

    if args.system and args.space:
        # Run single test
        run_calibration_test(args.system, args.space, algorithms=algorithms, verbose=True,
                           max_iter_override=args.max_iter, tolerance_override=args.tolerance,
                           seed=args.seed)
    elif args.system:
        # Run both spaces for this system
        run_calibration_test(args.system, "isolated", algorithms=algorithms, verbose=args.verbose,
                           max_iter_override=args.max_iter, tolerance_override=args.tolerance,
                           seed=args.seed)
        run_calibration_test(args.system, "full", algorithms=algorithms, verbose=True,
                           max_iter_override=args.max_iter, tolerance_override=args.tolerance,
                           seed=args.seed)
    elif args.space:
        # Run both systems for this space
        run_calibration_test("Si:P", args.space, algorithms=algorithms,
                           verbose=args.verbose or args.space == "full",
                           max_iter_override=args.max_iter, tolerance_override=args.tolerance,
                           seed=args.seed)
        run_calibration_test("Si:Bi", args.space, algorithms=algorithms,
                           verbose=args.verbose or args.space == "full",
                           max_iter_override=args.max_iter, tolerance_override=args.tolerance,
                           seed=args.seed)
    else:
        # Run all tests
        run_all_calibration_tests(algorithms=algorithms, verbose=args.verbose,
                                 max_iter_override=args.max_iter, tolerance_override=args.tolerance,
                                 seed=args.seed)
