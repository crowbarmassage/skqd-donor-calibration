#!/usr/bin/env python3
"""
Calibration Test Script.

Tests that the Classical Krylov implementation converges to experimental
binding energies for all 4 donor systems:
- Si:P isolated (2 qubits)
- Si:P full (12 qubits)
- Si:Bi isolated (2 qubits)
- Si:Bi full (12 qubits)

Experimental binding energies from Ramdas & Rodriguez (1981):
- Si:P: 45.59 meV
- Si:Bi: 70.98 meV
"""

import sys
import time
import numpy as np
from pathlib import Path

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


def run_calibration_test(donor_system: str, active_space: str, verbose: bool = True) -> dict:
    """
    Run calibration test for a single system.

    Args:
        donor_system: "Si:P" or "Si:Bi"
        active_space: "isolated" or "full"
        verbose: Print iteration progress

    Returns:
        Dictionary with test results
    """
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

    # Set parameters based on system size
    if active_space == "isolated":
        max_iter = 20
        tolerance = 1e-10
    else:
        max_iter = 50
        tolerance = 1e-6

    # Run Classical Krylov
    print(f"\nRunning Classical Krylov (max_iter={max_iter}, tol={tolerance})...")
    start_time = time.time()

    result = run_krylov_loop(
        H,
        initial_state=init_state,
        max_iterations=max_iter,
        residual_tolerance=tolerance,
        verbose=verbose
    )

    elapsed = time.time() - start_time

    # Analyze results
    krylov_E0 = result.final_ritz_energy
    error_vs_exact = abs(krylov_E0 - exact_E0)
    error_vs_exp = abs(krylov_E0 - expected_E0)

    print(f"\n--- Results ---")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations_to_converge}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Final energy: {krylov_E0:.6f} eV")
    print(f"Final residual: {result.final_residual_norm:.2e}")
    print(f"Error vs exact: {error_vs_exact:.2e} eV ({error_vs_exact*1000:.4f} meV)")
    print(f"Error vs experimental: {error_vs_exp:.2e} eV ({error_vs_exp*1000:.4f} meV)")

    # Determine pass/fail
    # For isolated systems, we should match exact (which matches experimental)
    # For full systems, numerical stability may limit precision
    if active_space == "isolated":
        passed = error_vs_exact < 1e-8 and error_vs_exp < 0.01 * MEV_TO_EV
    else:
        passed = error_vs_exact < 1e-4 and result.converged

    print(f"\nTest: {'PASS' if passed else 'FAIL'}")

    return {
        "donor_system": donor_system,
        "active_space": active_space,
        "n_qubits": n_qubits,
        "experimental_binding_meV": binding_meV,
        "expected_E0_eV": expected_E0,
        "exact_E0_eV": exact_E0,
        "krylov_E0_eV": krylov_E0,
        "converged": result.converged,
        "iterations": result.iterations_to_converge,
        "final_residual": result.final_residual_norm,
        "error_vs_exact_eV": error_vs_exact,
        "error_vs_exp_eV": error_vs_exp,
        "time_sec": elapsed,
        "passed": passed
    }


def run_all_calibration_tests(verbose: bool = False) -> dict:
    """
    Run all 4 calibration tests.

    Args:
        verbose: Print iteration-by-iteration progress

    Returns:
        Dictionary with all results
    """
    print("\n" + "#"*60)
    print("# SKQD DONOR CALIBRATION TEST SUITE")
    print("# Classical Krylov Convergence to Experimental Values")
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
        results[key] = run_calibration_test(donor_system, active_space, verbose=use_verbose)
        if not results[key]["passed"]:
            all_passed = False

    # Print summary
    print("\n" + "="*60)
    print("CALIBRATION TEST SUMMARY")
    print("="*60)

    header = f"{'System':<20} {'Qubits':<8} {'Converged':<10} {'Iter':<6} {'Error (meV)':<12} {'Status':<8}"
    print(header)
    print("-"*60)

    for key, data in results.items():
        error_meV = data["error_vs_exp_eV"] * 1000
        status = "PASS" if data["passed"] else "FAIL"
        conv = "Yes" if data["converged"] else "No"
        print(f"{key:<20} {data['n_qubits']:<8} {conv:<10} {data['iterations']:<6} {error_meV:<12.4f} {status:<8}")

    print("-"*60)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return {
        "results": results,
        "all_passed": all_passed
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run calibration tests for SKQD donor systems")
    parser.add_argument("--system", choices=["Si:P", "Si:Bi"], help="Test specific donor system")
    parser.add_argument("--space", choices=["isolated", "full"], help="Test specific active space")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show iteration progress")
    args = parser.parse_args()

    if args.system and args.space:
        # Run single test
        run_calibration_test(args.system, args.space, verbose=True)
    elif args.system:
        # Run both spaces for this system
        run_calibration_test(args.system, "isolated", verbose=args.verbose)
        run_calibration_test(args.system, "full", verbose=True)
    elif args.space:
        # Run both systems for this space
        run_calibration_test("Si:P", args.space, verbose=args.verbose or args.space == "full")
        run_calibration_test("Si:Bi", args.space, verbose=args.verbose or args.space == "full")
    else:
        # Run all tests
        run_all_calibration_tests(verbose=args.verbose)
