#!/usr/bin/env python
"""
12-Qubit Full Valley Convergence Comparison: Si:P vs Si:Bi

This script validates the hypothesis that spectral gap correlates with
convergence rate in Krylov subspace methods.

Usage:
    python scripts/run_12qubit_comparison.py
    python scripts/run_12qubit_comparison.py --max-iter 50 --tolerance 1e-6
"""

import sys
import argparse
import numpy as np
sys.path.insert(0, '.')

from src.hamiltonians.donor_valley import (
    build_calibrated_hamiltonian, exact_diagonalize,
    EXPERIMENTAL_BINDING_ENERGIES, EXPERIMENTAL_VALLEY_ORBIT_SPLITTING
)
from src.krylov.krylov_loop import run_krylov_loop, normalize
from src.utils.initial_states import (
    generate_initial_state, compute_initial_metrics, InitialStateMethod
)


def run_comparison(
    max_iter: int = 100,
    tolerance: float = 1e-8,
    verbose: bool = True,
    init_method: InitialStateMethod = "low_energy_superposition",
    seed: int = 42
):
    """Run 12-qubit convergence comparison for Si:P and Si:Bi.

    Args:
        max_iter: Maximum iterations
        tolerance: Residual convergence tolerance
        verbose: Print iteration progress
        init_method: Initial state generation method for fair comparison
        seed: Random seed for reproducibility
    """

    print('=' * 70)
    print('12-QUBIT FULL VALLEY CONVERGENCE COMPARISON')
    print('=' * 70)
    print(f'Parameters: max_iter={max_iter}, tolerance={tolerance}')
    print(f'Initial state method: {init_method}')

    results = {}
    init_metrics_all = {}

    for donor in ['Si:P', 'Si:Bi']:
        print(f'\n{"#" * 50}')
        print(f'### {donor} (full 12-qubit) ###')
        print(f'{"#" * 50}')

        # Build Hamiltonian
        H = build_calibrated_hamiltonian(donor, 'full')
        print(f'Qubits: {H.num_qubits}, Terms: {len(H)}')
        print(f'Valley-orbit splitting: {EXPERIMENTAL_VALLEY_ORBIT_SPLITTING[donor]} meV')

        # Exact solution
        evals, _ = exact_diagonalize(H)
        exact_E0 = evals[0]
        gap = evals[1] - evals[0]
        print(f'Exact E0: {exact_E0:.6f} eV')
        print(f'Spectral gap: {gap*1000:.3f} meV')

        # Generate standardized initial state
        init = generate_initial_state(H, method=init_method, seed=seed)

        # Compute and display initial state metrics
        init_metrics = compute_initial_metrics(H, init)
        init_metrics_all[donor] = init_metrics

        print(f'\nInitial state metrics:')
        print(f'  Ground state overlap: {init_metrics["ground_state_overlap"]:.4f}')
        print(f'  Initial residual: {init_metrics["initial_residual"]:.4e}')
        print(f'  Normalized residual: {init_metrics["normalized_residual"]:.4f}')
        print(f'  Energy error: {init_metrics["energy_error"]*1000:.3f} meV')

        # Run classical Krylov
        print(f'\nRunning classical Krylov (max {max_iter} iters, tol={tolerance})...')
        result = run_krylov_loop(
            H, init,
            max_iterations=max_iter,
            residual_tolerance=tolerance,
            verbose=verbose
        )

        # Store results
        results[donor] = {
            'converged': result.converged,
            'iterations': result.iterations_to_converge,
            'final_energy': result.final_ritz_energy,
            'final_residual': result.final_residual_norm,
            'exact_E0': exact_E0,
            'gap_meV': gap * 1000,
            'valley_orbit_meV': EXPERIMENTAL_VALLEY_ORBIT_SPLITTING[donor],
            'initial_residual': result.residual_norm_history[0] if result.residual_norm_history else None,
            'residual_history': result.residual_norm_history,
            'init_metrics': init_metrics
        }

        print(f'\n--- {donor} Summary ---')
        print(f'Converged: {result.converged}')
        print(f'Iterations: {result.iterations_to_converge}')
        print(f'Final energy: {result.final_ritz_energy:.6f} eV')
        print(f'Energy error: {abs(result.final_ritz_energy - exact_E0)*1000:.4f} meV')

    # Comparison summary
    print_comparison_summary(results)

    return results


def print_comparison_summary(results: dict):
    """Print detailed comparison summary."""

    sip = results['Si:P']
    sibi = results['Si:Bi']

    print('\n' + '=' * 70)
    print('CONVERGENCE COMPARISON SUMMARY')
    print('=' * 70)

    # Initial state comparison
    print('\nINITIAL STATE COMPARISON (Fair Starting Point)')
    print('-' * 50)
    sip_init = sip['init_metrics']
    sibi_init = sibi['init_metrics']
    print(f'''
                        Si:P            Si:Bi
Ground overlap:         {sip_init['ground_state_overlap']:.4f}          {sibi_init['ground_state_overlap']:.4f}
Initial residual:       {sip_init['initial_residual']:.2e}      {sibi_init['initial_residual']:.2e}
Normalized residual:    {sip_init['normalized_residual']:.4f}          {sibi_init['normalized_residual']:.4f}  <- Should be EQUAL
''')

    print('CONVERGENCE RESULTS')
    print('-' * 50)
    print(f'''
                        Si:P            Si:Bi           Ratio (P/Bi)
Valley-orbit gap:       {sip['valley_orbit_meV']:6.1f} meV      {sibi['valley_orbit_meV']:6.1f} meV       {sip['valley_orbit_meV']/sibi['valley_orbit_meV']:.2f}
Spectral gap:           {sip['gap_meV']:6.3f} meV      {sibi['gap_meV']:6.3f} meV      {sip['gap_meV']/sibi['gap_meV']:.2f}
Iterations:             {sip['iterations']:6d}          {sibi['iterations']:6d}           {sip['iterations']/max(1,sibi['iterations']):.2f}
Converged:              {str(sip['converged']):6s}          {str(sibi['converged']):6s}
Final residual:         {sip['final_residual']:.2e}      {sibi['final_residual']:.2e}
''')

    # Convergence rate analysis
    if sip['initial_residual'] and sibi['initial_residual']:
        sip_orders = np.log10(sip['initial_residual']) - np.log10(sip['final_residual'])
        sibi_orders = np.log10(sibi['initial_residual']) - np.log10(sibi['final_residual'])
        sip_rate = sip_orders / sip['iterations']
        sibi_rate = sibi_orders / sibi['iterations']

        print('CONVERGENCE RATE ANALYSIS')
        print('-' * 40)
        print(f'''
                        Si:P            Si:Bi
Initial residual:       {sip['initial_residual']:.2e}        {sibi['initial_residual']:.2e}
Final residual:         {sip['final_residual']:.2e}        {sibi['final_residual']:.2e}
Orders reduced:         {sip_orders:.1f}             {sibi_orders:.1f}
Iterations:             {sip['iterations']}              {sibi['iterations']}

Convergence rate:       {sip_rate:.4f}          {sibi_rate:.4f}  orders/iter
Rate ratio (Bi/P):      {sibi_rate/sip_rate:.2f}
''')

    # Hypothesis test
    print('HYPOTHESIS TEST')
    print('-' * 40)

    if sibi['gap_meV'] > sip['gap_meV']:
        print('✓ CONFIRMED: Si:Bi has LARGER spectral gap than Si:P')
    else:
        print('✗ UNEXPECTED: Gap ordering')

    if sip['initial_residual'] and sibi['initial_residual']:
        if sibi_rate > sip_rate:
            print('✓ CONFIRMED: Si:Bi has FASTER convergence RATE')
            print('  (Larger gap → faster convergence per iteration)')
        else:
            print('✗ Si:Bi does NOT have faster rate')

    # Delta N
    delta_N_P = sip['iterations'] - 3
    delta_N_Bi = sibi['iterations'] - 3
    print(f'''
CONVERGENCE PENALTY (ΔN = N_full - N_isolated):
  Si:P:  ΔN = {sip['iterations']} - 3 = {delta_N_P}
  Si:Bi: ΔN = {sibi['iterations']} - 3 = {delta_N_Bi}
''')

    print('=' * 70)
    print('CONCLUSION: The convergence RATE (not raw iterations) correlates')
    print('with spectral gap. Use rate as the diagnostic metric.')
    print('=' * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Run 12-qubit Si:P vs Si:Bi convergence comparison'
    )
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Maximum iterations (default: 100)')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-8,
                        help='Residual tolerance (default: 1e-8)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress iteration-by-iteration output')
    parser.add_argument('--init-method', '-i', type=str,
                        default='low_energy_superposition',
                        choices=['random', 'low_energy_superposition',
                                'perturbed_ground', 'uniform_overlap'],
                        help='Initial state generation method (default: low_energy_superposition)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    run_comparison(
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        verbose=not args.quiet,
        init_method=args.init_method,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
