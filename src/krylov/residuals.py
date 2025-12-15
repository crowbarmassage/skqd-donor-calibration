"""
Residual computation and stopping criteria for SKQD.

This module handles:
- Residual vector computation: r_k = H|ψ_k⟩ - E_k|ψ_k⟩
- Convergence checking against the contracted tolerance (1e-6)
- Residual analysis for METRICS_SPEC compliance
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, List, Optional
from dataclasses import dataclass


# Global contract: residual tolerance (from METRICS_SPEC)
RESIDUAL_TOLERANCE = 1e-6


@dataclass
class ResidualResult:
    """Results from residual computation."""
    residual_vector: np.ndarray
    residual_norm: float
    is_converged: bool
    iteration: int


def compute_residual(
    H_matrix: np.ndarray,
    ritz_vector: np.ndarray,
    ritz_energy: float
) -> Tuple[np.ndarray, float]:
    """
    Compute residual vector r = H|ψ⟩ - E|ψ⟩.

    Per METRICS_SPEC:
    r_k = H|φ_k⟩ - E_Ritz^(k)|φ_k⟩

    Args:
        H_matrix: Hamiltonian as dense matrix
        ritz_vector: Current Ritz vector |ψ_k⟩
        ritz_energy: Current Ritz energy E_k

    Returns:
        Tuple of (residual_vector, residual_norm)
    """
    residual = H_matrix @ ritz_vector - ritz_energy * ritz_vector
    norm = np.linalg.norm(residual)
    return residual, float(norm)


def check_convergence(
    residual_norm: float,
    tolerance: float = RESIDUAL_TOLERANCE
) -> bool:
    """
    Check if convergence criterion is met.

    Per METRICS_SPEC and global contract:
    Converged when ||r_k|| < 10^{-6}

    Args:
        residual_norm: Current residual norm
        tolerance: Convergence tolerance (default: 1e-6)

    Returns:
        True if converged
    """
    return residual_norm < tolerance


def analyze_residual_decay(
    residual_history: List[float]
) -> dict:
    """
    Analyze residual decay pattern.

    Computes metrics per METRICS_SPEC:
    - Log residual slope (α_r)
    - Decay rate characterization

    Args:
        residual_history: List of residual norms at each iteration

    Returns:
        Dictionary with decay analysis
    """
    if len(residual_history) < 2:
        return {
            "log_residual_slope": None,
            "decay_type": "insufficient_data",
            "num_points": len(residual_history)
        }

    # Filter valid (non-zero) residuals
    valid_data = [(i, r) for i, r in enumerate(residual_history) if r > 1e-15]

    if len(valid_data) < 2:
        return {
            "log_residual_slope": None,
            "decay_type": "converged_immediately",
            "num_points": len(valid_data)
        }

    iterations = np.array([x[0] for x in valid_data])
    log_residuals = np.log10(np.array([x[1] for x in valid_data]))

    # Linear fit: log(r) = slope * k + intercept
    coeffs = np.polyfit(iterations, log_residuals, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])

    # Characterize decay
    if slope < -1.0:
        decay_type = "fast_exponential"
    elif slope < -0.5:
        decay_type = "moderate_exponential"
    elif slope < -0.1:
        decay_type = "slow_exponential"
    else:
        decay_type = "plateau_or_stagnant"

    # Compute R² for fit quality
    predicted = coeffs[0] * iterations + coeffs[1]
    ss_res = np.sum((log_residuals - predicted) ** 2)
    ss_tot = np.sum((log_residuals - np.mean(log_residuals)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "log_residual_slope": slope,
        "intercept": intercept,
        "decay_type": decay_type,
        "r_squared": float(r_squared),
        "num_points": len(valid_data),
        "first_residual": float(valid_data[0][1]),
        "last_residual": float(valid_data[-1][1])
    }


def compute_convergence_summary(
    residual_history: List[float],
    ritz_history: List[float],
    tolerance: float = RESIDUAL_TOLERANCE
) -> dict:
    """
    Generate complete convergence summary per METRICS_SPEC.

    Args:
        residual_history: Residual norms at each iteration
        ritz_history: Ritz energies at each iteration
        tolerance: Convergence tolerance

    Returns:
        Summary dictionary for run log
    """
    # Find convergence iteration
    converged = False
    n_iter = len(residual_history)

    for i, r in enumerate(residual_history):
        if r < tolerance:
            converged = True
            n_iter = i + 1
            break

    # Decay analysis
    decay_analysis = analyze_residual_decay(residual_history[:n_iter])

    # Ritz stabilization (|ΔE_k|)
    ritz_changes = []
    for i in range(1, len(ritz_history)):
        delta = abs(ritz_history[i] - ritz_history[i-1])
        ritz_changes.append(float(delta))

    return {
        "iterations_to_converge": n_iter,
        "converged": converged,
        "final_residual_norm": float(residual_history[-1]) if residual_history else None,
        "final_ritz_energy": float(ritz_history[-1]) if ritz_history else None,
        "residual_norm_history": [float(r) for r in residual_history],
        "ritz_energy_history": [float(e) for e in ritz_history],
        "log_residual_slope": decay_analysis.get("log_residual_slope"),
        "ritz_stabilization_history": ritz_changes,
        "decay_analysis": decay_analysis
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian
    from src.krylov.krylov_loop import run_krylov_loop, normalize, get_hamiltonian_matrix

    print("Testing residual computation and termination...\n")

    # Run Krylov loop
    H = build_isolated_hamiltonian(12.0)
    dim = 2 ** H.num_qubits

    np.random.seed(42)
    init = np.random.randn(dim) + 1j * np.random.randn(dim)
    init, _ = normalize(init)

    result = run_krylov_loop(H, initial_state=init, max_iterations=20)

    print(f"Residual history: {result.residual_norm_history}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations_to_converge}")
    print(f"Final residual: {result.final_residual_norm:.2e}")

    # Analyze decay
    summary = compute_convergence_summary(
        result.residual_norm_history,
        result.ritz_energy_history
    )

    print(f"\nDecay analysis:")
    print(f"  Log residual slope: {summary['log_residual_slope']:.3f}")
    print(f"  Decay type: {summary['decay_analysis']['decay_type']}")

    # Step 4.1 test: residual decreases with k
    is_decreasing = all(
        result.residual_norm_history[i] >= result.residual_norm_history[i+1]
        for i in range(len(result.residual_norm_history) - 1)
    )
    print(f"\n--- Step 4.1 Test ---")
    print(f"Residual decreases with k: {is_decreasing}")
    if is_decreasing:
        print("PASSED")
    else:
        print("FAILED")

    # Step 4.2 test: termination at tolerance
    print(f"\n--- Step 4.2 Test ---")
    print(f"Tolerance: {RESIDUAL_TOLERANCE}")
    print(f"Final residual: {result.final_residual_norm:.2e}")
    print(f"N_iter logged: {summary['iterations_to_converge']}")
    if result.converged and result.final_residual_norm < RESIDUAL_TOLERANCE:
        print("PASSED: Terminated correctly at ||r|| < 1e-6")
    else:
        print("FAILED: Did not terminate at correct threshold")
