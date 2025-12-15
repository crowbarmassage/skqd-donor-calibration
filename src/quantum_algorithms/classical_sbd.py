"""
Classical Sample-Based Diagonalization (SBD).

This is the classical analog of SQD - instead of using quantum circuits
to sample configurations, we sample from a classical probability distribution
and build a Configuration Interaction (CI) subspace from those samples.

This provides a baseline for comparing against quantum SQD.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from qiskit.quantum_info import SparsePauliOp
import scipy.linalg as la


@dataclass
class ClassicalSBDResult:
    """Result from Classical Sample-Based Diagonalization."""
    converged: bool
    iterations_to_converge: int
    final_energy: float
    final_residual_norm: float

    # Histories
    energy_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)

    # Sample statistics
    total_samples: int = 0
    unique_configs: int = 0
    subspace_dim: int = 0


def get_hamiltonian_matrix(hamiltonian: SparsePauliOp) -> np.ndarray:
    """Convert SparsePauliOp to dense matrix."""
    matrix = hamiltonian.to_matrix()
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return np.asarray(matrix)


def sample_configurations_uniform(
    n_qubits: int,
    n_samples: int,
    seed: Optional[int] = None
) -> List[int]:
    """
    Sample computational basis configurations uniformly.

    Args:
        n_qubits: Number of qubits
        n_samples: Number of samples to draw
        seed: Random seed

    Returns:
        List of configuration indices (integers)
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits
    return list(np.random.randint(0, dim, size=n_samples))


def sample_configurations_importance(
    H_matrix: np.ndarray,
    n_samples: int,
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> List[int]:
    """
    Sample configurations using importance sampling based on diagonal energies.

    Lower energy configurations are sampled more frequently (Boltzmann-like).

    Args:
        H_matrix: Hamiltonian matrix
        n_samples: Number of samples
        temperature: Sampling temperature (lower = more focused on low energy)
        seed: Random seed

    Returns:
        List of configuration indices
    """
    if seed is not None:
        np.random.seed(seed)

    # Get diagonal elements (classical energies)
    diag = np.real(np.diag(H_matrix))

    # Shift to make all positive for probability calculation
    diag_shifted = diag - np.min(diag) + 1e-10

    # Boltzmann-like weights (favor lower energies)
    weights = np.exp(-diag_shifted / temperature)
    probs = weights / np.sum(weights)

    # Sample according to distribution
    dim = len(diag)
    configs = np.random.choice(dim, size=n_samples, p=probs)

    return list(configs)


def build_ci_subspace(
    configs: List[int],
    dim: int,
    max_configs: int = 50
) -> np.ndarray:
    """
    Build CI subspace basis from sampled configurations.

    Args:
        configs: List of configuration indices
        dim: Full Hilbert space dimension
        max_configs: Maximum number of unique configurations to keep

    Returns:
        Matrix where columns are basis vectors (shape: dim x n_basis)
    """
    # Get unique configurations, keeping most frequent
    unique, counts = np.unique(configs, return_counts=True)

    # Sort by frequency (most sampled first)
    sorted_idx = np.argsort(-counts)
    selected = unique[sorted_idx[:max_configs]]

    # Build basis vectors
    n_basis = len(selected)
    basis = np.zeros((dim, n_basis), dtype=complex)

    for i, config in enumerate(selected):
        basis[config, i] = 1.0

    return basis


def diagonalize_in_subspace(
    H_matrix: np.ndarray,
    basis: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Diagonalize Hamiltonian in the CI subspace.

    Args:
        H_matrix: Full Hamiltonian matrix
        basis: Subspace basis vectors (columns)

    Returns:
        Tuple of (eigenvalues, eigenvectors_in_subspace, ground_state_in_full_space)
    """
    # Project Hamiltonian into subspace: H_sub = B^† H B
    H_sub = basis.conj().T @ H_matrix @ basis

    # Solve eigenvalue problem in subspace
    eigenvalues, eigenvectors = la.eigh(H_sub)

    # Ground state in full space
    ground_coeffs = eigenvectors[:, 0]
    ground_state = basis @ ground_coeffs
    ground_state /= np.linalg.norm(ground_state)

    return eigenvalues, eigenvectors, ground_state


def compute_residual(
    H_matrix: np.ndarray,
    state: np.ndarray,
    energy: float
) -> float:
    """Compute residual norm ||H|ψ⟩ - E|ψ⟩||."""
    residual = H_matrix @ state - energy * state
    return np.linalg.norm(residual)


def run_classical_sbd(
    hamiltonian: SparsePauliOp,
    max_iterations: int = 10,
    samples_per_iteration: int = 1000,
    max_configs: int = 50,
    residual_tolerance: float = 1e-6,
    sampling_method: str = "importance",
    temperature: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = False
) -> ClassicalSBDResult:
    """
    Run Classical Sample-Based Diagonalization.

    This algorithm:
    1. Samples computational basis configurations from a classical distribution
    2. Builds a CI subspace from the most frequently sampled configurations
    3. Diagonalizes the Hamiltonian in this subspace
    4. Iteratively refines by resampling around the current estimate

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        max_iterations: Maximum number of sampling iterations
        samples_per_iteration: Number of samples per iteration
        max_configs: Maximum configurations to keep in subspace
        residual_tolerance: Convergence threshold
        sampling_method: "uniform" or "importance"
        temperature: Temperature for importance sampling
        seed: Random seed
        verbose: Print progress

    Returns:
        ClassicalSBDResult with convergence info
    """
    if seed is not None:
        np.random.seed(seed)

    H_matrix = get_hamiltonian_matrix(hamiltonian)
    n_qubits = hamiltonian.num_qubits
    dim = 2 ** n_qubits

    energy_history = []
    residual_history = []
    total_samples = 0

    converged = False
    all_configs = []

    for iteration in range(1, max_iterations + 1):
        # Sample configurations
        if sampling_method == "importance":
            configs = sample_configurations_importance(
                H_matrix, samples_per_iteration,
                temperature=temperature,
                seed=seed + iteration if seed else None
            )
        else:
            configs = sample_configurations_uniform(
                n_qubits, samples_per_iteration,
                seed=seed + iteration if seed else None
            )

        all_configs.extend(configs)
        total_samples += samples_per_iteration

        # Build CI subspace from all samples so far
        basis = build_ci_subspace(all_configs, dim, max_configs=max_configs)

        # Diagonalize in subspace
        eigenvalues, _, ground_state = diagonalize_in_subspace(H_matrix, basis)
        energy = float(eigenvalues[0])

        # Compute residual
        residual_norm = compute_residual(H_matrix, ground_state, energy)

        energy_history.append(energy)
        residual_history.append(residual_norm)

        if verbose:
            log_res = np.log10(residual_norm) if residual_norm > 0 else -15
            print(f"  iter {iteration:3d} | E = {energy:12.6f} eV | "
                  f"residual = {residual_norm:.2e} (log={log_res:.1f}) | "
                  f"configs = {basis.shape[1]:3d}")

        # Check convergence
        if residual_norm < residual_tolerance:
            converged = True
            break

    unique_configs = len(np.unique(all_configs))

    return ClassicalSBDResult(
        converged=converged,
        iterations_to_converge=iteration if converged else max_iterations,
        final_energy=energy_history[-1] if energy_history else 0.0,
        final_residual_norm=residual_history[-1] if residual_history else float('inf'),
        energy_history=energy_history,
        residual_history=residual_history,
        total_samples=total_samples,
        unique_configs=unique_configs,
        subspace_dim=basis.shape[1] if 'basis' in dir() else 0
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian, exact_diagonalize

    print("Testing Classical Sample-Based Diagonalization...")

    # Test on Si:P isolated
    H = build_isolated_hamiltonian(12.0)
    exact_energy = exact_diagonalize(H)[0]

    print(f"\nExact ground state energy: {exact_energy:.6f} eV")

    # Run classical SBD
    print("\n--- Uniform Sampling ---")
    result_uniform = run_classical_sbd(
        H,
        max_iterations=10,
        samples_per_iteration=100,
        max_configs=4,
        sampling_method="uniform",
        seed=42,
        verbose=True
    )
    print(f"Final energy: {result_uniform.final_energy:.6f} eV")
    print(f"Error: {abs(result_uniform.final_energy - exact_energy):.2e} eV")

    print("\n--- Importance Sampling ---")
    result_importance = run_classical_sbd(
        H,
        max_iterations=10,
        samples_per_iteration=100,
        max_configs=4,
        sampling_method="importance",
        temperature=0.01,
        seed=42,
        verbose=True
    )
    print(f"Final energy: {result_importance.final_energy:.6f} eV")
    print(f"Error: {abs(result_importance.final_energy - exact_energy):.2e} eV")
