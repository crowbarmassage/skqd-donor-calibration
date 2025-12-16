"""
Standardized Initial State Generation for Fair Algorithm Comparison.

This module provides consistent initial state generation across all materials
and algorithms, ensuring fair convergence comparisons.

The key insight: when comparing convergence across systems with different
energy scales (e.g., Si:P at 12 meV vs Si:Bi at 60 meV), using the same
random vector produces different initial residuals, biasing comparisons.

Solution: Use Hamiltonian-informed initial states that provide equivalent
"starting quality" for fair convergence rate comparisons.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple, Optional, Literal


def get_hamiltonian_matrix(hamiltonian: SparsePauliOp) -> np.ndarray:
    """Convert SparsePauliOp to dense numpy matrix."""
    matrix = hamiltonian.to_matrix()
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return np.asarray(matrix)


def generate_random_state(
    dim: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a normalized random state vector.

    Args:
        dim: Hilbert space dimension (2^n_qubits)
        seed: Random seed for reproducibility

    Returns:
        Normalized complex state vector
    """
    if seed is not None:
        np.random.seed(seed)

    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    return state / np.linalg.norm(state)


def generate_low_energy_superposition(
    hamiltonian: SparsePauliOp,
    num_states: int = 4,
    seed: Optional[int] = None,
    use_boltzmann: bool = False,
    temperature: float = 0.01
) -> np.ndarray:
    """
    Generate initial state as superposition of low-energy basis states.

    Uses Hamiltonian diagonal to identify low-energy computational basis
    states and creates a superposition, giving the algorithm a "warm start"
    that is consistent across different energy scales.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        num_states: Number of low-energy states to include
        seed: Random seed for phase randomization
        use_boltzmann: If True, weight by Boltzmann factors
        temperature: Temperature for Boltzmann weighting (in eV)

    Returns:
        Normalized initial state vector
    """
    if seed is not None:
        np.random.seed(seed)

    H_matrix = get_hamiltonian_matrix(hamiltonian)
    dim = H_matrix.shape[0]

    # Get diagonal energies
    diag = np.real(np.diag(H_matrix))

    # Sort indices by energy (lowest first)
    sorted_indices = np.argsort(diag)

    # Take lowest energy states
    low_indices = sorted_indices[:num_states]

    # Create superposition with random phases (for non-trivial starting point)
    state = np.zeros(dim, dtype=complex)

    if use_boltzmann:
        # Boltzmann-weighted superposition
        energies = diag[low_indices]
        energies = energies - np.min(energies)  # Shift to avoid overflow
        weights = np.exp(-energies / temperature)
        weights = weights / np.sum(weights)

        for idx, w in zip(low_indices, weights):
            phase = np.exp(2j * np.pi * np.random.random())
            state[idx] = np.sqrt(w) * phase
    else:
        # Equal superposition with random phases
        amplitude = 1.0 / np.sqrt(num_states)
        for idx in low_indices:
            phase = np.exp(2j * np.pi * np.random.random())
            state[idx] = amplitude * phase

    return state / np.linalg.norm(state)


def generate_perturbed_ground_eigenstate(
    hamiltonian: SparsePauliOp,
    perturbation_strength: float = 0.3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate initial state as perturbed ground state eigenstate.

    Finds the exact ground state and adds random perturbation,
    ensuring consistent initial overlap with the target.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        perturbation_strength: Fraction of state to be random noise
        seed: Random seed

    Returns:
        Normalized initial state vector
    """
    if seed is not None:
        np.random.seed(seed)

    H_matrix = get_hamiltonian_matrix(hamiltonian)
    dim = H_matrix.shape[0]

    # Get ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    ground_state = eigenvectors[:, 0]

    # Add random perturbation
    noise = np.random.randn(dim) + 1j * np.random.randn(dim)
    noise = noise / np.linalg.norm(noise)

    # Mix ground state with noise
    state = (1 - perturbation_strength) * ground_state + perturbation_strength * noise

    return state / np.linalg.norm(state)


def generate_uniform_quality_state(
    hamiltonian: SparsePauliOp,
    target_overlap: float = 0.1,
    seed: Optional[int] = None,
    max_attempts: int = 100
) -> np.ndarray:
    """
    Generate initial state with target overlap with ground state.

    This ensures all systems start with the same "distance" from the
    ground state, providing the fairest convergence comparison.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        target_overlap: Target |⟨ψ₀|ψ_init⟩|² overlap with ground state
        seed: Random seed
        max_attempts: Maximum iterations to achieve target overlap

    Returns:
        Normalized initial state with approximately target overlap
    """
    if seed is not None:
        np.random.seed(seed)

    H_matrix = get_hamiltonian_matrix(hamiltonian)
    dim = H_matrix.shape[0]

    # Get ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    ground_state = eigenvectors[:, 0]

    # Binary search for perturbation strength
    low, high = 0.0, 1.0

    for _ in range(max_attempts):
        mid = (low + high) / 2

        # Generate trial state
        noise = np.random.randn(dim) + 1j * np.random.randn(dim)
        noise = noise / np.linalg.norm(noise)

        state = (1 - mid) * ground_state + mid * noise
        state = state / np.linalg.norm(state)

        overlap = np.abs(np.vdot(ground_state, state)) ** 2

        if np.abs(overlap - target_overlap) < 0.01:
            return state

        if overlap > target_overlap:
            low = mid
        else:
            high = mid

    return state / np.linalg.norm(state)


def generate_normalized_residual_state(
    hamiltonian: SparsePauliOp,
    target_residual: float = 0.1,
    seed: Optional[int] = None,
    max_attempts: int = 50
) -> Tuple[np.ndarray, float]:
    """
    Generate initial state with target initial residual norm.

    This is the most direct way to ensure fair comparison:
    all systems start with the same residual magnitude.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        target_residual: Target initial residual norm (normalized by energy scale)
        seed: Random seed
        max_attempts: Maximum iterations

    Returns:
        Tuple of (initial_state, actual_initial_residual)
    """
    if seed is not None:
        np.random.seed(seed)

    H_matrix = get_hamiltonian_matrix(hamiltonian)
    dim = H_matrix.shape[0]

    # Get ground state and energy
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    E0 = eigenvalues[0]
    ground_state = eigenvectors[:, 0]

    # Energy scale for normalization
    energy_scale = np.abs(eigenvalues[-1] - eigenvalues[0])

    # Binary search for perturbation strength
    low, high = 0.0, 1.0
    best_state = None
    best_residual = float('inf')

    for _ in range(max_attempts):
        mid = (low + high) / 2

        # Generate trial state
        rng_state = np.random.get_state()
        noise = np.random.randn(dim) + 1j * np.random.randn(dim)
        noise = noise / np.linalg.norm(noise)

        state = (1 - mid) * ground_state + mid * noise
        state = state / np.linalg.norm(state)

        # Compute Rayleigh quotient (current energy estimate)
        energy = np.real(np.vdot(state, H_matrix @ state))

        # Compute residual: ||H|ψ⟩ - E|ψ⟩||
        residual = H_matrix @ state - energy * state
        residual_norm = np.linalg.norm(residual)

        # Normalize by energy scale for comparison
        normalized_residual = residual_norm / energy_scale

        if np.abs(normalized_residual - target_residual) < np.abs(best_residual - target_residual):
            best_state = state.copy()
            best_residual = residual_norm

        if np.abs(normalized_residual - target_residual) < 0.005:
            return state, residual_norm

        if normalized_residual < target_residual:
            low = mid
        else:
            high = mid

    return best_state, best_residual


InitialStateMethod = Literal[
    "random",
    "low_energy_superposition",
    "perturbed_ground",
    "uniform_overlap",
    "normalized_residual"
]


def generate_initial_state(
    hamiltonian: SparsePauliOp,
    method: InitialStateMethod = "low_energy_superposition",
    seed: int = 42,
    **kwargs
) -> np.ndarray:
    """
    Unified interface for initial state generation.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        method: Initial state generation method:
            - "random": Pure random state
            - "low_energy_superposition": Superposition of low-energy basis states
            - "perturbed_ground": Ground state with random perturbation
            - "uniform_overlap": State with target overlap with ground state
            - "normalized_residual": State with target initial residual
        seed: Random seed for reproducibility
        **kwargs: Method-specific arguments

    Returns:
        Normalized initial state vector
    """
    dim = 2 ** hamiltonian.num_qubits

    if method == "random":
        return generate_random_state(dim, seed)

    elif method == "low_energy_superposition":
        num_states = kwargs.get("num_states", 4)
        return generate_low_energy_superposition(
            hamiltonian, num_states=num_states, seed=seed
        )

    elif method == "perturbed_ground":
        perturbation = kwargs.get("perturbation_strength", 0.3)
        return generate_perturbed_ground_eigenstate(
            hamiltonian, perturbation_strength=perturbation, seed=seed
        )

    elif method == "uniform_overlap":
        target_overlap = kwargs.get("target_overlap", 0.1)
        return generate_uniform_quality_state(
            hamiltonian, target_overlap=target_overlap, seed=seed
        )

    elif method == "normalized_residual":
        target_residual = kwargs.get("target_residual", 0.1)
        state, _ = generate_normalized_residual_state(
            hamiltonian, target_residual=target_residual, seed=seed
        )
        return state

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_initial_metrics(
    hamiltonian: SparsePauliOp,
    initial_state: np.ndarray
) -> dict:
    """
    Compute diagnostic metrics for an initial state.

    Useful for verifying that initial states are equivalent across systems.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        initial_state: Initial state vector

    Returns:
        Dictionary of metrics
    """
    H_matrix = get_hamiltonian_matrix(hamiltonian)
    dim = H_matrix.shape[0]

    # Get exact ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    E0 = eigenvalues[0]
    ground_state = eigenvectors[:, 0]

    # Overlap with ground state
    overlap = np.abs(np.vdot(ground_state, initial_state)) ** 2

    # Initial energy expectation
    energy = np.real(np.vdot(initial_state, H_matrix @ initial_state))

    # Initial residual
    residual = H_matrix @ initial_state - energy * initial_state
    residual_norm = np.linalg.norm(residual)

    # Energy scale
    energy_scale = np.abs(eigenvalues[-1] - eigenvalues[0])
    spectral_gap = eigenvalues[1] - eigenvalues[0]

    return {
        "ground_state_overlap": overlap,
        "initial_energy": energy,
        "exact_ground_energy": E0,
        "energy_error": abs(energy - E0),
        "initial_residual": residual_norm,
        "normalized_residual": residual_norm / energy_scale,
        "energy_scale": energy_scale,
        "spectral_gap": spectral_gap,
        "spectral_gap_meV": spectral_gap * 1000
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_calibrated_hamiltonian

    print("Testing initial state generation methods...")
    print("=" * 60)

    for donor in ['Si:P', 'Si:Bi']:
        print(f"\n### {donor} ###")
        H = build_calibrated_hamiltonian(donor, 'isolated')

        for method in ["random", "low_energy_superposition", "uniform_overlap"]:
            print(f"\n  Method: {method}")
            state = generate_initial_state(H, method=method, seed=42)
            metrics = compute_initial_metrics(H, state)

            print(f"    Ground overlap: {metrics['ground_state_overlap']:.4f}")
            print(f"    Initial residual: {metrics['initial_residual']:.4e}")
            print(f"    Normalized residual: {metrics['normalized_residual']:.4f}")
            print(f"    Energy error: {metrics['energy_error']*1000:.3f} meV")
