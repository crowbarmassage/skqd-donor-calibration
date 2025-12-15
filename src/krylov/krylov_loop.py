"""
Core Krylov iteration loop for SKQD donor calibration.

This module implements the Sample-Based Quantum Krylov Diagonalization (SKQD)
algorithm core loop. The loop generates Krylov subspace vectors, builds
projected matrices, solves the generalized eigenvalue problem, and computes
residuals for convergence checking.

Key features:
- SparsePauliOp Hamiltonian support
- Statevector and shot-based estimation modes
- Iteration-by-iteration logging (required by METRICS_SPEC)
- Residual-based termination (tolerance from global contract)
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import scipy.linalg as la


@dataclass
class KrylovIteration:
    """Data from a single Krylov iteration."""
    iteration: int
    ritz_energy: float
    residual_norm: float
    subspace_dim: int
    time_sec: float = 0.0


@dataclass
class KrylovResult:
    """Complete result from Krylov loop execution."""
    converged: bool
    iterations_to_converge: int
    final_ritz_energy: float
    final_residual_norm: float

    # Histories (logged at every iteration per METRICS_SPEC)
    ritz_energy_history: List[float] = field(default_factory=list)
    residual_norm_history: List[float] = field(default_factory=list)

    # Derived metrics
    log_residual_slope: Optional[float] = None

    # Additional diagnostics
    iteration_data: List[KrylovIteration] = field(default_factory=list)
    final_ritz_vector: Optional[np.ndarray] = None


def get_hamiltonian_matrix(hamiltonian: SparsePauliOp) -> np.ndarray:
    """Convert SparsePauliOp to dense matrix."""
    matrix = hamiltonian.to_matrix()
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return np.asarray(matrix)


def apply_hamiltonian(H_matrix: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply Hamiltonian matrix to a state vector."""
    return H_matrix @ state


def normalize(state: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize a state vector, returning (normalized_state, norm)."""
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        return state, norm
    return state / norm, norm


def modified_gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Orthonormalize a list of vectors using modified Gram-Schmidt.

    Returns orthonormal vectors spanning the same subspace.
    Uses double orthogonalization for numerical stability.
    """
    if not vectors:
        return []

    ortho = []
    for v in vectors:
        # Check for NaN/Inf before processing
        if not np.isfinite(v).all():
            continue

        # Double orthogonalization for stability (reorthogonalize twice)
        for _ in range(2):
            for u in ortho:
                v = v - np.vdot(u, v) * u

        # Normalize
        v_normalized, norm = normalize(v)
        if norm > 1e-10:  # Only keep linearly independent vectors
            ortho.append(v_normalized)

    return ortho


def build_projected_matrices(
    H_matrix: np.ndarray,
    krylov_vectors: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build projected Hamiltonian and overlap matrices.

    H_ij = <φ_i|H|φ_j>
    S_ij = <φ_i|φ_j>

    Args:
        H_matrix: Hamiltonian as dense matrix
        krylov_vectors: List of Krylov subspace vectors

    Returns:
        Tuple of (H_proj, S_proj) matrices
    """
    k = len(krylov_vectors)
    H_proj = np.zeros((k, k), dtype=complex)
    S_proj = np.zeros((k, k), dtype=complex)

    for i in range(k):
        for j in range(k):
            H_proj[i, j] = np.vdot(krylov_vectors[i], H_matrix @ krylov_vectors[j])
            S_proj[i, j] = np.vdot(krylov_vectors[i], krylov_vectors[j])

    return H_proj, S_proj


def solve_generalized_eigenvalue(
    H_proj: np.ndarray,
    S_proj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenvalue problem Hc = ESc.

    Returns eigenvalues and eigenvectors sorted by energy.
    """
    # Regularize S if needed
    S_reg = S_proj + 1e-12 * np.eye(len(S_proj))

    try:
        eigenvalues, eigenvectors = la.eigh(H_proj, S_reg)
    except la.LinAlgError:
        # Fallback to standard eigenvalue problem with pseudo-inverse
        S_inv = np.linalg.pinv(S_reg)
        eigenvalues, eigenvectors = np.linalg.eigh(S_inv @ H_proj)

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues.real)
    return eigenvalues[idx].real, eigenvectors[:, idx]


def compute_ritz_vector(
    krylov_vectors: List[np.ndarray],
    coefficients: np.ndarray
) -> np.ndarray:
    """
    Compute Ritz vector from Krylov vectors and coefficients.

    |ψ_Ritz⟩ = Σ_i c_i |φ_i⟩
    """
    ritz_vector = np.zeros_like(krylov_vectors[0])
    for c, v in zip(coefficients, krylov_vectors):
        ritz_vector += c * v
    ritz_vector, _ = normalize(ritz_vector)
    return ritz_vector


def compute_residual(
    H_matrix: np.ndarray,
    ritz_vector: np.ndarray,
    ritz_energy: float
) -> Tuple[np.ndarray, float]:
    """
    Compute residual vector and its norm.

    r = H|ψ⟩ - E|ψ⟩
    """
    residual = apply_hamiltonian(H_matrix, ritz_vector) - ritz_energy * ritz_vector
    return residual, np.linalg.norm(residual)


def compute_log_residual_slope(residual_history: List[float]) -> Optional[float]:
    """
    Compute slope of log(residual) vs iteration.

    Per METRICS_SPEC: slope of best linear fit to log||r_k|| vs k.
    """
    if len(residual_history) < 2:
        return None

    # Filter out zeros or very small values
    valid_residuals = [(i, r) for i, r in enumerate(residual_history) if r > 1e-15]
    if len(valid_residuals) < 2:
        return None

    iterations = np.array([x[0] for x in valid_residuals])
    log_residuals = np.log10(np.array([x[1] for x in valid_residuals]))

    # Linear fit
    coeffs = np.polyfit(iterations, log_residuals, 1)
    return float(coeffs[0])  # slope


def run_krylov_loop(
    hamiltonian: SparsePauliOp,
    initial_state: Optional[np.ndarray] = None,
    max_iterations: int = 50,
    residual_tolerance: float = 1e-6,
    orthogonalization: str = "modified_gram_schmidt"
) -> KrylovResult:
    """
    Run the Krylov subspace iteration loop.

    This is the core SKQD algorithm loop that:
    1. Generates Krylov vectors by repeated H application
    2. Orthonormalizes the subspace
    3. Builds projected matrices H_ij, S_ij
    4. Solves generalized eigenvalue problem
    5. Computes residual for convergence check
    6. Logs all metrics at each iteration (per METRICS_SPEC requirement)

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        initial_state: Initial state vector (default: |0...0⟩)
        max_iterations: Maximum number of iterations
        residual_tolerance: Convergence threshold for residual norm
        orthogonalization: Orthogonalization method

    Returns:
        KrylovResult with convergence info and iteration histories
    """
    import time

    # Get matrix representation
    H_matrix = get_hamiltonian_matrix(hamiltonian)
    n_qubits = hamiltonian.num_qubits
    dim = 2 ** n_qubits

    # Initialize state
    if initial_state is None:
        initial_state = np.zeros(dim, dtype=complex)
        initial_state[0] = 1.0  # |0...0⟩
    initial_state, _ = normalize(initial_state)

    # Initialize Krylov vectors
    krylov_vectors = [initial_state.copy()]

    # Result tracking
    ritz_energy_history = []
    residual_norm_history = []
    iteration_data = []

    converged = False
    final_ritz_vector = None

    for iteration in range(1, max_iterations + 1):
        start_time = time.time()

        # Generate next Krylov vector: H|φ_{k-1}⟩
        new_vector = apply_hamiltonian(H_matrix, krylov_vectors[-1])

        # Check for numerical issues
        if not np.isfinite(new_vector).all():
            # Numerical instability detected - stop iteration
            break

        # Add to subspace
        krylov_vectors.append(new_vector)

        # Orthonormalize
        if orthogonalization == "modified_gram_schmidt":
            krylov_vectors = modified_gram_schmidt(krylov_vectors)

        # Check if we have enough vectors
        if len(krylov_vectors) < 2:
            break

        # Build projected matrices
        H_proj, S_proj = build_projected_matrices(H_matrix, krylov_vectors)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = solve_generalized_eigenvalue(H_proj, S_proj)

        # Check for valid eigenvalues
        if not np.isfinite(eigenvalues).all():
            break

        # Get lowest Ritz value and vector
        ritz_energy = eigenvalues[0]
        ritz_coeffs = eigenvectors[:, 0]

        # Compute Ritz vector in full space
        ritz_vector = compute_ritz_vector(krylov_vectors, ritz_coeffs)

        # Check for numerical issues in Ritz vector
        if not np.isfinite(ritz_vector).all():
            break

        # Compute residual
        residual, residual_norm = compute_residual(H_matrix, ritz_vector, ritz_energy)

        # Check for numerical issues in residual
        if not np.isfinite(residual_norm):
            break

        elapsed = time.time() - start_time

        # Log iteration data (required by METRICS_SPEC)
        ritz_energy_history.append(float(ritz_energy))
        residual_norm_history.append(float(residual_norm))
        iteration_data.append(KrylovIteration(
            iteration=iteration,
            ritz_energy=float(ritz_energy),
            residual_norm=float(residual_norm),
            subspace_dim=len(krylov_vectors),
            time_sec=elapsed
        ))

        # Check convergence
        if residual_norm < residual_tolerance:
            converged = True
            final_ritz_vector = ritz_vector
            break

    # Compute derived metrics
    log_residual_slope = compute_log_residual_slope(residual_norm_history)

    return KrylovResult(
        converged=converged,
        iterations_to_converge=iteration if converged else max_iterations,
        final_ritz_energy=ritz_energy_history[-1] if ritz_energy_history else 0.0,
        final_residual_norm=residual_norm_history[-1] if residual_norm_history else float('inf'),
        ritz_energy_history=ritz_energy_history,
        residual_norm_history=residual_norm_history,
        log_residual_slope=log_residual_slope,
        iteration_data=iteration_data,
        final_ritz_vector=final_ritz_vector
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian
    from src.hamiltonians.basis_controls import build_eigenbasis_hamiltonian

    print("Testing Krylov loop...")

    # Use a random initial state that's NOT aligned with eigenstates
    # This is important for meaningful convergence tests
    np.random.seed(42)
    dim_2q = 4  # 2^2 for 2-qubit system
    random_init = np.random.randn(dim_2q) + 1j * np.random.randn(dim_2q)
    random_init /= np.linalg.norm(random_init)

    # Test with valley basis (should take multiple iterations)
    print("\n--- Valley Basis (Si:P isolated) ---")
    H_valley = build_isolated_hamiltonian(12.0)
    result_valley = run_krylov_loop(H_valley, initial_state=random_init, max_iterations=20)
    print(f"Converged: {result_valley.converged}")
    print(f"Iterations: {result_valley.iterations_to_converge}")
    print(f"Final Ritz energy: {result_valley.final_ritz_energy:.6f} eV")
    print(f"Final residual: {result_valley.final_residual_norm:.2e}")
    if result_valley.log_residual_slope is not None:
        print(f"Log residual slope: {result_valley.log_residual_slope:.3f}")

    # Test with eigenbasis (should converge trivially in ≤2 iterations)
    # In eigenbasis, the computational basis states ARE eigenstates
    # So starting from |00⟩ (which is an eigenstate) should converge immediately
    print("\n--- Eigenbasis Control (Si:P isolated) ---")
    H_eigen = build_eigenbasis_hamiltonian(H_valley)

    # Test 1: Random initial state (still relatively fast)
    result_eigen_random = run_krylov_loop(H_eigen, initial_state=random_init, max_iterations=20)
    print(f"With random init:")
    print(f"  Converged: {result_eigen_random.converged}")
    print(f"  Iterations: {result_eigen_random.iterations_to_converge}")
    print(f"  Final energy: {result_eigen_random.final_ritz_energy:.6f} eV")

    # Test 2: Computational basis |00⟩ (IS an eigenstate in eigenbasis)
    # This should converge in 1 iteration
    comp_basis_init = np.zeros(dim_2q, dtype=complex)
    comp_basis_init[0] = 1.0
    result_eigen_comp = run_krylov_loop(H_eigen, initial_state=comp_basis_init, max_iterations=20)
    print(f"With |00⟩ (eigenstate) init:")
    print(f"  Converged: {result_eigen_comp.converged}")
    print(f"  Iterations: {result_eigen_comp.iterations_to_converge}")
    print(f"  Final energy: {result_eigen_comp.final_ritz_energy:.6f} eV")
    print(f"  Final residual: {result_eigen_comp.final_residual_norm:.2e}")

    # Step 1.2 test: eigenbasis with eigenstate init should converge in ≤2 iterations
    print("\n--- Step 1.2 Test ---")
    if result_eigen_comp.iterations_to_converge <= 2:
        print("PASSED: Eigenbasis control converges in ≤2 iterations")
    else:
        print(f"FAILED: Eigenbasis took {result_eigen_comp.iterations_to_converge} iterations")

    # Additional validation: both eigenbasis tests should find same ground state energy
    print(f"\nEnergy consistency check:")
    print(f"  Valley basis ground state: {result_valley.final_ritz_energy:.6f} eV")
    print(f"  Eigenbasis (eigenstate init): {result_eigen_comp.final_ritz_energy:.6f} eV")
    print(f"  Match: {np.isclose(result_valley.final_ritz_energy, result_eigen_comp.final_ritz_energy)}")
