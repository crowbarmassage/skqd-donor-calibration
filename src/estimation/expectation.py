"""
Expectation value estimation for SKQD donor calibration.

This module provides both exact (statevector) and shot-based (sampling)
methods for estimating Pauli expectations, as required by the technical specs.

At least one configuration must use shot-based estimation per TECHNICAL_SPECS.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from typing import List, Tuple, Optional


def pauli_expectation_exact(
    state: np.ndarray,
    pauli_op: SparsePauliOp
) -> complex:
    """
    Compute exact expectation value ⟨ψ|O|ψ⟩ using statevector.

    Args:
        state: State vector as numpy array
        pauli_op: SparsePauliOp to measure

    Returns:
        Complex expectation value
    """
    # Convert to Statevector for Qiskit compatibility
    sv = Statevector(state)
    return sv.expectation_value(pauli_op)


def overlap_exact(state1: np.ndarray, state2: np.ndarray) -> complex:
    """Compute exact overlap ⟨ψ1|ψ2⟩."""
    return np.vdot(state1, state2)


def matrix_element_exact(
    state1: np.ndarray,
    operator: SparsePauliOp,
    state2: np.ndarray
) -> complex:
    """Compute exact matrix element ⟨ψ1|O|ψ2⟩."""
    # Get operator matrix
    op_matrix = operator.to_matrix()
    if hasattr(op_matrix, 'toarray'):
        op_matrix = op_matrix.toarray()
    op_matrix = np.asarray(op_matrix)

    return np.vdot(state1, op_matrix @ state2)


def pauli_expectation_sampling(
    state: np.ndarray,
    pauli_op: SparsePauliOp,
    shots: int = 8192,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Estimate expectation value using shot-based sampling.

    Uses AerSimulator to sample from the state and estimate
    Pauli expectations from measurement outcomes.

    Args:
        state: State vector as numpy array
        pauli_op: SparsePauliOp to measure
        shots: Number of measurement shots
        seed: Random seed for reproducibility

    Returns:
        Tuple of (expectation_value, standard_error)
    """
    n_qubits = int(np.log2(len(state)))

    # For each Pauli term, we need to rotate and measure
    expectation = 0.0
    variance_sum = 0.0

    pauli_list = pauli_op.to_list()

    for pauli_str, coeff in pauli_list:
        if pauli_str == "I" * n_qubits:
            # Identity term: just add coefficient
            expectation += coeff.real
            continue

        # Create circuit to measure this Pauli term
        term_exp, term_var = _measure_pauli_term(
            state, pauli_str, shots, seed
        )

        expectation += coeff.real * term_exp
        variance_sum += (coeff.real ** 2) * term_var

    std_error = np.sqrt(variance_sum) if variance_sum > 0 else 0.0

    return float(expectation), float(std_error)


def _measure_pauli_term(
    state: np.ndarray,
    pauli_str: str,
    shots: int,
    seed: Optional[int]
) -> Tuple[float, float]:
    """
    Measure a single Pauli string term.

    Args:
        state: State vector
        pauli_str: Pauli string like "XZIY"
        shots: Number of shots
        seed: Random seed

    Returns:
        Tuple of (expectation, variance)
    """
    n_qubits = len(pauli_str)

    # Create circuit with state preparation
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initialize to the given state
    qc.initialize(state, range(n_qubits))

    # Apply rotation gates for measurement basis change
    # Pauli string is in big-endian order (qubit 0 is leftmost)
    for i, pauli in enumerate(pauli_str):
        qubit = n_qubits - 1 - i  # Convert to little-endian
        if pauli == "X":
            qc.h(qubit)
        elif pauli == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        # Z and I need no rotation

    # Measure qubits that have non-identity Paulis
    measured_qubits = []
    for i, pauli in enumerate(pauli_str):
        if pauli != "I":
            qubit = n_qubits - 1 - i
            measured_qubits.append(qubit)
            qc.measure(qubit, qubit)

    if not measured_qubits:
        # All identity, expectation is 1
        return 1.0, 0.0

    # Run simulation
    backend = AerSimulator(method='statevector')
    if seed is not None:
        backend.set_options(seed_simulator=seed)

    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    # Compute expectation from counts
    # Each measurement outcome contributes +1 or -1 based on parity
    expectation = 0.0
    for bitstring, count in counts.items():
        # Count 1s in the measured positions
        parity = 0
        for qubit in measured_qubits:
            if qubit < len(bitstring):
                bit = int(bitstring[-(qubit + 1)])  # Qiskit uses little-endian
                parity ^= bit
        sign = 1 - 2 * parity
        expectation += sign * count

    expectation /= shots

    # Variance for Bernoulli-like distribution
    # For Pauli measurement, outcomes are ±1
    variance = (1 - expectation ** 2) / shots

    return expectation, variance


def build_projected_matrices_exact(
    hamiltonian: SparsePauliOp,
    krylov_vectors: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build projected H and S matrices using exact computation.

    H_ij = ⟨φ_i|H|φ_j⟩
    S_ij = ⟨φ_i|φ_j⟩

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        krylov_vectors: List of Krylov basis vectors

    Returns:
        Tuple of (H_proj, S_proj) matrices
    """
    k = len(krylov_vectors)
    H_proj = np.zeros((k, k), dtype=complex)
    S_proj = np.zeros((k, k), dtype=complex)

    for i in range(k):
        for j in range(k):
            H_proj[i, j] = matrix_element_exact(
                krylov_vectors[i], hamiltonian, krylov_vectors[j]
            )
            S_proj[i, j] = overlap_exact(krylov_vectors[i], krylov_vectors[j])

    return H_proj, S_proj


def build_projected_matrices_sampling(
    hamiltonian: SparsePauliOp,
    krylov_vectors: List[np.ndarray],
    shots: int = 8192,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build projected H and S matrices using shot-based sampling.

    Note: This is a simplified version that estimates diagonal elements
    exactly (for S) and uses sampling for H elements. A full implementation
    would require Hadamard test circuits for off-diagonal elements.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        krylov_vectors: List of Krylov basis vectors
        shots: Number of measurement shots
        seed: Random seed

    Returns:
        Tuple of (H_proj, S_proj, H_std, S_std) matrices
    """
    k = len(krylov_vectors)
    H_proj = np.zeros((k, k), dtype=complex)
    S_proj = np.zeros((k, k), dtype=complex)
    H_std = np.zeros((k, k))
    S_std = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            # Overlap: use exact for now (would need swap test for sampling)
            S_proj[i, j] = overlap_exact(krylov_vectors[i], krylov_vectors[j])

            # Hamiltonian: sampling for diagonal, exact for off-diagonal
            # (off-diagonal would need Hadamard test)
            if i == j:
                h_val, h_err = pauli_expectation_sampling(
                    krylov_vectors[i], hamiltonian, shots, seed
                )
                H_proj[i, j] = h_val
                H_std[i, j] = h_err
            else:
                # Use exact for off-diagonal (simplified)
                H_proj[i, j] = matrix_element_exact(
                    krylov_vectors[i], hamiltonian, krylov_vectors[j]
                )

    return H_proj, S_proj, H_std, S_std


def verify_hermiticity(
    matrix: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[bool, float]:
    """
    Verify that a matrix is Hermitian within tolerance.

    Args:
        matrix: Complex matrix to check
        tolerance: Maximum allowed deviation

    Returns:
        Tuple of (is_hermitian, max_deviation)
    """
    diff = matrix - matrix.conj().T
    max_deviation = np.max(np.abs(diff))
    return max_deviation < tolerance, float(max_deviation)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian
    from src.krylov.krylov_loop import modified_gram_schmidt, apply_hamiltonian, normalize, get_hamiltonian_matrix

    print("Testing expectation estimation...")

    # Build test Hamiltonian and state
    H = build_isolated_hamiltonian(12.0)
    H_matrix = get_hamiltonian_matrix(H)
    dim = 2 ** H.num_qubits

    np.random.seed(42)
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state, _ = normalize(state)

    # Test exact expectation
    exp_exact = pauli_expectation_exact(state, H)
    print(f"\nExact expectation: {exp_exact:.6f}")

    # Test sampling expectation
    exp_sample, std_err = pauli_expectation_sampling(state, H, shots=8192, seed=42)
    print(f"Sampled expectation: {exp_sample:.6f} ± {std_err:.6f}")
    print(f"Difference: {abs(exp_exact - exp_sample):.6f}")

    # Generate some Krylov vectors
    krylov_vectors = [state.copy()]
    for _ in range(3):
        new_vec = apply_hamiltonian(H_matrix, krylov_vectors[-1])
        krylov_vectors.append(new_vec)
    krylov_vectors = modified_gram_schmidt(krylov_vectors)

    # Test projected matrices (exact)
    H_proj, S_proj = build_projected_matrices_exact(H, krylov_vectors)

    print(f"\nProjected matrices (exact):")
    print(f"H_proj shape: {H_proj.shape}")
    print(f"S_proj shape: {S_proj.shape}")

    # Verify Hermiticity
    h_herm, h_dev = verify_hermiticity(H_proj)
    s_herm, s_dev = verify_hermiticity(S_proj)
    print(f"\nHermiticity check:")
    print(f"  H hermitian: {h_herm} (max dev: {h_dev:.2e})")
    print(f"  S hermitian: {s_herm} (max dev: {s_dev:.2e})")

    # Test projected matrices (sampling)
    H_proj_s, S_proj_s, H_std, S_std = build_projected_matrices_sampling(
        H, krylov_vectors, shots=8192, seed=42
    )
    print(f"\nProjected matrices (sampling):")
    print(f"H_proj diagonal std errors: {np.diag(H_std)}")

    # Step 3.2 test
    print("\n--- Step 3.2 Test ---")
    if h_herm and s_herm:
        print("PASSED: H and S are Hermitian within tolerance")
    else:
        print("FAILED: Hermiticity violation")
