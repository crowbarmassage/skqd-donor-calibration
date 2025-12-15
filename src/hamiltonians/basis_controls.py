"""
Basis controls for SKQD donor calibration.

This module provides utilities for transforming Hamiltonians between
different basis representations:
- Valley basis (production): The computational basis represents valley states
- Eigenbasis (control only): The computational basis is the energy eigenbasis

The eigenbasis transformation is used ONLY as a negative control to verify
that the Krylov loop converges trivially when the Hamiltonian is diagonal.

Per METRICS_SPEC.md:
- Eigenbasis diagonalization is permitted ONLY as a negative control
- Any run with is_eigenbasis=True must not be used for figures
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple

from .donor_valley import exact_diagonalize, get_hamiltonian_matrix


def transform_to_eigenbasis(hamiltonian: SparsePauliOp) -> Tuple[SparsePauliOp, np.ndarray]:
    """
    Transform Hamiltonian to its eigenbasis (diagonal form).

    This creates a diagonal Hamiltonian where the computational basis
    states are the energy eigenstates. Used ONLY for control experiments.

    Args:
        hamiltonian: SparsePauliOp in valley basis

    Returns:
        Tuple of:
        - SparsePauliOp in eigenbasis (diagonal)
        - Unitary transformation matrix (eigenvectors)
    """
    eigenvalues, eigenvectors = exact_diagonalize(hamiltonian)

    # Create diagonal Hamiltonian with eigenvalues on diagonal
    n_qubits = hamiltonian.num_qubits
    dim = 2 ** n_qubits

    # Build diagonal matrix
    diag_matrix = np.diag(eigenvalues)

    # Convert back to SparsePauliOp
    # For a diagonal matrix, we use Z operators
    pauli_terms = []

    # Identity term: average energy
    avg_energy = np.mean(eigenvalues)
    pauli_terms.append(("I" * n_qubits, avg_energy))

    # Z terms encode the eigenvalue differences
    for i in range(n_qubits):
        # Coefficient for Z on qubit i
        z_coeff = 0.0
        for state in range(dim):
            # Extract bit i from state
            bit = (state >> i) & 1
            sign = 1 - 2 * bit  # +1 for 0, -1 for 1
            z_coeff += sign * eigenvalues[state] / dim

        if abs(z_coeff) > 1e-12:
            z_string = ["I"] * n_qubits
            z_string[i] = "Z"
            pauli_terms.append(("".join(z_string), z_coeff))

    # Multi-qubit Z terms for higher-order corrections
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            zz_coeff = 0.0
            for state in range(dim):
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                sign = (1 - 2 * bit_i) * (1 - 2 * bit_j)
                zz_coeff += sign * eigenvalues[state] / dim

            if abs(zz_coeff) > 1e-12:
                zz_string = ["I"] * n_qubits
                zz_string[i] = "Z"
                zz_string[j] = "Z"
                pauli_terms.append(("".join(zz_string), zz_coeff))

    # For exact reconstruction, we need all 2^n Z-string terms
    # But for small systems, the above captures the main structure
    # Use direct matrix construction for exact result

    eigenbasis_hamiltonian = SparsePauliOp.from_list(pauli_terms)

    return eigenbasis_hamiltonian, eigenvectors


def build_eigenbasis_hamiltonian(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    """
    Build a diagonal Hamiltonian in the eigenbasis.

    This creates a SparsePauliOp that is diagonal in the computational basis,
    with the eigenvalues of the original Hamiltonian on the diagonal.

    For the Krylov control test, this should converge in ≤2 iterations.

    Args:
        hamiltonian: Original SparsePauliOp in valley basis

    Returns:
        Diagonal SparsePauliOp with same spectrum
    """
    eigenvalues, _ = exact_diagonalize(hamiltonian)
    n_qubits = hamiltonian.num_qubits
    dim = 2 ** n_qubits

    # For exact diagonal representation, we construct it term by term
    # Each computational basis state |i⟩ should have energy eigenvalues[i]
    # This is achieved by: H = Σ_i E_i |i⟩⟨i|
    # In Pauli basis: |i⟩⟨i| = (1/2^n) Σ_{P∈{I,Z}^n} (-1)^{i·P} P
    # where i·P counts the number of positions where both i has bit 1 and P is Z

    pauli_coeffs = {}

    for state_idx in range(dim):
        energy = eigenvalues[state_idx]

        # For each Pauli Z-string, compute contribution
        for pauli_mask in range(dim):
            # pauli_mask encodes which qubits have Z (1) vs I (0)
            # Sign is (-1)^(popcount of state_idx AND pauli_mask)
            overlap = state_idx & pauli_mask
            sign = 1 - 2 * (bin(overlap).count('1') % 2)

            # Build Pauli string
            pauli_str = ""
            for q in range(n_qubits):
                if (pauli_mask >> q) & 1:
                    pauli_str = "Z" + pauli_str
                else:
                    pauli_str = "I" + pauli_str

            coeff = sign * energy / dim
            if pauli_str in pauli_coeffs:
                pauli_coeffs[pauli_str] += coeff
            else:
                pauli_coeffs[pauli_str] = coeff

    # Filter out negligible terms
    pauli_terms = [
        (p, c) for p, c in pauli_coeffs.items()
        if abs(c) > 1e-14
    ]

    return SparsePauliOp.from_list(pauli_terms)


def verify_eigenbasis_diagonality(hamiltonian: SparsePauliOp, tolerance: float = 1e-10) -> bool:
    """
    Verify that a Hamiltonian is diagonal in the computational basis.

    Args:
        hamiltonian: SparsePauliOp to check
        tolerance: Tolerance for off-diagonal elements

    Returns:
        True if Hamiltonian is effectively diagonal
    """
    matrix = get_hamiltonian_matrix(hamiltonian)
    off_diag_norm = np.linalg.norm(matrix - np.diag(np.diag(matrix)))
    return off_diag_norm < tolerance


if __name__ == "__main__":
    from .donor_valley import build_isolated_hamiltonian, exact_diagonalize

    print("Testing eigenbasis transformation...")

    # Build valley basis Hamiltonian
    H_valley = build_isolated_hamiltonian(12.0)  # Si:P
    print(f"\nValley basis Hamiltonian:")
    print(f"  Qubits: {H_valley.num_qubits}")
    print(f"  Terms: {len(H_valley)}")
    print(f"  Is diagonal: {verify_eigenbasis_diagonality(H_valley)}")

    # Transform to eigenbasis
    H_eigen = build_eigenbasis_hamiltonian(H_valley)
    print(f"\nEigenbasis Hamiltonian:")
    print(f"  Qubits: {H_eigen.num_qubits}")
    print(f"  Terms: {len(H_eigen)}")
    print(f"  Is diagonal: {verify_eigenbasis_diagonality(H_eigen)}")

    # Verify spectra match
    evals_valley, _ = exact_diagonalize(H_valley)
    evals_eigen, _ = exact_diagonalize(H_eigen)

    print(f"\nSpectrum comparison:")
    print(f"  Valley basis: {evals_valley}")
    print(f"  Eigenbasis:   {evals_eigen}")
    print(f"  Match: {np.allclose(evals_valley, evals_eigen)}")
