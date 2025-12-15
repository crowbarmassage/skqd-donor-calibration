"""
Valley-basis Hamiltonians for donor systems (Si:P, Si:Bi).

This module constructs SparsePauliOp Hamiltonians for:
- Isolated (2-qubit): A1-only orbital with spin
- Full (12-qubit): 6-valley manifold with spin

The Hamiltonians are constructed in the VALLEY BASIS (non-eigen basis),
which is a hard requirement per METRICS_SPEC.md.

Physical model:
- Single-electron effective Hamiltonian for group-V donors in Si
- Valley-orbit coupling parameterized by valley_orbit_splitting
- Si:P has small splitting (~12 meV), Si:Bi has large splitting (~60 meV)
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple


# Energy scale conversion: meV to eV
MEV_TO_EV = 1e-3


def build_isolated_hamiltonian(
    valley_orbit_splitting_meV: float,
    off_diagonal_coupling_scale: float = 1.0
) -> SparsePauliOp:
    """
    Build 2-qubit isolated (A1-only) Hamiltonian in valley basis.

    The isolated Hamiltonian represents a minimal model where only the
    A1 symmetric state and its spin degree of freedom are active.

    In the valley basis, this is NOT diagonal - the valley basis states
    are the raw valley states, and A1 is a specific superposition.

    For 2 qubits, we model:
    - Qubit 0: "valley-like" orbital degree of freedom
    - Qubit 1: spin degree of freedom

    The ground state is the A1-like symmetric state.

    Args:
        valley_orbit_splitting_meV: Valley-orbit energy splitting (meV)
        off_diagonal_coupling_scale: Scale factor for off-diagonal terms

    Returns:
        SparsePauliOp representing the Hamiltonian
    """
    # Convert to eV
    delta = valley_orbit_splitting_meV * MEV_TO_EV

    # Base energy scale (arbitrary reference)
    E0 = -0.05  # eV, negative to set ground state energy scale

    # Construct Hamiltonian terms
    # In valley basis (computational basis), the Hamiltonian has:
    # 1. Diagonal terms: on-site energies
    # 2. Off-diagonal terms: valley mixing (makes it non-trivial)

    # Pauli terms for 2-qubit system:
    # II: identity (energy offset)
    # ZI, IZ: diagonal (z) terms on each qubit
    # XI, IX, YI, IY: single-qubit off-diagonal
    # XX, YY, ZZ, XY, etc.: two-qubit terms

    # Off-diagonal coupling strength
    # This creates valley mixing - the key physics
    coupling = delta * off_diagonal_coupling_scale * 0.1

    pauli_terms = [
        ("II", E0),                           # Energy offset
        ("ZI", delta / 2),                    # Valley splitting contribution
        ("IZ", delta / 4),                    # Spin-like term
        ("XX", coupling),                     # Valley mixing (XX)
        ("YY", coupling),                     # Valley mixing (YY)
        ("ZZ", delta * 0.05),                 # Spin-valley coupling
    ]

    return SparsePauliOp.from_list(pauli_terms)


def build_full_valley_hamiltonian(
    valley_orbit_splitting_meV: float,
    off_diagonal_coupling_scale: float = 1.0
) -> SparsePauliOp:
    """
    Build 12-qubit full valley manifold Hamiltonian in valley basis.

    The full Hamiltonian represents all 6 valleys (±kx, ±ky, ±kz)
    with spin, giving 12 spin-orbitals and thus 12 qubits.

    Physical model:
    - 6 valley states with inter-valley coupling
    - Spin degree of freedom for each valley
    - Valley-orbit splitting sets the energy scale between A1 and T2 states

    Valley indexing (qubit pairs):
    - Qubits 0-1: +kx (spin up/down)
    - Qubits 2-3: -kx (spin up/down)
    - Qubits 4-5: +ky (spin up/down)
    - Qubits 6-7: -ky (spin up/down)
    - Qubits 8-9: +kz (spin up/down)
    - Qubits 10-11: -kz (spin up/down)

    Args:
        valley_orbit_splitting_meV: Valley-orbit energy splitting (meV)
        off_diagonal_coupling_scale: Scale factor for off-diagonal terms

    Returns:
        SparsePauliOp representing the 12-qubit Hamiltonian
    """
    # Convert to eV
    delta = valley_orbit_splitting_meV * MEV_TO_EV

    # Base energy
    E0 = -0.05  # eV

    # Number of qubits
    n_qubits = 12

    # Inter-valley coupling strength
    # Scales with valley-orbit splitting - represents effective mixing
    # Larger delta creates both larger gaps AND stronger effective coupling
    # The net effect on convergence depends on the spectral structure
    coupling = delta * off_diagonal_coupling_scale * 0.1

    pauli_terms = []

    # Identity term (energy offset)
    pauli_terms.append(("I" * n_qubits, E0))

    # On-site energies for each valley pair
    # These create the diagonal structure in valley basis
    for valley_idx in range(6):
        qubit_up = 2 * valley_idx
        qubit_down = 2 * valley_idx + 1

        # Z terms give valley-dependent energy
        z_term_up = ["I"] * n_qubits
        z_term_up[qubit_up] = "Z"
        pauli_terms.append(("".join(z_term_up), delta / 6))

        z_term_down = ["I"] * n_qubits
        z_term_down[qubit_down] = "Z"
        pauli_terms.append(("".join(z_term_down), delta / 12))

    # Inter-valley coupling terms (XX + YY between valley pairs)
    # This creates the valley mixing that makes convergence harder
    valley_pairs = [
        (0, 1),   # +kx ↔ -kx
        (2, 3),   # +ky ↔ -ky
        (4, 5),   # +kz ↔ -kz
        (0, 2),   # +kx ↔ +ky
        (1, 3),   # -kx ↔ -ky
        (0, 4),   # +kx ↔ +kz
        (1, 5),   # -kx ↔ -kz
        (2, 4),   # +ky ↔ +kz
        (3, 5),   # -ky ↔ -kz
    ]

    for v1, v2 in valley_pairs:
        # Couple the spin-up orbitals
        q1_up = 2 * v1
        q2_up = 2 * v2

        # XX coupling
        xx_term = ["I"] * n_qubits
        xx_term[q1_up] = "X"
        xx_term[q2_up] = "X"
        pauli_terms.append(("".join(xx_term), coupling))

        # YY coupling
        yy_term = ["I"] * n_qubits
        yy_term[q1_up] = "Y"
        yy_term[q2_up] = "Y"
        pauli_terms.append(("".join(yy_term), coupling))

        # Couple the spin-down orbitals
        q1_down = 2 * v1 + 1
        q2_down = 2 * v2 + 1

        xx_term_down = ["I"] * n_qubits
        xx_term_down[q1_down] = "X"
        xx_term_down[q2_down] = "X"
        pauli_terms.append(("".join(xx_term_down), coupling))

        yy_term_down = ["I"] * n_qubits
        yy_term_down[q1_down] = "Y"
        yy_term_down[q2_down] = "Y"
        pauli_terms.append(("".join(yy_term_down), coupling))

    # Spin-valley coupling (ZZ terms within each valley)
    for valley_idx in range(6):
        q_up = 2 * valley_idx
        q_down = 2 * valley_idx + 1

        zz_term = ["I"] * n_qubits
        zz_term[q_up] = "Z"
        zz_term[q_down] = "Z"
        pauli_terms.append(("".join(zz_term), delta * 0.02))

    # Symmetry-breaking terms to ensure non-degenerate ground state
    # This represents small effective field / spin-orbit effects that
    # break exact valley and spin degeneracies
    # Different valleys get slightly different energies
    valley_asymmetry = delta * 0.01
    for valley_idx in range(6):
        q_up = 2 * valley_idx
        z_term = ["I"] * n_qubits
        z_term[q_up] = "Z"
        # Add valley-dependent asymmetry (breaks valley degeneracy)
        pauli_terms.append(("".join(z_term), valley_asymmetry * (valley_idx + 1) / 10))

    # Add weak effective Zeeman-like splitting (breaks spin degeneracy)
    zeeman = delta * 0.005
    for qubit in range(n_qubits):
        z_term = ["I"] * n_qubits
        z_term[qubit] = "Z"
        # Spin-up (even qubits) vs spin-down (odd qubits)
        sign = 1.0 if qubit % 2 == 0 else -1.0
        pauli_terms.append(("".join(z_term), sign * zeeman))

    return SparsePauliOp.from_list(pauli_terms)


def build_hamiltonian_from_config(config: dict) -> SparsePauliOp:
    """
    Build Hamiltonian from experiment configuration.

    Args:
        config: Configuration dictionary with hamiltonian and active_space sections

    Returns:
        SparsePauliOp Hamiltonian
    """
    active_space = config.get("active_space", {})
    hamiltonian_config = config.get("hamiltonian", {})

    space_type = active_space.get("type")
    delta = hamiltonian_config.get("valley_orbit_splitting_meV", 12.0)
    coupling_scale = hamiltonian_config.get("off_diagonal_coupling_scale", 1.0)

    if space_type == "isolated":
        return build_isolated_hamiltonian(delta, coupling_scale)
    elif space_type == "full":
        return build_full_valley_hamiltonian(delta, coupling_scale)
    else:
        raise ValueError(f"Unknown active space type: {space_type}")


def get_hamiltonian_matrix(hamiltonian: SparsePauliOp) -> np.ndarray:
    """Convert SparsePauliOp to dense matrix for exact diagonalization."""
    matrix = hamiltonian.to_matrix()
    # Handle both sparse and dense returns from different qiskit versions
    if hasattr(matrix, 'toarray'):
        return matrix.toarray()
    return np.asarray(matrix)


def exact_diagonalize(hamiltonian: SparsePauliOp) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform exact diagonalization of the Hamiltonian.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted by energy
    """
    matrix = get_hamiltonian_matrix(hamiltonian)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Sort by eigenvalue (should already be sorted by eigh, but be explicit)
    sort_idx = np.argsort(eigenvalues)
    return eigenvalues[sort_idx], eigenvectors[:, sort_idx]


def get_spectral_gap(eigenvalues: np.ndarray) -> float:
    """
    Compute the spectral gap (E1 - E0).

    Args:
        eigenvalues: Sorted eigenvalues

    Returns:
        Energy gap between ground and first excited state
    """
    if len(eigenvalues) < 2:
        return 0.0
    return eigenvalues[1] - eigenvalues[0]


def hamiltonian_to_metadata(
    hamiltonian: SparsePauliOp,
    config: dict
) -> dict:
    """
    Generate metadata dictionary for a Hamiltonian.

    This produces the artifact required by Step 1.1.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        config: Configuration dictionary

    Returns:
        Metadata dictionary suitable for JSON serialization
    """
    # Exact diagonalization for spectral info
    eigenvalues, _ = exact_diagonalize(hamiltonian)

    return {
        "representation": "SparsePauliOp",
        "num_qubits": hamiltonian.num_qubits,
        "term_count": len(hamiltonian),
        "energy_units": "eV",
        "config_source": {
            "system": config.get("system"),
            "active_space_type": config.get("active_space", {}).get("type"),
            "valley_orbit_splitting_meV": config.get("hamiltonian", {}).get("valley_orbit_splitting_meV"),
            "off_diagonal_coupling_scale": config.get("hamiltonian", {}).get("off_diagonal_coupling_scale"),
        },
        "spectral_properties": {
            "ground_state_energy": float(eigenvalues[0]),
            "first_excited_energy": float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
            "spectral_gap": float(get_spectral_gap(eigenvalues)),
            "is_ground_state_non_degenerate": bool(
                len(eigenvalues) > 1 and
                abs(eigenvalues[1] - eigenvalues[0]) > 1e-10
            ),
        },
        "basis_info": {
            "basis_type": "valley_basis",
            "is_eigenbasis": False,
            "description": "Computational basis represents valley states, not energy eigenstates"
        }
    }


if __name__ == "__main__":
    # Test the Hamiltonian construction
    import json

    print("Testing isolated Hamiltonian (Si:P, 12 meV)...")
    H_isolated_P = build_isolated_hamiltonian(12.0)
    print(f"  Qubits: {H_isolated_P.num_qubits}")
    print(f"  Terms: {len(H_isolated_P)}")

    evals_P, _ = exact_diagonalize(H_isolated_P)
    print(f"  Ground state energy: {evals_P[0]:.6f} eV")
    print(f"  Spectral gap: {get_spectral_gap(evals_P):.6f} eV")
    print(f"  Non-degenerate: {abs(evals_P[1] - evals_P[0]) > 1e-10}")

    print("\nTesting isolated Hamiltonian (Si:Bi, 60 meV)...")
    H_isolated_Bi = build_isolated_hamiltonian(60.0)
    evals_Bi, _ = exact_diagonalize(H_isolated_Bi)
    print(f"  Ground state energy: {evals_Bi[0]:.6f} eV")
    print(f"  Spectral gap: {get_spectral_gap(evals_Bi):.6f} eV")
    print(f"  Non-degenerate: {abs(evals_Bi[1] - evals_Bi[0]) > 1e-10}")

    print("\nTesting full Hamiltonian (Si:P, 12 meV)...")
    H_full_P = build_full_valley_hamiltonian(12.0)
    print(f"  Qubits: {H_full_P.num_qubits}")
    print(f"  Terms: {len(H_full_P)}")
    evals_full_P, _ = exact_diagonalize(H_full_P)
    print(f"  Ground state energy: {evals_full_P[0]:.6f} eV")
    print(f"  Spectral gap: {get_spectral_gap(evals_full_P):.6f} eV")

    print("\nTesting full Hamiltonian (Si:Bi, 60 meV)...")
    H_full_Bi = build_full_valley_hamiltonian(60.0)
    evals_full_Bi, _ = exact_diagonalize(H_full_Bi)
    print(f"  Ground state energy: {evals_full_Bi[0]:.6f} eV")
    print(f"  Spectral gap: {get_spectral_gap(evals_full_Bi):.6f} eV")

    # Verify expected ordering: Si:Bi should have larger gap
    print("\n--- Verification ---")
    print(f"Isolated gap (Si:P): {get_spectral_gap(evals_P):.6f} eV")
    print(f"Isolated gap (Si:Bi): {get_spectral_gap(evals_Bi):.6f} eV")
    print(f"Full gap (Si:P): {get_spectral_gap(evals_full_P):.6f} eV")
    print(f"Full gap (Si:Bi): {get_spectral_gap(evals_full_Bi):.6f} eV")

    gap_ratio_isolated = get_spectral_gap(evals_Bi) / get_spectral_gap(evals_P)
    gap_ratio_full = get_spectral_gap(evals_full_Bi) / get_spectral_gap(evals_full_P)
    print(f"\nGap ratio (Bi/P) isolated: {gap_ratio_isolated:.2f}")
    print(f"Gap ratio (Bi/P) full: {gap_ratio_full:.2f}")
    print("Expected: Si:Bi > Si:P (larger gap = easier convergence)")
