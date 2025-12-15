"""
Sample-based Quantum Diagonalization (SQD) Implementation.

This module implements SQD following the IBM Quantum learning course:
https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/sqd-overview
https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/sqd-implementation

SQD uses sampled bit-strings from quantum circuits to build a
Configuration Interaction (CI) subspace, then diagonalizes the
Hamiltonian in this subspace.

Key steps:
1. Prepare a trial state on quantum hardware
2. Sample bit-strings (configurations) from measurements
3. Build subspace from most frequent configurations
4. Diagonalize H in this subspace classically
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from collections import Counter
import scipy.linalg as la


@dataclass
class SQDResult:
    """Result from SQD algorithm execution."""
    converged: bool
    iterations_to_converge: int
    final_energy: float
    final_residual_norm: float

    energy_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)

    # SQD-specific metrics
    subspace_size: int = 0
    unique_configurations: int = 0
    total_shots: int = 0

    method: str = "SQD"


def create_ansatz_circuit(
    num_qubits: int,
    ansatz_type: str = "efficient_su2",
    reps: int = 2,
    parameters: Optional[np.ndarray] = None
) -> QuantumCircuit:
    """
    Create parameterized ansatz circuit for state preparation.

    Args:
        num_qubits: Number of qubits
        ansatz_type: Type of ansatz ("efficient_su2" or "two_local")
        reps: Number of repetitions
        parameters: Parameter values (if None, random)

    Returns:
        QuantumCircuit with bound parameters
    """
    if ansatz_type == "efficient_su2":
        ansatz = EfficientSU2(num_qubits, reps=reps)
    else:
        ansatz = TwoLocal(
            num_qubits,
            rotation_blocks=['ry', 'rz'],
            entanglement_blocks='cx',
            reps=reps
        )

    # Bind parameters
    if parameters is None:
        np.random.seed(42)
        parameters = np.random.randn(ansatz.num_parameters) * 0.1

    param_dict = dict(zip(ansatz.parameters, parameters))
    bound_circuit = ansatz.assign_parameters(param_dict)

    return bound_circuit


def sample_configurations(
    circuit: QuantumCircuit,
    shots: int = 10000,
    seed: Optional[int] = None
) -> Dict[str, int]:
    """
    Sample bit-string configurations from a quantum circuit.

    Args:
        circuit: Quantum circuit to sample from
        shots: Number of measurement shots
        seed: Random seed

    Returns:
        Dictionary mapping bit-strings to counts
    """
    # Add measurements
    qc = circuit.copy()
    qc.measure_all()

    # Run simulation
    backend = AerSimulator()
    if seed is not None:
        backend.set_options(seed_simulator=seed)

    job = backend.run(qc, shots=shots)
    counts = job.result().get_counts()

    return counts


def select_configurations(
    counts: Dict[str, int],
    max_configs: int = 100,
    threshold_fraction: float = 0.001
) -> List[str]:
    """
    Select most important configurations from sampled counts.

    Args:
        counts: Dictionary of bit-string counts
        max_configs: Maximum number of configurations to keep
        threshold_fraction: Minimum fraction of total shots

    Returns:
        List of selected bit-string configurations
    """
    total_shots = sum(counts.values())
    threshold = threshold_fraction * total_shots

    # Filter by threshold and sort by count
    filtered = [(bs, c) for bs, c in counts.items() if c >= threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    # Take top configurations
    selected = [bs for bs, _ in filtered[:max_configs]]

    return selected


def bitstring_to_statevector(
    bitstring: str,
    num_qubits: int
) -> np.ndarray:
    """
    Convert a bit-string to a computational basis state vector.

    Args:
        bitstring: Bit-string like "01101"
        num_qubits: Number of qubits

    Returns:
        State vector with 1 at the index corresponding to the bit-string
    """
    dim = 2 ** num_qubits

    # Convert bit-string to index (Qiskit uses little-endian)
    # Reverse the string for proper indexing
    idx = int(bitstring[::-1], 2)

    state = np.zeros(dim, dtype=complex)
    state[idx] = 1.0

    return state


def build_subspace_basis(
    configurations: List[str],
    num_qubits: int
) -> List[np.ndarray]:
    """
    Build orthonormal subspace basis from configurations.

    Args:
        configurations: List of bit-string configurations
        num_qubits: Number of qubits

    Returns:
        List of orthonormal basis vectors
    """
    basis = []

    for config in configurations:
        state = bitstring_to_statevector(config, num_qubits)
        basis.append(state)

    # Configurations are orthogonal by construction (computational basis states)
    return basis


def build_projected_hamiltonian(
    hamiltonian: SparsePauliOp,
    basis: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build projected Hamiltonian and overlap matrices in subspace.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        basis: Subspace basis vectors

    Returns:
        Tuple of (H_proj, S_proj) matrices
    """
    k = len(basis)
    H_proj = np.zeros((k, k), dtype=complex)
    S_proj = np.eye(k, dtype=complex)  # Orthonormal basis

    # Get Hamiltonian matrix
    H_matrix = hamiltonian.to_matrix()
    if hasattr(H_matrix, 'toarray'):
        H_matrix = H_matrix.toarray()
    H_matrix = np.asarray(H_matrix)

    for i in range(k):
        for j in range(k):
            H_proj[i, j] = np.vdot(basis[i], H_matrix @ basis[j])

    return H_proj, S_proj


def solve_subspace_eigenvalue(
    H_proj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve eigenvalue problem in subspace.

    For orthonormal basis, this is just standard eigenvalue problem.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H_proj)
    idx = np.argsort(eigenvalues.real)
    return eigenvalues[idx].real, eigenvectors[:, idx]


def compute_ritz_vector(
    basis: List[np.ndarray],
    coefficients: np.ndarray
) -> np.ndarray:
    """Compute Ritz vector from basis and coefficients."""
    ritz = np.zeros_like(basis[0])
    for c, state in zip(coefficients, basis):
        ritz += c * state
    norm = np.linalg.norm(ritz)
    if norm > 1e-10:
        ritz = ritz / norm
    return ritz


def compute_residual(
    H_matrix: np.ndarray,
    ritz_vector: np.ndarray,
    ritz_energy: float
) -> float:
    """Compute residual norm."""
    residual = H_matrix @ ritz_vector - ritz_energy * ritz_vector
    return float(np.linalg.norm(residual))


def run_sqd(
    hamiltonian: SparsePauliOp,
    max_iterations: int = 10,
    shots_per_iteration: int = 10000,
    max_configs: int = 50,
    residual_tolerance: float = 1e-6,
    ansatz_reps: int = 2,
    seed: Optional[int] = None
) -> SQDResult:
    """
    Run Sample-based Quantum Diagonalization algorithm.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        max_iterations: Maximum number of iterations
        shots_per_iteration: Shots per sampling round
        max_configs: Maximum configurations per iteration
        residual_tolerance: Convergence tolerance
        ansatz_reps: Ansatz circuit repetitions
        seed: Random seed

    Returns:
        SQDResult with convergence info
    """
    num_qubits = hamiltonian.num_qubits
    dim = 2 ** num_qubits

    # Get Hamiltonian matrix
    H_matrix = hamiltonian.to_matrix()
    if hasattr(H_matrix, 'toarray'):
        H_matrix = H_matrix.toarray()
    H_matrix = np.asarray(H_matrix)

    # Track results
    energy_history = []
    residual_history = []
    all_configurations: Set[str] = set()
    total_shots = 0

    converged = False
    final_energy = 0.0
    final_residual = float('inf')

    for iteration in range(max_iterations):
        # Create ansatz with varying parameters each iteration
        if seed is not None:
            np.random.seed(seed + iteration)
        params = np.random.randn(
            EfficientSU2(num_qubits, reps=ansatz_reps).num_parameters
        ) * (0.1 + 0.1 * iteration)

        ansatz = create_ansatz_circuit(
            num_qubits,
            ansatz_type="efficient_su2",
            reps=ansatz_reps,
            parameters=params
        )

        # Sample configurations
        counts = sample_configurations(
            ansatz,
            shots=shots_per_iteration,
            seed=seed + iteration if seed else None
        )
        total_shots += shots_per_iteration

        # Select important configurations
        new_configs = select_configurations(counts, max_configs)

        # Add to accumulated configurations
        all_configurations.update(new_configs)

        # Build subspace from all accumulated configurations
        config_list = list(all_configurations)

        # Limit total subspace size
        if len(config_list) > max_configs * 2:
            # Keep most common from recent sampling
            config_list = list(all_configurations)[:max_configs * 2]

        basis = build_subspace_basis(config_list, num_qubits)

        if len(basis) < 1:
            continue

        # Build and solve projected problem
        H_proj, S_proj = build_projected_hamiltonian(hamiltonian, basis)
        eigenvalues, eigenvectors = solve_subspace_eigenvalue(H_proj)

        # Get ground state estimate
        ritz_energy = eigenvalues[0]
        ritz_coeffs = eigenvectors[:, 0]
        ritz_vector = compute_ritz_vector(basis, ritz_coeffs)

        # Compute residual
        residual_norm = compute_residual(H_matrix, ritz_vector, ritz_energy)

        energy_history.append(float(ritz_energy))
        residual_history.append(residual_norm)

        final_energy = ritz_energy
        final_residual = residual_norm

        # Check convergence
        if residual_norm < residual_tolerance:
            converged = True
            break

    return SQDResult(
        converged=converged,
        iterations_to_converge=len(energy_history),
        final_energy=float(final_energy),
        final_residual_norm=final_residual,
        energy_history=energy_history,
        residual_history=residual_history,
        subspace_size=len(all_configurations),
        unique_configurations=len(all_configurations),
        total_shots=total_shots,
        method="SQD"
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian

    print("Testing SQD implementation...")

    # Build Hamiltonian
    H = build_isolated_hamiltonian(12.0)
    print(f"Hamiltonian: {H.num_qubits} qubits, {len(H)} terms")

    # Run SQD
    result = run_sqd(
        H,
        max_iterations=5,
        shots_per_iteration=5000,
        max_configs=10,
        seed=42
    )

    print(f"\nSQD Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations_to_converge}")
    print(f"  Final energy: {result.final_energy:.6f} eV")
    print(f"  Final residual: {result.final_residual_norm:.2e}")
    print(f"  Subspace size: {result.subspace_size}")
    print(f"  Total shots: {result.total_shots}")
