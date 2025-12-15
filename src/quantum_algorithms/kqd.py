"""
Krylov Quantum Diagonalization (KQD) Implementation.

This module implements KQD following the IBM Quantum learning course:
https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/krylov

KQD builds a Krylov subspace using quantum circuits:
- |φ₀⟩ = initial state
- |φₖ⟩ = H^k |φ₀⟩ (via real-time evolution)

Then projects the Hamiltonian onto this subspace and solves
the generalized eigenvalue problem classically.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import scipy.linalg as la
import time


@dataclass
class KQDResult:
    """Result from KQD algorithm execution."""
    converged: bool
    iterations_to_converge: int
    final_energy: float
    final_residual_norm: float

    energy_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)

    # Quantum-specific metrics
    circuit_depths: List[int] = field(default_factory=list)
    total_circuits: int = 0
    total_shots: int = 0

    method: str = "KQD"


def create_time_evolution_circuit(
    hamiltonian: SparsePauliOp,
    time: float,
    num_qubits: int,
    trotter_steps: int = 1
) -> QuantumCircuit:
    """
    Create circuit for time evolution e^{-iHt}.

    Uses Suzuki-Trotter decomposition for the evolution.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        time: Evolution time
        num_qubits: Number of qubits
        trotter_steps: Number of Trotter steps

    Returns:
        QuantumCircuit implementing e^{-iHt}
    """
    # Create evolution gate using Suzuki-Trotter
    evolution_gate = PauliEvolutionGate(
        hamiltonian,
        time=time,
        synthesis=SuzukiTrotter(order=2, reps=trotter_steps)
    )

    qc = QuantumCircuit(num_qubits)
    qc.append(evolution_gate, range(num_qubits))

    return qc


def prepare_krylov_states_quantum(
    hamiltonian: SparsePauliOp,
    initial_state: np.ndarray,
    num_krylov: int,
    evolution_time: float = 0.1,
    trotter_steps: int = 2,
    use_statevector: bool = True,
    shots: int = 8192,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Prepare Krylov basis states using quantum circuits.

    Generates: |φ₀⟩, e^{-iHt}|φ₀⟩, e^{-2iHt}|φ₀⟩, ...

    Note: For small evolution times, e^{-iHt} ≈ I - iHt, so this
    approximates the power method |φₖ⟩ ∝ H^k|φ₀⟩.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        initial_state: Initial state vector
        num_krylov: Number of Krylov vectors to generate
        evolution_time: Time step for each evolution
        trotter_steps: Trotter steps per evolution
        use_statevector: If True, use exact statevector simulation
        shots: Number of shots for sampling (if not statevector)
        seed: Random seed

    Returns:
        Tuple of (krylov_states, circuit_depths)
    """
    num_qubits = int(np.log2(len(initial_state)))
    krylov_states = []
    circuit_depths = []

    # Backend setup
    if use_statevector:
        backend = AerSimulator(method='statevector')
    else:
        backend = AerSimulator(method='automatic')

    if seed is not None:
        backend.set_options(seed_simulator=seed)

    # First Krylov vector is just the initial state
    krylov_states.append(initial_state.copy())
    circuit_depths.append(0)

    # Generate subsequent Krylov vectors via time evolution
    current_state = initial_state.copy()

    for k in range(1, num_krylov):
        # Create circuit: initialize + evolve
        qc = QuantumCircuit(num_qubits)
        qc.initialize(current_state, range(num_qubits))

        # Add time evolution
        evo_circuit = create_time_evolution_circuit(
            hamiltonian, evolution_time, num_qubits, trotter_steps
        )
        qc.compose(evo_circuit, inplace=True)

        # Save statevector
        qc.save_statevector()

        # Run circuit
        job = backend.run(qc, shots=1)
        result = job.result()

        # Get evolved state
        evolved_state = np.asarray(result.get_statevector())

        # Normalize
        norm = np.linalg.norm(evolved_state)
        if norm > 1e-10:
            evolved_state = evolved_state / norm

        krylov_states.append(evolved_state)
        circuit_depths.append(qc.depth())

        # Update current state for next iteration
        current_state = evolved_state

    return krylov_states, circuit_depths


def estimate_matrix_elements_quantum(
    hamiltonian: SparsePauliOp,
    krylov_states: List[np.ndarray],
    use_hadamard_test: bool = False,
    shots: int = 8192,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate projected Hamiltonian and overlap matrices.

    H_ij = ⟨φᵢ|H|φⱼ⟩
    S_ij = ⟨φᵢ|φⱼ⟩

    For now, uses exact statevector computation. A full implementation
    would use Hadamard tests for matrix elements.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        krylov_states: List of Krylov basis vectors
        use_hadamard_test: If True, use Hadamard test circuits (not implemented)
        shots: Number of shots for sampling
        seed: Random seed

    Returns:
        Tuple of (H_proj, S_proj) matrices
    """
    k = len(krylov_states)
    H_proj = np.zeros((k, k), dtype=complex)
    S_proj = np.zeros((k, k), dtype=complex)

    # Get Hamiltonian matrix
    H_matrix = hamiltonian.to_matrix()
    if hasattr(H_matrix, 'toarray'):
        H_matrix = H_matrix.toarray()
    H_matrix = np.asarray(H_matrix)

    for i in range(k):
        for j in range(k):
            # Overlap
            S_proj[i, j] = np.vdot(krylov_states[i], krylov_states[j])

            # Hamiltonian matrix element
            H_proj[i, j] = np.vdot(krylov_states[i], H_matrix @ krylov_states[j])

    return H_proj, S_proj


def orthonormalize_krylov(
    krylov_states: List[np.ndarray]
) -> List[np.ndarray]:
    """Orthonormalize Krylov states using modified Gram-Schmidt."""
    ortho_states = []

    for v in krylov_states:
        v_ortho = v.copy()
        for u in ortho_states:
            v_ortho = v_ortho - np.vdot(u, v_ortho) * u

        norm = np.linalg.norm(v_ortho)
        if norm > 1e-10:
            ortho_states.append(v_ortho / norm)

    return ortho_states


def solve_projected_eigenvalue(
    H_proj: np.ndarray,
    S_proj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve generalized eigenvalue problem Hc = ESc.

    Returns eigenvalues and eigenvectors sorted by energy.
    """
    # Regularize S
    S_reg = S_proj + 1e-10 * np.eye(len(S_proj))

    try:
        eigenvalues, eigenvectors = la.eigh(H_proj, S_reg)
    except la.LinAlgError:
        # Fallback
        S_inv = np.linalg.pinv(S_reg)
        eigenvalues, eigenvectors = np.linalg.eigh(S_inv @ H_proj)

    idx = np.argsort(eigenvalues.real)
    return eigenvalues[idx].real, eigenvectors[:, idx]


def compute_ritz_vector(
    krylov_states: List[np.ndarray],
    coefficients: np.ndarray
) -> np.ndarray:
    """Compute Ritz vector from Krylov states and coefficients."""
    ritz = np.zeros_like(krylov_states[0])
    for c, state in zip(coefficients, krylov_states):
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
    """Compute residual norm ||H|ψ⟩ - E|ψ⟩||."""
    residual = H_matrix @ ritz_vector - ritz_energy * ritz_vector
    return float(np.linalg.norm(residual))


def run_kqd(
    hamiltonian: SparsePauliOp,
    initial_state: Optional[np.ndarray] = None,
    max_krylov_dim: int = 20,
    residual_tolerance: float = 1e-6,
    evolution_time: float = 0.1,
    trotter_steps: int = 2,
    use_statevector: bool = True,
    shots: int = 8192,
    seed: Optional[int] = None
) -> KQDResult:
    """
    Run Krylov Quantum Diagonalization algorithm.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        initial_state: Initial state (default: |0...0⟩)
        max_krylov_dim: Maximum Krylov subspace dimension
        residual_tolerance: Convergence tolerance
        evolution_time: Time step for quantum evolution
        trotter_steps: Trotter steps per evolution
        use_statevector: Use exact statevector simulation
        shots: Number of shots for sampling
        seed: Random seed

    Returns:
        KQDResult with convergence info and metrics
    """
    num_qubits = hamiltonian.num_qubits
    dim = 2 ** num_qubits

    # Initialize state
    if initial_state is None:
        initial_state = np.zeros(dim, dtype=complex)
        initial_state[0] = 1.0

    # Get Hamiltonian matrix for residual computation
    H_matrix = hamiltonian.to_matrix()
    if hasattr(H_matrix, 'toarray'):
        H_matrix = H_matrix.toarray()
    H_matrix = np.asarray(H_matrix)

    # Results tracking
    energy_history = []
    residual_history = []
    all_circuit_depths = []

    converged = False
    final_energy = 0.0
    final_residual = float('inf')

    # Iteratively build Krylov subspace
    for k in range(2, max_krylov_dim + 1):
        # Generate Krylov states
        krylov_states, circuit_depths = prepare_krylov_states_quantum(
            hamiltonian, initial_state, k,
            evolution_time, trotter_steps,
            use_statevector, shots, seed
        )
        all_circuit_depths.extend(circuit_depths)

        # Orthonormalize
        krylov_states = orthonormalize_krylov(krylov_states)

        if len(krylov_states) < 2:
            continue

        # Build projected matrices
        H_proj, S_proj = estimate_matrix_elements_quantum(
            hamiltonian, krylov_states, shots=shots, seed=seed
        )

        # Solve eigenvalue problem
        eigenvalues, eigenvectors = solve_projected_eigenvalue(H_proj, S_proj)

        # Get ground state estimate
        ritz_energy = eigenvalues[0]
        ritz_coeffs = eigenvectors[:, 0]
        ritz_vector = compute_ritz_vector(krylov_states, ritz_coeffs)

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

    return KQDResult(
        converged=converged,
        iterations_to_converge=len(energy_history),
        final_energy=float(final_energy),
        final_residual_norm=final_residual,
        energy_history=energy_history,
        residual_history=residual_history,
        circuit_depths=all_circuit_depths,
        total_circuits=len(all_circuit_depths),
        total_shots=shots * len(all_circuit_depths) if not use_statevector else 0,
        method="KQD"
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian

    print("Testing KQD implementation...")

    # Build Hamiltonian
    H = build_isolated_hamiltonian(12.0)
    print(f"Hamiltonian: {H.num_qubits} qubits, {len(H)} terms")

    # Random initial state
    np.random.seed(42)
    dim = 2 ** H.num_qubits
    init = np.random.randn(dim) + 1j * np.random.randn(dim)
    init /= np.linalg.norm(init)

    # Run KQD
    result = run_kqd(
        H,
        initial_state=init,
        max_krylov_dim=10,
        evolution_time=0.5,
        trotter_steps=2
    )

    print(f"\nKQD Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations_to_converge}")
    print(f"  Final energy: {result.final_energy:.6f} eV")
    print(f"  Final residual: {result.final_residual_norm:.2e}")
    print(f"  Circuit depths: {result.circuit_depths[:5]}...")
