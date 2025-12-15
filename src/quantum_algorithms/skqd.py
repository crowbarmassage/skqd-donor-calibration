"""
Sample-based Krylov Quantum Diagonalization (SKQD) Implementation.

This module implements SKQD following the IBM Quantum learning course:
https://quantum.cloud.ibm.com/learning/en/courses/quantum-diagonalization-algorithms/skqd

SKQD combines the Krylov subspace approach with shot-based sampling:
1. Build Krylov subspace via quantum time evolution
2. Estimate matrix elements using sampling (not exact statevector)
3. Use Hadamard tests or related techniques for overlaps

This provides a practical quantum algorithm that can run on
near-term quantum hardware with shot noise.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from qiskit_aer import AerSimulator
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import scipy.linalg as la


@dataclass
class SKQDResult:
    """Result from SKQD algorithm execution."""
    converged: bool
    iterations_to_converge: int
    final_energy: float
    final_residual_norm: float

    energy_history: List[float] = field(default_factory=list)
    residual_history: List[float] = field(default_factory=list)

    # SKQD-specific metrics
    energy_std_history: List[float] = field(default_factory=list)
    total_shots: int = 0
    circuit_depths: List[int] = field(default_factory=list)

    method: str = "SKQD"


def create_hadamard_test_circuit(
    state_circuit: QuantumCircuit,
    operator: SparsePauliOp,
    num_qubits: int,
    real_part: bool = True
) -> QuantumCircuit:
    """
    Create Hadamard test circuit for estimating ⟨ψ|O|ψ⟩.

    The Hadamard test uses an ancilla qubit to extract
    real or imaginary parts of expectation values.

    Args:
        state_circuit: Circuit that prepares |ψ⟩
        operator: Operator O to measure
        num_qubits: Number of system qubits
        real_part: If True, measure real part; else imaginary part

    Returns:
        Hadamard test circuit
    """
    # Create registers
    ancilla = QuantumRegister(1, 'anc')
    system = QuantumRegister(num_qubits, 'sys')
    c_anc = ClassicalRegister(1, 'c_anc')

    qc = QuantumCircuit(ancilla, system, c_anc)

    # Hadamard on ancilla
    qc.h(ancilla[0])

    # Prepare state on system qubits
    qc.compose(state_circuit, qubits=system, inplace=True)

    # For imaginary part, add S gate
    if not real_part:
        qc.sdg(ancilla[0])

    # Controlled-O operation
    # For simplicity, we implement this for Pauli operators
    for pauli_str, coeff in operator.to_list():
        if pauli_str == "I" * num_qubits:
            continue
        # Create controlled Pauli gate
        for i, p in enumerate(pauli_str[::-1]):  # Reverse for Qiskit ordering
            if p == 'X':
                qc.cx(ancilla[0], system[i])
            elif p == 'Y':
                qc.cy(ancilla[0], system[i])
            elif p == 'Z':
                qc.cz(ancilla[0], system[i])

    # Final Hadamard
    qc.h(ancilla[0])

    # Measure ancilla
    qc.measure(ancilla[0], c_anc[0])

    return qc


def estimate_expectation_sampling(
    state: np.ndarray,
    hamiltonian: SparsePauliOp,
    shots: int = 8192,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Estimate expectation value using shot-based sampling.

    Measures each Pauli term separately and combines results.

    Args:
        state: State vector
        hamiltonian: SparsePauliOp
        shots: Number of shots per term
        seed: Random seed

    Returns:
        Tuple of (expectation, standard_error)
    """
    num_qubits = int(np.log2(len(state)))
    expectation = 0.0
    variance_sum = 0.0

    backend = AerSimulator()
    if seed is not None:
        backend.set_options(seed_simulator=seed)

    for pauli_str, coeff in hamiltonian.to_list():
        if pauli_str == "I" * num_qubits:
            expectation += coeff.real
            continue

        # Create measurement circuit
        qc = QuantumCircuit(num_qubits, num_qubits)
        qc.initialize(state, range(num_qubits))

        # Rotate to Pauli eigenbasis
        for i, p in enumerate(pauli_str[::-1]):
            if p == 'X':
                qc.h(i)
            elif p == 'Y':
                qc.sdg(i)
                qc.h(i)

        # Measure
        qc.measure(range(num_qubits), range(num_qubits))

        # Run
        job = backend.run(qc, shots=shots)
        counts = job.result().get_counts()

        # Compute expectation from parity
        term_exp = 0.0
        for bitstring, count in counts.items():
            # Compute parity of measured qubits with non-I Paulis
            parity = 0
            for i, p in enumerate(pauli_str[::-1]):
                if p != 'I':
                    bit_idx = num_qubits - 1 - i
                    if bit_idx < len(bitstring):
                        parity ^= int(bitstring[bit_idx])
            sign = 1 - 2 * parity
            term_exp += sign * count

        term_exp /= shots
        expectation += coeff.real * term_exp
        variance_sum += (coeff.real ** 2) * (1 - term_exp ** 2) / shots

    std_error = np.sqrt(variance_sum) if variance_sum > 0 else 0.0
    return float(expectation), float(std_error)


def prepare_krylov_states_skqd(
    hamiltonian: SparsePauliOp,
    initial_state: np.ndarray,
    num_krylov: int,
    evolution_time: float = 0.1,
    trotter_steps: int = 2,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Prepare Krylov states using quantum time evolution.

    This is similar to KQD but prepares states for sampling-based
    matrix element estimation.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        initial_state: Initial state
        num_krylov: Number of Krylov vectors
        evolution_time: Time step
        trotter_steps: Trotter steps
        seed: Random seed

    Returns:
        Tuple of (krylov_states, circuit_depths)
    """
    num_qubits = int(np.log2(len(initial_state)))
    krylov_states = [initial_state.copy()]
    circuit_depths = [0]

    backend = AerSimulator(method='statevector')
    if seed is not None:
        backend.set_options(seed_simulator=seed)

    current_state = initial_state.copy()

    for k in range(1, num_krylov):
        # Create evolution circuit
        qc = QuantumCircuit(num_qubits)
        qc.initialize(current_state, range(num_qubits))

        # Time evolution
        evo_gate = PauliEvolutionGate(
            hamiltonian,
            time=evolution_time,
            synthesis=SuzukiTrotter(order=2, reps=trotter_steps)
        )
        qc.append(evo_gate, range(num_qubits))
        qc.save_statevector()

        # Transpile to decompose PauliEvolutionGate into basic gates
        qc_transpiled = transpile(qc, backend, optimization_level=0)

        job = backend.run(qc_transpiled, shots=1)
        evolved = np.asarray(job.result().get_statevector())

        norm = np.linalg.norm(evolved)
        if norm > 1e-10:
            evolved = evolved / norm

        krylov_states.append(evolved)
        circuit_depths.append(qc_transpiled.depth())
        current_state = evolved

    return krylov_states, circuit_depths


def estimate_matrix_elements_skqd(
    hamiltonian: SparsePauliOp,
    krylov_states: List[np.ndarray],
    shots: int = 8192,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate projected matrices using sampling.

    This is the key SKQD innovation - using shot-based estimation
    instead of exact statevector computation.

    Args:
        hamiltonian: SparsePauliOp
        krylov_states: Krylov basis states
        shots: Shots per estimate
        seed: Random seed

    Returns:
        Tuple of (H_proj, S_proj, H_std) matrices
    """
    k = len(krylov_states)
    H_proj = np.zeros((k, k), dtype=complex)
    S_proj = np.zeros((k, k), dtype=complex)
    H_std = np.zeros((k, k))

    # Get Hamiltonian matrix for off-diagonal elements
    H_matrix = hamiltonian.to_matrix()
    if hasattr(H_matrix, 'toarray'):
        H_matrix = H_matrix.toarray()
    H_matrix = np.asarray(H_matrix)

    for i in range(k):
        for j in range(k):
            # Overlap (computed exactly for stability)
            S_proj[i, j] = np.vdot(krylov_states[i], krylov_states[j])

            # Hamiltonian diagonal elements via sampling
            if i == j:
                h_val, h_err = estimate_expectation_sampling(
                    krylov_states[i], hamiltonian, shots,
                    seed=(seed + i * k + j) if seed else None
                )
                H_proj[i, j] = h_val
                H_std[i, j] = h_err
            else:
                # Off-diagonal: use exact for now
                # (Full SKQD would use swap test or related)
                H_proj[i, j] = np.vdot(krylov_states[i], H_matrix @ krylov_states[j])

    return H_proj, S_proj, H_std


def orthonormalize(states: List[np.ndarray]) -> List[np.ndarray]:
    """Modified Gram-Schmidt orthonormalization."""
    ortho = []
    for v in states:
        v_ortho = v.copy()
        for u in ortho:
            v_ortho = v_ortho - np.vdot(u, v_ortho) * u
        norm = np.linalg.norm(v_ortho)
        if norm > 1e-10:
            ortho.append(v_ortho / norm)
    return ortho


def solve_generalized_eigenvalue(
    H_proj: np.ndarray,
    S_proj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem."""
    S_reg = S_proj + 1e-10 * np.eye(len(S_proj))
    try:
        eigenvalues, eigenvectors = la.eigh(H_proj, S_reg)
    except la.LinAlgError:
        S_inv = np.linalg.pinv(S_reg)
        eigenvalues, eigenvectors = np.linalg.eigh(S_inv @ H_proj)

    idx = np.argsort(eigenvalues.real)
    return eigenvalues[idx].real, eigenvectors[:, idx]


def compute_ritz_vector(
    krylov_states: List[np.ndarray],
    coefficients: np.ndarray
) -> np.ndarray:
    """Compute Ritz vector."""
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
    """Compute residual norm."""
    residual = H_matrix @ ritz_vector - ritz_energy * ritz_vector
    return float(np.linalg.norm(residual))


def run_skqd(
    hamiltonian: SparsePauliOp,
    initial_state: Optional[np.ndarray] = None,
    max_krylov_dim: int = 20,
    residual_tolerance: float = 1e-6,
    evolution_time: float = 0.1,
    trotter_steps: int = 2,
    shots: int = 8192,
    seed: Optional[int] = None,
    verbose: bool = False
) -> SKQDResult:
    """
    Run Sample-based Krylov Quantum Diagonalization.

    This combines Krylov subspace methods with shot-based sampling,
    representing a practical quantum algorithm for near-term devices.

    Args:
        hamiltonian: SparsePauliOp Hamiltonian
        initial_state: Initial state (default: |0...0⟩)
        max_krylov_dim: Maximum Krylov dimension
        residual_tolerance: Convergence tolerance
        evolution_time: Time step for evolution
        trotter_steps: Trotter steps
        shots: Shots for sampling
        seed: Random seed
        verbose: Print iteration progress

    Returns:
        SKQDResult with convergence info
    """
    num_qubits = hamiltonian.num_qubits
    dim = 2 ** num_qubits

    if initial_state is None:
        initial_state = np.zeros(dim, dtype=complex)
        initial_state[0] = 1.0

    # Get Hamiltonian matrix
    H_matrix = hamiltonian.to_matrix()
    if hasattr(H_matrix, 'toarray'):
        H_matrix = H_matrix.toarray()
    H_matrix = np.asarray(H_matrix)

    # Results tracking
    energy_history = []
    residual_history = []
    energy_std_history = []
    all_circuit_depths = []
    total_shots = 0

    converged = False
    final_energy = 0.0
    final_residual = float('inf')

    for k in range(2, max_krylov_dim + 1):
        # Generate Krylov states
        krylov_states, depths = prepare_krylov_states_skqd(
            hamiltonian, initial_state, k,
            evolution_time, trotter_steps, seed
        )
        all_circuit_depths.extend(depths)

        # Orthonormalize
        krylov_states = orthonormalize(krylov_states)

        if len(krylov_states) < 2:
            continue

        # Estimate matrix elements with sampling
        H_proj, S_proj, H_std = estimate_matrix_elements_skqd(
            hamiltonian, krylov_states, shots, seed
        )
        total_shots += shots * len(krylov_states)

        # Solve eigenvalue problem
        eigenvalues, eigenvectors = solve_generalized_eigenvalue(H_proj, S_proj)

        # Get ground state
        ritz_energy = eigenvalues[0]
        ritz_coeffs = eigenvectors[:, 0]
        ritz_vector = compute_ritz_vector(krylov_states, ritz_coeffs)

        # Compute residual
        residual_norm = compute_residual(H_matrix, ritz_vector, ritz_energy)

        # Estimate energy uncertainty from sampling
        energy_std = np.sqrt(np.sum(np.abs(ritz_coeffs) ** 2 * np.diag(H_std) ** 2))

        energy_history.append(float(ritz_energy))
        residual_history.append(residual_norm)
        energy_std_history.append(float(energy_std))

        final_energy = ritz_energy
        final_residual = residual_norm

        if verbose:
            log_res = np.log10(residual_norm) if residual_norm > 0 else -15
            print(f"  iter {k - 1:3d} | E = {ritz_energy:12.6f} eV | "
                  f"residual = {residual_norm:.2e} (log={log_res:.1f}) | "
                  f"krylov_dim = {len(krylov_states):3d} | E_std = {energy_std:.2e}")

        if residual_norm < residual_tolerance:
            converged = True
            break

    return SKQDResult(
        converged=converged,
        iterations_to_converge=len(energy_history),
        final_energy=float(final_energy),
        final_residual_norm=final_residual,
        energy_history=energy_history,
        residual_history=residual_history,
        energy_std_history=energy_std_history,
        total_shots=total_shots,
        circuit_depths=all_circuit_depths,
        method="SKQD"
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from src.hamiltonians.donor_valley import build_isolated_hamiltonian

    print("Testing SKQD implementation...")

    # Build Hamiltonian
    H = build_isolated_hamiltonian(12.0)
    print(f"Hamiltonian: {H.num_qubits} qubits, {len(H)} terms")

    # Random initial state
    np.random.seed(42)
    dim = 2 ** H.num_qubits
    init = np.random.randn(dim) + 1j * np.random.randn(dim)
    init /= np.linalg.norm(init)

    # Run SKQD
    result = run_skqd(
        H,
        initial_state=init,
        max_krylov_dim=10,
        shots=4096,
        seed=42
    )

    print(f"\nSKQD Results:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations_to_converge}")
    print(f"  Final energy: {result.final_energy:.6f} eV")
    print(f"  Final residual: {result.final_residual_norm:.2e}")
    print(f"  Total shots: {result.total_shots}")
    if result.energy_std_history:
        print(f"  Final energy std: {result.energy_std_history[-1]:.2e}")
