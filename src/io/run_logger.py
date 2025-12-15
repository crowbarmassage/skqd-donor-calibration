"""
Run logger for SKQD donor calibration.

Creates JSON run logs that comply with RUN_LOG_SCHEMA.md.
Each execution must emit exactly one run log per experiment.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_RAW_DIR = PROJECT_ROOT / "results" / "raw"


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]  # Short hash
    except Exception:
        pass
    return "unknown"


def generate_run_id(material: str, active_space: str) -> str:
    """Generate unique run ID."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"run_{material}_{active_space}_{timestamp}_{short_uuid}"


def create_run_log(
    config: dict,
    krylov_result: Any,  # KrylovResult
    hamiltonian_info: dict,
    timing_info: dict,
    is_eigenbasis_control: bool = False
) -> dict:
    """
    Create a run log dictionary compliant with RUN_LOG_SCHEMA.md.

    Args:
        config: Experiment configuration
        krylov_result: KrylovResult from run_krylov_loop
        hamiltonian_info: Hamiltonian metadata
        timing_info: Timing information
        is_eigenbasis_control: Whether this is an eigenbasis control run

    Returns:
        Run log dictionary ready for JSON serialization
    """
    material = config.get("system", "unknown")
    active_space = config.get("active_space", {}).get("type", "unknown")
    num_qubits = config.get("active_space", {}).get("num_qubits", 0)

    # Determine spatial/spin orbital counts
    if active_space == "isolated":
        spatial_orbitals = 1
        spin_orbitals = 2
    else:  # full
        spatial_orbitals = 6
        spin_orbitals = 12

    # Build basis description
    basis = config.get("active_space", {}).get("basis", "valley_basis")
    if active_space == "isolated":
        basis_desc = "A1-only orbital with spin"
    else:
        basis_desc = "Six-valley basis (±kx, ±ky, ±kz) with spin"

    run_log = {
        # Run identification
        "run_id": generate_run_id(material.replace(":", ""), active_space),
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit_hash": get_git_commit_hash(),

        # Experiment classification
        "experiment_type": "donor_calibration",
        "material_system": material,
        "active_space_type": active_space,

        # Basis declaration
        "basis_description": basis_desc,
        "is_eigenbasis": is_eigenbasis_control,

        # Hilbert space size
        "qubit_count": num_qubits,
        "spatial_orbital_count": spatial_orbitals,
        "spin_orbital_count": spin_orbitals,

        # Hamiltonian specification
        "hamiltonian_spec": {
            "representation": "SparsePauliOp",
            "term_count": hamiltonian_info.get("term_count", 0),
            "energy_units": "eV",
            "valley_orbit_splitting_meV": config.get("hamiltonian", {}).get("valley_orbit_splitting_meV"),
            "central_cell_parameter": config.get("hamiltonian", {}).get("off_diagonal_coupling_scale", 1.0),
            "diagonal_terms_only": False,
            "notes": f"{material} parameterization"
        },

        # Simulation backend
        "simulation_backend": {
            "backend_name": config.get("backend", {}).get("simulator", "AerSimulator"),
            "method": config.get("backend", {}).get("mode", "statevector"),
            "shots_per_expectation": config.get("backend", {}).get("shots"),
            "seed_simulator": config.get("random_seed"),
            "seed_transpiler": config.get("random_seed")
        },

        # Krylov parameters
        "krylov_parameters": {
            "max_iterations": config.get("krylov", {}).get("max_iterations", 50),
            "residual_tolerance": config.get("krylov", {}).get("residual_tolerance", 1e-6),
            "initial_state_type": "random",
            "orthogonalization": config.get("krylov", {}).get("orthogonalization", "modified_gram_schmidt")
        },

        # Convergence results (primary data per METRICS_SPEC)
        "convergence_results": {
            "iterations_to_converge": krylov_result.iterations_to_converge,
            "converged": krylov_result.converged,
            "final_ritz_energy": float(krylov_result.final_ritz_energy),
            "ritz_energy_history": [float(e) for e in krylov_result.ritz_energy_history],
            "residual_norm_history": [float(r) for r in krylov_result.residual_norm_history],
            "log_residual_slope": float(krylov_result.log_residual_slope) if krylov_result.log_residual_slope else None,
            "convergence_penalty": None  # Set later for full runs
        },

        # Timing results
        "timing_results": {
            "total_wall_time_sec": timing_info.get("total_time", 0.0),
            "time_per_iteration_sec": timing_info.get("per_iteration", [])
        },

        # Sampling results
        "sampling_results": {
            "expectation_estimator": "Exact" if config.get("backend", {}).get("mode") == "statevector" else "PauliSampling",
            "energy_std_per_iteration": [],
            "residual_std_per_iteration": []
        },

        # Control flags
        "control_flags": {
            "eigenbasis_control": is_eigenbasis_control,
            "initial_state_sensitivity_test": False,
            "basis_rotation_test": False
        },

        # Status
        "status": "success" if krylov_result.converged else "failed",
        "failure_reason": None if krylov_result.converged else "Did not converge within max iterations"
    }

    return run_log


def save_run_log(run_log: dict, output_dir: Optional[Path] = None) -> Path:
    """
    Save run log to JSON file.

    File naming per RUN_LOG_SCHEMA:
    results/raw/run_<material>_<active_space>_<timestamp>.json

    Args:
        run_log: Run log dictionary
        output_dir: Output directory (default: results/raw/)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = RESULTS_RAW_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_log['run_id']}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(run_log, f, indent=2)

    return filepath


def compute_convergence_penalty(
    n_iter_full: int,
    n_iter_isolated: int
) -> int:
    """
    Compute convergence penalty ΔN = N_full - N_isolated.

    Per METRICS_SPEC:
    - Large penalty → strong interference from environmental states
    - Small penalty → effective isolation
    """
    return n_iter_full - n_iter_isolated


if __name__ == "__main__":
    print("Run logger module loaded successfully")
    print(f"Git commit: {get_git_commit_hash()}")
    print(f"Output directory: {RESULTS_RAW_DIR}")
