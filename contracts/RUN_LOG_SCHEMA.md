```md
# RUN_LOG_SCHEMA.md
**Canonical Run Log Schema for Donor Calibration (SKQD / Qiskit-Aer)**

This document defines the **authoritative JSON schema** for all run logs produced during the donor calibration experiments (Si:P vs Si:Bi).

Every execution of the calibration pipeline **must** emit exactly one run log conforming to this schema.
Runs that do not conform are considered **invalid** and may not be used for figures, tables, or proposal claims.

This schema is intentionally verbose to ensure:
- reproducibility,
- auditability,
- reviewer-defensibility,
- and post hoc analysis without reruns.

---

## 1. File Naming Convention

Each run log must be saved as:

```

results/
run_<material>*<active_space>*<timestamp>.json

```

Example:
```

run_SiP_full_2025-03-18T14-22-05.json

````

---

## 2. Top-Level Schema (Required Fields)

```json
{
  "run_id": "string",
  "timestamp_utc": "ISO-8601 string",
  "git_commit_hash": "string",

  "experiment_type": "donor_calibration",
  "material_system": "Si:P | Si:Bi",
  "active_space_type": "isolated | full",

  "basis_description": "string",
  "is_eigenbasis": false,

  "qubit_count": "integer",
  "spatial_orbital_count": "integer",
  "spin_orbital_count": "integer",

  "hamiltonian_spec": { },
  "simulation_backend": { },
  "krylov_parameters": { },

  "convergence_results": { },
  "timing_results": { },
  "sampling_results": { },

  "control_flags": { },
  "status": "success | failed",
  "failure_reason": "string | null"
}
````

---

## 3. Field-by-Field Specification

### 3.1 Run Identification

```json
"run_id": "UUID or human-readable unique string",
"timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ",
"git_commit_hash": "full or short commit hash"
```

**Purpose**

* Enables exact reproduction of the run.
* Ties numerical results to a specific code state.

---

### 3.2 Experiment Classification

```json
"experiment_type": "donor_calibration",
"material_system": "Si:P",
"active_space_type": "full"
```

**Constraints**

* `experiment_type` must always be `"donor_calibration"` for this repo.
* `active_space_type`:

  * `"isolated"` → 2-qubit A₁-only proxy
  * `"full"` → 12-qubit valley manifold

---

### 3.3 Basis Declaration (Critical Control)

```json
"basis_description": "Six-valley basis (±kx, ±ky, ±kz) with spin",
"is_eigenbasis": false
```

**Hard Rule**

* `is_eigenbasis` **must be false** for all production runs.
* Any run with `true` here is a **control-only diagnostic** and must not be used for figures.

---

### 3.4 Hilbert Space Size

```json
"qubit_count": 12,
"spatial_orbital_count": 6,
"spin_orbital_count": 12
```

**Consistency Checks**

* `spin_orbital_count = spatial_orbital_count × 2`
* `qubit_count = spin_orbital_count`

---

### 3.5 Hamiltonian Specification

```json
"hamiltonian_spec": {
  "representation": "SparsePauliOp",
  "term_count": "integer",
  "energy_units": "eV",

  "valley_orbit_splitting_meV": "float",
  "central_cell_parameter": "float",

  "diagonal_terms_only": false,
  "notes": "string"
}
```

**Purpose**

* Encodes which physical regime is being modeled (P vs Bi).
* Allows downstream grouping by effective gap size.

---

### 3.6 Simulation Backend (Qiskit / Aer)

```json
"simulation_backend": {
  "backend_name": "AerSimulator",
  "method": "statevector | density_matrix",
  "shots_per_expectation": "integer",
  "seed_simulator": "integer",
  "seed_transpiler": "integer"
}
```

**Constraints**

* At least one configuration **must** use finite shots (not pure statevector).
* Seeds must be logged explicitly.

---

### 3.7 Krylov / SKQD Parameters

```json
"krylov_parameters": {
  "max_iterations": "integer",
  "residual_tolerance": 1e-6,
  "initial_state_type": "valley_localized | random | symmetry_adapted",
  "orthogonalization": "full | partial | none"
}
```

**Purpose**

* Makes convergence claims falsifiable.
* Ensures iteration counts are comparable across runs.

---

### 3.8 Convergence Results (Primary Data)

```json
"convergence_results": {
  "iterations_to_converge": "integer",
  "converged": true,

  "final_ritz_energy": "float",
  "ritz_energy_history": ["float", "..."],

  "residual_norm_history": ["float", "..."],
  "log_residual_slope": "float",

  "convergence_penalty": "float | null"
}
```

**Definitions**

* `iterations_to_converge`:

  * smallest k such that ‖rₖ‖ < tolerance
* `convergence_penalty`:

  * only populated for `active_space_type = full`
  * computed as:
    [
    \Delta N = N_{\text{full}} - N_{\text{isolated}}
    ]

---

### 3.9 Timing Results

```json
"timing_results": {
  "total_wall_time_sec": "float",
  "time_per_iteration_sec": ["float", "..."]
}
```

**Purpose**

* Secondary diagnostic only.
* Used to verify scaling sanity, not physics.

---

### 3.10 Sampling / Noise Diagnostics

```json
"sampling_results": {
  "expectation_estimator": "PauliSampling | Exact",
  "energy_std_per_iteration": ["float", "..."],
  "residual_std_per_iteration": ["float", "..."]
}
```

**Purpose**

* Quantifies shot-noise-induced volatility.
* Supports claims about SKQD robustness under sampling noise.

---

### 3.11 Control Flags

```json
"control_flags": {
  "eigenbasis_control": false,
  "initial_state_sensitivity_test": false,
  "basis_rotation_test": false
}
```

**Purpose**

* Allows clean filtering of control vs production runs.

---

### 3.12 Run Status

```json
"status": "success",
"failure_reason": null
```

If `status = "failed"`:

* `failure_reason` must be a short diagnostic string
* Partial data may be present but run is excluded from analysis

---

## 4. Minimal Valid Example (Skeleton)

```json
{
  "run_id": "run_001",
  "timestamp_utc": "2025-03-18T14:22:05Z",
  "git_commit_hash": "a1b2c3d",

  "experiment_type": "donor_calibration",
  "material_system": "Si:P",
  "active_space_type": "full",

  "basis_description": "Six-valley basis with spin",
  "is_eigenbasis": false,

  "qubit_count": 12,
  "spatial_orbital_count": 6,
  "spin_orbital_count": 12,

  "hamiltonian_spec": {
    "representation": "SparsePauliOp",
    "term_count": 84,
    "energy_units": "eV",
    "valley_orbit_splitting_meV": 12.0,
    "central_cell_parameter": 1.0,
    "diagonal_terms_only": false,
    "notes": "Si:P parameterization"
  },

  "simulation_backend": {
    "backend_name": "AerSimulator",
    "method": "statevector",
    "shots_per_expectation": 8192,
    "seed_simulator": 42,
    "seed_transpiler": 42
  },

  "krylov_parameters": {
    "max_iterations": 80,
    "residual_tolerance": 1e-6,
    "initial_state_type": "valley_localized",
    "orthogonalization": "full"
  },

  "convergence_results": {
    "iterations_to_converge": 46,
    "converged": true,
    "final_ritz_energy": -0.0453,
    "ritz_energy_history": [],
    "residual_norm_history": [],
    "log_residual_slope": -0.31,
    "convergence_penalty": 18
  },

  "timing_results": {
    "total_wall_time_sec": 12.7,
    "time_per_iteration_sec": []
  },

  "sampling_results": {
    "expectation_estimator": "PauliSampling",
    "energy_std_per_iteration": [],
    "residual_std_per_iteration": []
  },

  "control_flags": {
    "eigenbasis_control": false,
    "initial_state_sensitivity_test": false,
    "basis_rotation_test": false
  },

  "status": "success",
  "failure_reason": null
}
```

---

## 5. Enforcement Rule

Any figure, table, or claim included in the proposal **must be traceable** to one or more run logs that:

* conform to this schema,
* have `status = "success"`,
* and satisfy the control constraints defined above.

This schema is part of the scientific contract of the project.

