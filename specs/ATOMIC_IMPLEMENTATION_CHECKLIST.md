# ATOMIC_IMPLEMENTATION_CHECKLIST.md
# Atomic Implementation Checklist: Donor Calibration (Si:P vs Si:Bi) in Qiskit/Aer
# VERSION 2.0 — Updated 2025-12-15
# Previous version archived at: archive/checklist_versions/ATOMIC_IMPLEMENTATION_CHECKLIST_v1_20251215.md

---

## STATUS SUMMARY

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 0 | Environment & Contract | ✅ COMPLETE | Repository structure verified |
| Phase 1 | Isolated Hamiltonians | ✅ COMPLETE | 2-qubit Si:P/Si:Bi working |
| Phase 2 | Full Valley Hamiltonians | ✅ COMPLETE | 12-qubit implemented |
| Phase 3 | Krylov Loop | ✅ COMPLETE | Classical + quantum variants |
| Phase 4 | Residual & Metrics | ✅ COMPLETE | All logging functional |
| Phase 5 | Metric Logging | ✅ COMPLETE | Verbose iteration output |
| Phase 6 | Figure Generation | 🔲 PENDING | Not yet started |
| Phase 7 | Reproducibility | 🔲 PENDING | Partial (seeds work) |
| **Phase 8** | **Algorithm Extensions** | ✅ COMPLETE | **NEW: SQD, SKQD, Classical-SBD** |
| **Phase 9** | **Larger System Scaling** | 🔲 PENDING | **NEW: Future work** |

---

## Phase 0 — Environment & Contract Validation ✅ COMPLETE

### Step 0.1 — Verify repository structure ✅
- **Status:** DONE
- **Artifact:** Directory tree matches spec

### Step 0.2 — Lock global numerical contracts ✅
- **Status:** DONE
- **Contracts:**
  - Residual tolerance: `1e-6` (configurable via `--tolerance`)
  - Max iterations: `20` (configurable via `--max-iter`)
  - Log residuals at every iteration: YES (with `-v` flag)
  - Basis: valley basis (non-eigen)

---

## Phase 1 — Hamiltonian Construction (Isolated Baseline) ✅ COMPLETE

### Step 1.1 — Implement 2-qubit A1-only Hamiltonian ✅
- **Status:** DONE
- **Location:** `src/hamiltonians/donor_valley.py::build_isolated_hamiltonian()`
- **Test results:**
  - Si:P (VO=11.7 meV): E₀ = -0.045590 eV ✅
  - Si:Bi (VO=60.0 meV): E₀ = -0.070980 eV ✅
- **Artifact:** Hamiltonians calibrated to experimental binding energies

### Step 1.2 — Negative control: eigenbasis trivialization ✅
- **Status:** DONE
- **Result:** Classical Krylov converges in 3 iterations (dimension 4 = full Hilbert space)

---

## Phase 2 — Hamiltonian Construction (Full Valley Manifold) ✅ COMPLETE

### Step 2.1 — Implement 12-qubit valley-basis Hamiltonian ✅
- **Status:** DONE
- **Location:** `src/hamiltonians/donor_valley.py::build_full_hamiltonian()`
- **Test:** Spectrum shows correct gap structure

### Step 2.2 — Validate non-eigen basis ✅
- **Status:** DONE
- **Result:** Off-diagonal Pauli terms present

---

## Phase 3 — Krylov Loop Implementation ✅ COMPLETE

### Step 3.1 — Implement Krylov state generation ✅
- **Status:** DONE
- **Location:** `src/krylov/krylov_loop.py`
- **Test:** Norms verified ≈ 1

### Step 3.2 — Implement projected matrix estimation ✅
- **Status:** DONE
- **Implementations:**
  - Classical (exact): `src/krylov/krylov_loop.py`
  - KQD (quantum evolution): `src/quantum_algorithms/kqd.py`
  - SKQD (shot-based): `src/quantum_algorithms/skqd.py`

### Step 3.3 — Solve generalized eigenproblem ✅
- **Status:** DONE
- **Test:** Ritz values decrease monotonically

---

## Phase 4 — Residual & Metric Extraction ✅ COMPLETE

### Step 4.1 — Compute residual vector ✅
- **Status:** DONE
- **Location:** All algorithm implementations

### Step 4.2 — Termination condition ✅
- **Status:** DONE
- **Configurable:** `--tolerance` flag

---

## Phase 5 — Metric Logging ✅ COMPLETE

### Step 5.1 — Log primary metrics ✅
- **Status:** DONE
- **Metrics logged:**
  - N_iter (iterations to converge)
  - Energy history
  - Residual history (with `--verbose`)
  - Convergence status
  - Execution time

### Step 5.2 — Algorithm comparison ✅
- **Status:** DONE
- **Location:** `scripts/run_calibration_test.py`
- **Features:**
  - `--algorithm` flag to select: classical, classical-sbd, kqd, sqd, skqd, all
  - `--space` flag: isolated, full, all
  - Algorithm-specific pass/fail tolerances

---

## Phase 6 — Figure Generation 🔲 PENDING

### Step 6.1 — Generate residual decay plot 🔲
- **Status:** NOT STARTED
- **Spec:** `figures/calibration/fig_residual_decay_donors.pdf`

### Step 6.2 — Generate Ritz stabilization plot 🔲
- **Status:** NOT STARTED
- **Spec:** `figures/calibration/fig_ritz_stabilization_donors.pdf`

---

## Phase 7 — Reproducibility & Archival 🔲 PARTIAL

### Step 7.1 — Snapshot environment 🔲
- **Status:** PARTIAL
- **Done:** Seeds work, algorithms reproducible
- **TODO:** Environment JSON logging

### Step 7.2 — Archive specs ✅
- **Status:** DONE (this document)

---

## Phase 8 — Algorithm Extensions ✅ COMPLETE (NEW)

### Step 8.1 — Classical Krylov ✅
- **Status:** DONE
- **Location:** `src/krylov/krylov_loop.py::run_krylov_loop()`
- **Features:** Verbose iteration output, configurable tolerance

### Step 8.2 — Classical SBD (Subspace-based Diagonalization) ✅
- **Status:** DONE
- **Location:** `src/krylov/krylov_loop.py::run_classical_sbd()`
- **Features:** Enumerates computational basis, builds CI subspace

### Step 8.3 — KQD (Krylov Quantum Diagonalization) ✅
- **Status:** DONE
- **Location:** `src/quantum_algorithms/kqd.py::run_kqd()`
- **Features:**
  - Quantum time evolution via PauliEvolutionGate
  - Suzuki-Trotter decomposition
  - Transpilation for Aer compatibility

### Step 8.4 — SQD (Sample-based Quantum Diagonalization) ✅
- **Status:** DONE
- **Location:** `src/quantum_algorithms/sqd.py::run_sqd()`
- **Features:**
  - EfficientSU2 ansatz sampling
  - Hamiltonian seeding from diagonal (low-energy configs)
  - Transpilation for Aer compatibility
  - 20,000 shots per iteration

### Step 8.5 — SKQD (Sample-based Krylov QD) ✅
- **Status:** DONE
- **Location:** `src/quantum_algorithms/skqd.py::run_skqd()`
- **Features:**
  - Krylov subspace via quantum evolution
  - Shot-based matrix element estimation
  - 16,384 shots for reduced noise

---

## Phase 9 — Larger System Scaling 🔲 FUTURE WORK (NEW)

### Step 9.1 — Symmetry-Projected Subspaces 🔲
- **Status:** PLANNED
- **Description:** Project onto states with correct quantum numbers (particle number, spin)
- **Benefit:** Dramatically reduces Hilbert space

### Step 9.2 — Selected CI / Adaptive Sampling 🔲
- **Status:** PLANNED
- **Description:** Iteratively grow CI subspace with important configurations
- **Reference:** CIPSI, ASCI, SHCI algorithms

### Step 9.3 — Perturbative Corrections (SQD + PT2) 🔲
- **Status:** PLANNED
- **Description:** Use SQD subspace for zeroth order, add PT2 corrections
- **Benefit:** Recovers correlation energy from excluded space

### Step 9.4 — Tensor Network Subspace Methods 🔲
- **Status:** PLANNED
- **Description:** MPS/DMRG-like structures for 1D-ish systems
- **Benefit:** Polynomial scaling for area-law states

### Step 9.5 — Multi-Reference Starting Points 🔲
- **Status:** PLANNED
- **Description:** Run from multiple initial states in parallel
- **Benefit:** Explores disconnected important regions

---

## Current Test Results (Isolated Systems)

```
Algorithm       | Si:P Status | Si:Bi Status | Error Tolerance
----------------|-------------|--------------|----------------
Classical       | ✅ PASS     | ✅ PASS      | < 1e-8 eV
Classical-SBD   | ✅ PASS     | ✅ PASS      | < 1e-8 eV
KQD             | ✅ PASS     | ✅ PASS      | < 1e-8 eV
SQD             | ✅ PASS     | ✅ PASS      | < 1e-8 eV (H-seeded)
SKQD            | ✅ PASS     | ✅ PASS      | < 0.5 meV (shot noise)
```

---

## Command Reference

```bash
# Run all algorithms on isolated systems
python scripts/run_calibration_test.py -a all --space isolated -v

# Run specific algorithm
python scripts/run_calibration_test.py -a classical --space full

# Override hyperparameters
python scripts/run_calibration_test.py -a kqd --max-iter 30 --tolerance 1e-8 -v

# Algorithm choices: classical, classical-sbd, kqd, sqd, skqd, all
# Space choices: isolated, full, all
```

---

## Key Improvements Made (Session Summary)

1. **Added `--algorithm` flag** to select which algorithms to run
2. **Added `--max-iter` and `--tolerance` flags** for hyperparameter override
3. **Fixed Aer transpilation** for PauliEvolutionGate (KQD, SKQD) and EfficientSU2 (SQD)
4. **Added verbose iteration output** to all quantum algorithms
5. **Implemented Hamiltonian seeding for SQD** - seeds CI subspace with low-energy configs from H diagonal
6. **Increased shots** for sampling methods (SQD: 20K, SKQD: 16K)
7. **Added algorithm-specific tolerances** (0.5 meV for SKQD due to shot noise)

---

## Next Steps (Recommended Priority)

1. **Phase 6**: Generate publication-quality figures
2. **Phase 7**: Complete environment snapshotting
3. **Test on 12-qubit full systems**: Validate algorithm scaling
4. **Phase 9**: Implement symmetry projection for larger systems
