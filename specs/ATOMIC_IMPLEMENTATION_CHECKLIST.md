# ATOMIC_IMPLEMENTATION_CHECKLIST.md
# Atomic Implementation Checklist: Donor Calibration (Si:P vs Si:Bi) in Qiskit/Aer
# UPDATED AFTER CROSS-CHECK WITH METRICS_SPEC.md and TECHNICAL_SPECS.md

## Phase 0 — Environment & Contract Validation

### Step 0.1 — Verify repository structure
- **Action:** Ensure directory tree matches `DIRECTORY_TREE.md`
- **Test (pass/fail):**
  - `tree calibration_donor_skqd/` matches spec exactly
- **Artifact:** none
- **Failure modes:**
  - Missing archive/ subdirs → create immediately
  - Misnamed files → rename before proceeding

---

### Step 0.2 — Lock global numerical contracts
- **Action:** Define and freeze:
  - residual tolerance = `1e-6`
  - log residuals at every Krylov iteration
  - basis = valley basis (non-eigen)
- **Test:**
  - Config loader prints these values at runtime start
- **Artifact:** `results/metadata/global_contract.json`
- **Failure modes:**
  - Tolerance mismatch across modules
  - Silent defaults in Krylov loop

---

## Phase 1 — Hamiltonian Construction (Isolated Baseline)

### Step 1.1 — Implement 2-qubit A1-only Hamiltonian
- **Action:** Build `SparsePauliOp` for isolated donor qubit
- **Test:**
  - Exact diagonalization yields single non-degenerate ground state
- **Artifact:** `results/metadata/hamiltonian_isolated.json`
- **Failure modes:**
  - Accidental valley mixing terms included
  - Wrong qubit indexing

---

### Step 1.2 — Negative control: eigenbasis trivialization
- **Action:** Diagonalize isolated Hamiltonian in eigenbasis
- **Test:**
  - Krylov converges in ≤2 iterations
- **Artifact:** `results/logs/control_eigenbasis_isolated.json`
- **Failure modes:**
  - Krylov loop incorrectly implemented
  - Residual computation bug

---

## Phase 2 — Hamiltonian Construction (Full Valley Manifold)

### Step 2.1 — Implement 12-qubit valley-basis Hamiltonian
- **Action:** Construct valley-mixing Hamiltonian for Si:P and Si:Bi
- **Test:**
  - Exact spectrum shows smaller gap for P than Bi
- **Artifact:** `results/metadata/hamiltonian_full_{P,Bi}.json`
- **Failure modes:**
  - Mixing strengths swapped
  - Valley indexing inconsistent

---

### Step 2.2 — Validate non-eigen basis
- **Action:** Confirm Hamiltonian is not diagonal in computational basis
- **Test:**
  - Off-diagonal Pauli terms present
- **Artifact:** none
- **Failure modes:**
  - Accidental basis diagonalization

---

## Phase 3 — Krylov Loop Implementation

### Step 3.1 — Implement Krylov state generation
- **Action:** Generate Krylov vectors |φ_k⟩ = H^k |φ_0⟩ (normalized)
- **Test:**
  - Norm of each |φ_k⟩ ≈ 1 (±1e-6)
- **Artifact:** `results/logs/krylov_norms.json`
- **Failure modes:**
  - Numerical instability
  - Missing normalization

---

### Step 3.2 — Implement projected matrix estimation (sampling)
- **Action:** Estimate H_ij and S_ij via AerSimulator
- **Test:**
  - H and S are Hermitian within tolerance
- **Artifact:** `results/logs/projected_matrices_k.json`
- **Failure modes:**
  - Shot noise too large
  - Incorrect Pauli expectation mapping

---

### Step 3.3 — Solve generalized eigenproblem
- **Action:** Solve Hc = ESc at each k
- **Test:**
  - Lowest Ritz value decreases monotonically
- **Artifact:** `results/logs/ritz_values.json`
- **Failure modes:**
  - Ill-conditioned S
  - Eigenvalue ordering bug

---

## Phase 4 — Residual & Metric Extraction

### Step 4.1 — Compute residual vector
- **Action:** Compute r_k = H|ψ_k⟩ − E_k|ψ_k⟩
- **Test:**
  - ||r_k|| decreases with k
- **Artifact:** `results/logs/residuals.json`
- **Failure modes:**
  - Ritz vector reconstruction error
  - H application bug

---

### Step 4.2 — Termination condition
- **Action:** Stop when ||r_k|| < 1e-6
- **Test:**
  - Termination iteration logged as N_iter
- **Artifact:** `results/logs/convergence_summary.json`
- **Failure modes:**
  - Early termination
  - Never terminating

---

## Phase 5 — Metric Logging (METRICS_SPEC Compliance)

### Step 5.1 — Log primary metrics
- **Action:** Record:
  - N_iter
  - residual history
  - Ritz stabilization
- **Test:**
  - JSON matches METRICS_SPEC field names exactly
- **Artifact:** `results/logs/run_{material}_{space}.json`
- **Failure modes:**
  - Missing fields
  - Inconsistent naming

---

### Step 5.2 — Compute convergence penalty
- **Action:** ΔN = N_full − N_isolated
- **Test:**
  - ΔN(P) > ΔN(Bi)
- **Artifact:** `results/tables/convergence_penalty.csv`
- **Failure modes:**
  - Wrong baseline association
  - Mis-labeled runs

---

## Phase 6 — Figure Generation (FIGURE_CONTRACT Compliance)

### Step 6.1 — Generate residual decay plot
- **Action:** Plot ||r_k|| vs k (log scale)
- **Test:**
  - Matches Figure 1 spec exactly
- **Artifact:** `figures/calibration/fig_residual_decay_donors.pdf`
- **Failure modes:**
  - Axis mismatch
  - Missing curves

---

### Step 6.2 — Generate Ritz stabilization plot
- **Action:** Plot |ΔE_k| vs k
- **Test:**
  - Si:P stabilizes later than Si:Bi
- **Artifact:** `figures/calibration/fig_ritz_stabilization_donors.pdf`
- **Failure modes:**
  - Incorrect differencing
  - Noise-dominated plot

---

## Phase 7 — Reproducibility & Archival

### Step 7.1 — Snapshot environment
- **Action:** Save:
  - Python version
  - Qiskit version
  - Seed values
- **Test:**
  - Metadata file present and complete
- **Artifact:** `results/metadata/environment.json`
- **Failure modes:**
  - Missing seed logging

---

### Step 7.2 — Archive specs
- **Action:** Move prior versions of:
  - checklist
  - TECHNICAL_SPECS
  - DIRECTORY_TREE
- **Test:**
  - Timestamped copies exist in archive/
- **Artifact:** archived md files
- **Failure modes:**
  - Overwrite without archive
