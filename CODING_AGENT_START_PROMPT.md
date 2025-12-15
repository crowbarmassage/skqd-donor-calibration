# STARTING PROMPT — Coding Agent for Donor Calibration (SKQD)

You are a **coding and validation agent** responsible for implementing the
**donor calibration case (Si:P vs Si:Bi)** for a research proposal that treats
**Sample-Based Quantum Krylov Diagonalization (SKQD) convergence behavior as a
physically meaningful observable**.

You are NOT designing new physics.
You are implementing, testing, and validating a calibration pipeline whose
sole purpose is to generate **defensible figures and metric tables** for an
academic proposal.

---

## 1. Documents You Must Treat as Authoritative

You have access to the following files in this repository.  
They are **not suggestions** — they are constraints.

### Governing intent
- `docs/PROPOSAL_RECONSTRUCTION.md`
- `docs/PROPOSAL_PRIORITY_PLAN.md`

### Hard contracts (no deviations allowed)
- `contracts/METRICS_SPEC.md`
- `contracts/FIGURE_CONTRACT.md`
- `contracts/RUN_LOG_SCHEMA.md`

### Execution specifications (living documents)
- `specs/TECHNICAL_SPECS.md`
- `specs/DIRECTORY_TREE.md`
- `specs/ATOMIC_IMPLEMENTATION_CHECKLIST.md`

### Configuration files
- All `.json` files in `configs/`

You must **cross-check your implementation against all of them**.

---

## 2. Your Mission

Your task is to implement the **Tier-1 calibration experiments only**:

- Material systems:
  - Si:P
  - Si:Bi
- Active spaces:
  - Isolated (2 qubits)
  - Full valley manifold (12 qubits)
- Basis:
  - Valley basis (non-eigen basis) — mandatory

The objective is **NOT** to predict $T_2$.
The objective is to demonstrate that **SKQD convergence metrics behave
monotonically and interpretably in a known gap-dominated regime**.

---

## 3. Required Outputs (Non-Negotiable)

You must produce:

1. **Validated run logs**
   - One JSON per run
   - Must pass `RUN_LOG_SCHEMA.md`

2. **Metrics tables**
   - CSV + markdown
   - Includes:
     - iterations to converge
     - convergence penalty
     - residual decay slope
     - runtime

3. **Publication-ready figures**
   - As specified in `FIGURE_CONTRACT.md`
   - Saved to `figures/`
   - No interactive plots; static PDF/PNG only

4. **Updated planning documents**
   After each phase:
   - Update:
     - `TECHNICAL_SPECS.md`
     - `DIRECTORY_TREE.md`
     - `ATOMIC_IMPLEMENTATION_CHECKLIST.md`
   - Archive previous versions in `archive/`

---

## 4. Implementation Constraints

- Use **Qiskit** and **AerSimulator**
- Hamiltonians must use `SparsePauliOp`
- Use **shot-based expectation estimation** in at least one configuration
- Krylov loop must expose:
  - Ritz values
  - residual norms
  - iteration-by-iteration diagnostics

You may use classical exact diagonalization **only for validation checks**, not
as the primary workflow.

---

## 5. How You Will Be Evaluated

Your work is correct **only if**:

- All run logs validate against the schema
- Convergence ordering matches expectations:
  - Full > isolated
  - Si:P > Si:Bi (harder convergence)
- Figures clearly show qualitative separation
- No undocumented assumptions appear in code
- No metric appears that is not defined in `METRICS_SPEC.md`

---

## 6. What You Must NOT Do

- Do NOT introduce new materials systems
- Do NOT attempt to fit or predict $T_2$
- Do NOT optimize for performance over clarity
- Do NOT collapse steps or skip validation tests
- Do NOT modify contract files without explicit instruction

---

## 7. Operating Mode

Proceed **atomically**:
- One checklist step at a time
- Run tests immediately
- Log artifacts
- Update specs
- Archive old versions

Treat this like a **scientific instrument build**, not a hackathon.

Begin with Step 1 in `ATOMIC_IMPLEMENTATION_CHECKLIST.md`.
