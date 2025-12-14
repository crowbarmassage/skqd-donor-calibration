# TECHNICAL_SPECS.md
# UPDATED TECHNICAL SPECIFICATION — Donor Calibration (Si:P vs Si:Bi)

## Status
This specification is **synchronized** with:
- METRICS_SPEC.md
- FIGURE_CONTRACT.md
- ATOMIC_IMPLEMENTATION_CHECKLIST.md

Any deviation invalidates calibration results.

---

## Scientific Scope (Locked)

- Tier: Calibration only (Tier 1)
- Mechanism: Gap-dominated valley-orbit splitting
- Purpose: Interpretability validation, not discovery
- Claim boundary:
  - Qualitative ordering only
  - No quantitative T₂ prediction

---

## Models

### Systems
- Si:P (small valley-orbit splitting)
- Si:Bi (large valley-orbit splitting)

### Active Spaces
| Case | Qubits | Description |
|----|----|----|
| Isolated | 2 | A₁ orbital, spin only |
| Full | 12 | 6 valleys × spin |

### Basis
- Valley basis (mandatory)
- Eigenbasis used only as negative control

---

## Hamiltonian Representation

- Qiskit `SparsePauliOp`
- Single-electron effective Hamiltonian
- Valley mixing terms parameterized per donor
- No explicit many-body interactions

---

## Algorithm

### Krylov Loop
- Sample-based Krylov-like iteration
- Shot-based expectation estimation
- Generalized eigenproblem at each iteration

### Termination
- Residual norm < 1e-6

---

## Metrics (Authoritative)
Defined exactly in `METRICS_SPEC.md`:
- N_iter
- ΔN_Krylov
- Residual history
- Residual decay slope
- Ritz stabilization

No additional metrics permitted in analysis.

---

## Outputs

### Logs
- One JSON per run
- Schema validated against METRICS_SPEC

### Tables
- CSV + markdown:
  - Si:P vs Si:Bi
  - isolated vs full

### Figures
- Exactly those in FIGURE_CONTRACT.md
- No exploratory figures included in proposal

---

## Success Criteria

Calibration is successful if:
1. N_iter(full, P) > N_iter(full, Bi)
2. ΔN_Krylov(P) > ΔN_Krylov(Bi)
3. Eigenbasis control converges trivially
4. Residual decay curves are monotonic and separable

---

## Non-Goals (Explicit)

- No noise operators
- No symmetry tests
- No scaling beyond 12 qubits
- No performance benchmarking claims

---

## Versioning & Archival Policy

- This file is updated after each phase
- Previous versions archived under:
  - `archive/tech_specs/`
- Changes must be atomic and documented
