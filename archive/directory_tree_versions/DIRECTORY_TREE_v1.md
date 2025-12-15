# DIRECTORY_TREE.md
**Canonical Repository Structure for Donor Calibration (SKQD / Qiskit-Aer)**

This document defines the **authoritative directory tree** for the donor calibration project.
All code, data, figures, logs, and specifications must conform to this structure.

Any deviation must be explicitly documented in the technical specs and justified.

This structure is designed to support:
- reproducibility,
- auditability,
- clean separation of concerns,
- iterative development with archival history,
- direct traceability from figures → logs → code → specs.

---

## 1. Top-Level Repository Layout

```
skqd-donor-calibration/
├── README.md
├── contracts/           # Hard contracts (frozen)
├── specs/               # Execution specifications (living documents)
├── docs/                # Proposal context and planning
├── configs/             # Experiment configuration files
├── src/                 # Source code
├── scripts/             # Entry point scripts
├── results/             # Run logs and tables
├── figures/             # Publication figures
├── notebooks/           # Exploration (non-authoritative)
├── archive/             # Version control beyond git
└── environment/         # Environment specifications
```

---

## 2. Contracts Directory (`contracts/`)

```
contracts/
├── METRICS_SPEC.md          # Definitions of all convergence metrics
├── FIGURE_CONTRACT.md       # Mapping: figures → metrics → proposal sections
└── RUN_LOG_SCHEMA.md        # JSON schema for all run logs
```

**Rules**
- These files define the **contract** between science, code, and figures.
- These are FROZEN - no modifications without explicit instruction.

---

## 3. Specifications Directory (`specs/`)

```
specs/
├── DIRECTORY_TREE.md                  # This file: canonical repo structure
├── TECHNICAL_SPECS.md                 # Technical specification of the calibration project
└── ATOMIC_IMPLEMENTATION_CHECKLIST.md # Step-by-step implementation plan
```

**Rules**
- These are living documents that evolve with implementation.
- Any change to methodology must update TECHNICAL_SPECS.md and be archived.

---

## 4. Documentation Directory (`docs/`)

```
docs/
├── PROPOSAL_RECONSTRUCTION.md    # Intellectual history and decision rationale
└── PROPOSAL_PRIORITY_PLAN.md     # Expansion strategy and priorities
```

**Rules**
- Context documents for the research proposal.
- Reference material for understanding project goals.

---

## 5. Archive Directory (`archive/`)

```
archive/
├── checklist_versions/
│   ├── ATOMIC_IMPLEMENTATION_CHECKLIST_v1.md
│   └── ...
├── tech_specs_versions/
│   ├── TECHNICAL_SPECS_v1.md
│   └── ...
├── directory_tree_versions/
│   ├── DIRECTORY_TREE_v1.md
│   └── ...
└── notes/
    ├── deprecated_ideas.md
    └── rationale_logs.md
```

**Purpose**
- Captures methodological evolution.
- Allows post hoc reconstruction of decisions (critical for reviewer questions).

---

## 6. Source Code (`src/`)

```
src/
├── __init__.py
├── hamiltonians/
│   ├── __init__.py
│   ├── donor_valley.py          # Valley-basis Hamiltonians (Si:P, Si:Bi)
│   └── basis_controls.py        # Eigenbasis / rotated basis controls
├── krylov/
│   ├── __init__.py
│   ├── krylov_loop.py           # Core SKQD-like iteration loop
│   ├── subspace_builder.py      # Krylov vector construction
│   ├── projected_matrices.py    # H_ij, S_ij estimation
│   └── residuals.py             # Residual computation and stopping criteria
├── estimation/
│   ├── __init__.py
│   ├── expectation.py           # Pauli expectation estimation
│   └── sampling.py              # Shot-based sampling utilities
├── analysis/
│   ├── __init__.py
│   ├── convergence_metrics.py   # N_iter, residual slope, Ritz stabilization
│   └── validation.py            # Consistency and sanity checks
├── io/
│   ├── __init__.py
│   ├── run_logger.py            # JSON log writer (RUN_LOG_SCHEMA compliant)
│   └── config_loader.py         # Load JSON configs
└── utils/
    ├── __init__.py
    ├── seeds.py
    ├── timing.py
    └── linear_algebra.py
```

**Rules**
- No plotting code in `src/`.
- No hard-coded parameters; all parameters come from `configs/`.

---

## 7. Configuration Files (`configs/`)

```
configs/
├── sip_isolated.json      # Si:P isolated (2-qubit) configuration
├── sip_full.json          # Si:P full valley manifold (12-qubit) configuration
├── sibi_isolated.json     # Si:Bi isolated (2-qubit) configuration
├── sibi_full.json         # Si:Bi full valley manifold (12-qubit) configuration
└── CODING_AGENT_START_PROMPT.md  # Instructions for implementation agent
```

**Rules**
- All numerical experiments must be reproducible from configs alone.
- Config files are immutable once used for a published figure.

---

## 8. Results (`results/`)

```
results/
├── raw/
│   ├── run_SiP_isolated_*.json
│   ├── run_SiP_full_*.json
│   ├── run_SiBi_isolated_*.json
│   └── run_SiBi_full_*.json
├── tables/
│   ├── donor_calibration_metrics.csv
│   └── donor_calibration_metrics.md
├── summaries/
│   ├── aggregation_report.json
│   └── aggregation_notes.md
└── metadata/
    ├── global_contract.json
    ├── hamiltonian_isolated.json
    ├── hamiltonian_full_P.json
    ├── hamiltonian_full_Bi.json
    └── environment.json
```

**Rules**
- Only RUN_LOG_SCHEMA-compliant JSON files allowed in `raw/`.
- Aggregated tables must cite exact run IDs.

---

## 9. Figures (`figures/`)

```
figures/
├── calibration/
│   ├── fig_residual_decay_donors.pdf
│   ├── fig_ritz_stabilization_donors.pdf
│   └── convergence_penalty_bar_chart.pdf
└── drafts/
    └── exploratory_plots/
```

**Rules**
- Only figures listed in FIGURE_CONTRACT.md may appear in the proposal.
- Draft figures must not be cited.

---

## 10. Scripts (`scripts/`)

```
scripts/
├── run_experiment.py            # Entry point for calibration runs
├── aggregate_results.py         # Build tables from run logs
├── generate_figures.py          # Produce publication-ready figures
└── validate_run_logs.py         # Schema + consistency validation
```

**Rules**
- Scripts are reproducible CLI tools.
- No logic duplication with `src/`.

---

## 11. Notebooks (`notebooks/`) — Optional / Non-Authoritative

```
notebooks/
├── exploration.ipynb
└── debugging.ipynb
```

**Rules**
- Notebooks are for development only.
- No figure or result may depend on a notebook.

---

## 12. Environment Specification (`environment/`)

```
environment/
├── requirements.txt
├── environment.yml              # Conda environment (optional)
└── system_snapshot.txt          # Python, OS, Qiskit versions
```

**Rules**
- Every run log must be reproducible under one environment snapshot.

---

## 13. Invariants (Non-Negotiable)

- Every figure → references run logs → reference configs → reference code.
- Every methodological change → update TECHNICAL_SPECS.md → archive old version.
- No silent changes.
- No orphaned results.

This directory structure is part of the scientific method for this project.
