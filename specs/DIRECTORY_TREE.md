
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
├── DIRECTORY_TREE.md
├── METRICS_SPEC.md
├── FIGURE_CONTRACT.md
├── RUN_LOG_SCHEMA.md
├── CODING_CHECKLIST.md
├── TECH_SPECS.md
├── archive/
├── src/
├── configs/
├── results/
├── figures/
├── notebooks/
├── scripts/
└── environment/

```

---

## 2. Top-Level Documentation Files

### Required Markdown Files (Authoritative Specs)

```

├── DIRECTORY_TREE.md        # This file: canonical repo structure
├── METRICS_SPEC.md          # Definitions of all convergence metrics
├── FIGURE_CONTRACT.md       # Mapping: figures → metrics → proposal sections
├── RUN_LOG_SCHEMA.md        # JSON schema for all run logs
├── CODING_CHECKLIST.md      # Atomic implementation checklist (living document)
├── TECH_SPECS.md            # Technical specification of the calibration project

```

**Rules**
- These files define the **contract** between science, code, and figures.
- Any change to methodology must update TECH_SPECS.md and be archived.

---

## 3. Archive Directory (Version Control Beyond Git)

```

archive/
├── checklist_versions/
│   ├── CODING_CHECKLIST_v1.md
│   ├── CODING_CHECKLIST_v2.md
│   └── ...
├── tech_specs_versions/
│   ├── TECH_SPECS_v1.md
│   ├── TECH_SPECS_v2.md
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

## 4. Source Code (`src/`)

```

src/
├── **init**.py
├── hamiltonians/
│   ├── **init**.py
│   ├── donor_valley.py          # Valley-basis Hamiltonians (Si:P, Si:Bi)
│   └── basis_controls.py        # Eigenbasis / rotated basis controls
├── krylov/
│   ├── **init**.py
│   ├── krylov_loop.py           # Core SKQD-like iteration loop
│   ├── subspace_builder.py      # Krylov vector construction
│   ├── projected_matrices.py    # H_ij, S_ij estimation
│   └── residuals.py             # Residual computation and stopping criteria
├── estimation/
│   ├── **init**.py
│   ├── expectation.py           # Pauli expectation estimation
│   └── sampling.py              # Shot-based sampling utilities
├── analysis/
│   ├── **init**.py
│   ├── convergence_metrics.py   # N_iter, residual slope, Ritz stabilization
│   └── validation.py            # Consistency and sanity checks
├── io/
│   ├── **init**.py
│   ├── run_logger.py            # JSON log writer (RUN_LOG_SCHEMA compliant)
│   └── config_loader.py         # Load YAML/JSON configs
└── utils/
├── **init**.py
├── seeds.py
├── timing.py
└── linear_algebra.py

```

**Rules**
- No plotting code in `src/`.
- No hard-coded parameters; all parameters come from `configs/`.

---

## 5. Configuration Files (`configs/`)

```

configs/
├── materials/
│   ├── SiP.yaml                 # Physical parameters for Si:P
│   └── SiBi.yaml                # Physical parameters for Si:Bi
├── active_spaces/
│   ├── isolated.yaml            # 2-qubit A1-only configuration
│   └── full.yaml                # 12-qubit valley manifold configuration
├── krylov/
│   ├── default.yaml             # max_iter, tolerance, orthogonalization
│   └── stress_test.yaml         # altered tolerances for controls
├── backend/
│   ├── aer_statevector.yaml
│   └── aer_sampling.yaml
└── experiments/
├── donor_calibration.yaml   # Master experiment config

```

**Rules**
- All numerical experiments must be reproducible from configs alone.
- Config files are immutable once used for a published figure.

---

## 6. Results (`results/`)

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
└── summaries/
├── aggregation_report.json
└── aggregation_notes.md

```

**Rules**
- Only RUN_LOG_SCHEMA-compliant JSON files allowed in `raw/`.
- Aggregated tables must cite exact run IDs.

---

## 7. Figures (`figures/`)

```

figures/
├── calibration/
│   ├── residual_decay_SiP_vs_SiBi.pdf
│   ├── ritz_stabilization_SiP_vs_SiBi.pdf
│   └── convergence_penalty_bar_chart.pdf
└── drafts/
├── exploratory_plots/
└── deprecated/

```

**Rules**
- Only figures listed in FIGURE_CONTRACT.md may appear in the proposal.
- Draft figures must not be cited.

---

## 8. Scripts (`scripts/`)

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

## 9. Notebooks (`notebooks/`) — Optional / Non-Authoritative

```

notebooks/
├── exploration.ipynb
└── debugging.ipynb

```

**Rules**
- Notebooks are for development only.
- No figure or result may depend on a notebook.

---

## 10. Environment Specification (`environment/`)

```

environment/
├── requirements.txt
├── environment.yml              # Conda environment
└── system_snapshot.txt          # Python, OS, Qiskit versions

```

**Rules**
- Every run log must be reproducible under one environment snapshot.

---

## 11. Invariants (Non-Negotiable)

- Every figure → references run logs → reference configs → reference code.
- Every methodological change → update TECH_SPECS.md → archive old version.
- No silent changes.
- No orphaned results.

This directory structure is part of the scientific method for this project.
```
