# METRICS_SPEC.md
# Convergence Metrics Specification (Calibration Tier: Donor Systems)

## Purpose
This document defines, unambiguously and exhaustively, the convergence metrics used in the donor calibration case (Si:P vs Si:Bi).  
These metrics are treated as **algorithmic observables**, not numerical artifacts, and are interpreted **qualitatively**, not as predictors of absolute coherence times.

This file is **authoritative**. All code, logs, plots, and tables must conform to these definitions exactly.

---

## Global Conventions

- Residual tolerance:  
  \[
  \|r\| < 10^{-6}
  \]
- Krylov iteration index: \( k = 1, 2, \dots \)
- All convergence metrics are reported **per run**.
- Ordering matters; absolute values do not.

---

## Primary Metrics (Must Be Reported)

### 1. Krylov Iteration Count to Fixed Residual
**Symbol:** \( N_{\text{iter}} \)

**Definition:**  
The minimum number of Krylov iterations required such that:
\[
\|r_k\| < 10^{-6}
\]

**Operational Meaning:**  
Measures how difficult it is for the algorithm to isolate the target eigenstate from nearby states.

**Interpretation (Calibration Regime):**
- Smaller \( N_{\text{iter}} \) → stronger isolation
- Larger \( N_{\text{iter}} \) → denser spectrum / stronger coupling

**Expected Ordering:**
\[
N_{\text{iter}}(\text{Si:P, full}) > N_{\text{iter}}(\text{Si:Bi, full})
\]

---

### 2. Convergence Penalty
**Symbol:** \( \Delta N_{\text{Krylov}} \)

**Definition:**
\[
\Delta N_{\text{Krylov}} =
N_{\text{iter}}^{(\text{full})} -
N_{\text{iter}}^{(\text{isolated})}
\]

**Purpose:**  
Normalizes away trivial system-size effects by comparing against an isolated-qubit baseline.

**Interpretation:**
- Large penalty → strong interference from environmental states
- Small penalty → effective isolation

**Calibration Expectation:**
\[
\Delta N_{\text{Krylov}}(\text{Si:P}) >
\Delta N_{\text{Krylov}}(\text{Si:Bi})
\]

---

### 3. Residual Norm History
**Symbol:** \( \|r_k\| \)

**Definition:**  
The Euclidean norm of the residual vector at iteration \( k \):
\[
r_k = H |\phi_k\rangle - E_{\text{Ritz}}^{(k)} |\phi_k\rangle
\]

**Usage:**  
Logged as a full sequence \(\{\|r_1\|, \|r_2\|, \dots\}\).

---

### 4. Residual Decay Slope
**Symbol:** \( \alpha_r \)

**Definition:**  
Slope of the best linear fit to:
\[
\log \|r_k\| \quad \text{vs} \quad k
\]

**Interpretation:**
- Steeper (more negative) slope → clean spectral separation
- Shallow slope / plateaus → near-degeneracy

---

### 5. Ritz Value Stabilization
**Symbol:** \( \delta E_k \)

**Definition:**
\[
\delta E_k =
\left| E_{\text{Ritz}}^{(k)} - E_{\text{Ritz}}^{(k-1)} \right|
\]

**Usage:**  
Tracked as a sequence to diagnose root mixing and near-degeneracy.

---

## Secondary Metrics (Logged, Not Used for Claims)

### Runtime per Iteration
- Wall-clock time per Krylov iteration
- Used only to document feasibility

### Sampling Budget
- Number of shots per expectation value
- Used to assess noise-induced volatility, not physics

---

## Explicit Non-Claims
- No metric predicts absolute \( T_2 \)
- Metrics are not monotonic across all materials classes
- Metrics are only meaningful under **non-eigen basis construction**

---

## Basis Requirement (Hard Constraint)
- Hamiltonians **must** be constructed in a non-eigen basis (valley basis).
- Eigenbasis diagonalization is permitted **only** as a negative control.

Failure to meet this requirement invalidates all metrics.
