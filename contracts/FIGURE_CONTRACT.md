# FIGURE_CONTRACT.md
# Figure Specification Contract for Calibration Case

## Purpose
This document defines **proposal-ready figures** required from the donor calibration code.
Figures must be publication-quality and directly support claims in the proposal.

Exploratory plots are not acceptable substitutes.

---

## Figure 1 — Residual Norm Decay Comparison

**Filename:**  
`fig_residual_decay_donors.pdf`

**Content:**
- Semi-log plot of \( \|r_k\| \) vs iteration \( k \)
- Four curves:
  - Si:P (isolated)
  - Si:P (full)
  - Si:Bi (isolated)
  - Si:Bi (full)

**Axes:**
- X: Krylov iteration \( k \)
- Y: Residual norm \( \|r_k\| \) (log scale)

**What It Proves:**
- Full-space Si:P converges more slowly than Si:Bi
- Isolated baselines converge rapidly for both

**Proposal Section Supported:**
- Appendix A (Calibration)
- Convergence diagnostics interpretability

---

## Figure 2 — Ritz Stabilization Trajectories

**Filename:**  
`fig_ritz_stabilization_donors.pdf`

**Content:**
- Plot of \( \delta E_k \) vs iteration \( k \)
- Same four curves as Figure 1

**Axes:**
- X: Krylov iteration \( k \)
- Y: \( |E_{\text{Ritz}}^{(k)} - E_{\text{Ritz}}^{(k-1)}| \)

**What It Proves:**
- Near-degenerate manifolds produce delayed stabilization
- Si:Bi stabilizes earlier than Si:P

**Proposal Section Supported:**
- Appendix A
- Metric definition validation

---

## Optional Figure 3 — Convergence Penalty Summary

**Filename:**  
`fig_convergence_penalty_bar.pdf`

**Content:**
- Bar chart of \( \Delta N_{\text{Krylov}} \)
- Bars:
  - Si:P
  - Si:Bi

**Axes:**
- X: Donor species
- Y: Convergence penalty

**What It Proves:**
- Quantitative ordering consistent with known valley-orbit physics

---

## Formatting Requirements
- Vector PDF preferred
- Font size ≥ 9 pt when embedded
- Colorblind-safe palette
- No legends inside plot region

---

## Prohibited Figures
- Energy-only plots without convergence context
- Eigenbasis-only convergence plots
- Figures lacking explicit metric labels
