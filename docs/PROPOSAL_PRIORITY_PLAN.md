# Priority Expansion Plan: Strengthening the Main Proposal Without Dilution

## 0) Expansion principle (the rule that prevents “more pages” from weakening the story)
Every added paragraph must do at least one of the following:
1) Increase **reviewer confidence** (clarify novelty, controls, feasibility, risk management), OR  
2) Increase **mechanistic clarity** (how convergence maps to decoherence channel), OR  
3) Increase **experimental evaluability** (what you will compute, how you will compare, what constitutes success/failure).

If a candidate addition does not satisfy (1–3), it goes to future work or is cut.

---

## 1) Absolute priorities (must-do expansions)

### P1 — Tighten and deepen the “Convergence Signature → Decoherence Channel” taxonomy (core intellectual asset)
**What to add**
- Convert the current taxonomy from a 3-row table into a structured mapping with:
  - (a) **signature definition** (operational; metric-based),
  - (b) **algorithmic cause** (Krylov theory intuition),
  - (c) **physical interpretation** (which coupling structure produces it),
  - (d) **expected ablation response** (what happens when removing orbitals),
  - (e) **control to falsify** (how to prove it’s not numerics).

**Why it strengthens**
- Turns a suggestive framing into a **testable classification framework**.
- Reviewers can see what data you will produce and how it supports conclusions.

**New intellectual value**
- Establishes “convergence signatures” as a reproducible diagnostic language, not a metaphor.

**Estimated page contribution**
- +1.0 to +1.5 pages (with one expanded table + a short subsection per channel).

---

### P2 — Expand the Intermediate tier into a fully evaluable “Perturbation Test” protocol (NV vs SiV)
**What to add**
- A crisp protocol subsection:
  - define $H_{\text{total}} = H_0 + \lambda \hat{V}_{\text{noise}}$,
  - define the **observable**: susceptibility slope $\chi = [E(\lambda)-E(0)]/\lambda$,
  - define the **expected symmetry outcome**: $\chi \approx 0$ in protected manifold,
  - define **what you will log**: $\chi$, plus convergence metrics under $\lambda$ sweep.
- Add a one-paragraph “why this is not just selection rules”:
  - the novelty is not parity itself; it’s using SKQD observables as a diagnostic readout under structured perturbations within active-space ablation.

**Why it strengthens**
- Shows the framework handles **noise operators** and not just static diagonalization.

**New intellectual value**
- Makes the intermediate tier a proper bridge from “gap calibration” to “manifold shielding.”

**Estimated page contribution**
- +0.8 to +1.2 pages (plus one figure; see F2).

---

### P3 — Promote the Cr:SiC advanced case into a mini-proposal inside the proposal (capstone must “feel fundable”)
**What to add**
- Recast as a three-part plan with explicit deliverables:
  1) **Model construction & active space definition** (18 qubits: 3d shell + ligands).
  2) **Two-regime test**: free-ion vs ligand-field (unquenched vs quenched).
  3) **Diagnostic readout**: manifold collapse measured via convergence stiffness metrics.
- Add explicit “what success looks like”:
  - unquenched: slow convergence + high volatility + delayed Ritz stabilization,
  - quenched: rapid convergence + stable Ritz + steep residual slope.
- Add “what could break” and mitigation:
  - basis dependence → include basis-rotation sensitivity check,
  - parameter sensitivity → show invariance of qualitative ordering across a small parameter window.

**Why it strengthens**
- This is the non-obvious claim; it must look like a real research plan, not a concept note.

**New intellectual value**
- Establishes “convergence stiffness as entropy proxy / manifold diagnostic” as the capstone contribution.

**Estimated page contribution**
- +1.5 to +2.5 pages (especially if paired with a figure + a compact workflow diagram).

---

### P4 — Add explicit control experiments and failure modes (this is where proposals become “review-proof”)
**What to add**
Create a “Controls & Failure Modes” subsection with:
- **Control A (basis trivialization)**: eigenbasis makes convergence trivial → therefore use valley / localized / non-eigen bases intentionally.
- **Control B (initial-state sensitivity)**: multiple initial vectors; report variance in $N_{\text{iter}}$.
- **Control C (active-space sanity)**: if you ablate orbitals that are not coupled, metrics should not change.
- **Failure mode 1**: convergence dominated by numerical conditioning / shot noise → mitigation: condition number logging + repeated sampling.
- **Failure mode 2**: apparent trends collapse when tolerances vary → mitigation: fixed residual threshold + tolerance sweep in supplement.

**Why it strengthens**
- Reviewers look for ways to say “this is numerics”; this section preempts them.

**New intellectual value**
- Clarifies what your observables measure and what they do not measure.

**Estimated page contribution**
- +0.8 to +1.3 pages.

---

### P5 — Add a “Project Plan & Feasibility” block (timeline + compute scale) to satisfy grant expectations
**What to add**
- A short plan:
  - Stage 1: donor appendix calibration (completed logic; minimal compute),
  - Stage 2: NV/SiV perturbation test (small active spaces),
  - Stage 3: Cr:SiC capstone (18 qubits; target run count; metric extraction),
  - Stage 4: ablation sweep automation + taxonomy mapping.
- Include explicit output artifacts:
  - metric logs, plots, taxonomy classification, ranked candidate list.

**Why it strengthens**
- Converts idea into an executable plan.

**New intellectual value**
- Signals competence and reduces feasibility risk.

**Estimated page contribution**
- +0.5 to +0.8 pages (plus a compact figure or table).

---

## 2) Secondary priorities (nice-to-have, conditional)

### S1 — Comparative “Why not DFT/DMRG/DMET alone?” section (short, surgical)
**Add**
- A half-page positioning:
  - These methods optimize energies and wavefunctions, but do not produce a **diagnostic of isolation robustness** under ablation/perturbation.
  - Your novelty is an additional observable layer: convergence behavior.

**Benefit**
- Helps reviewers place novelty without feeling threatened by established methods.

**Page contribution**
- +0.4 to +0.7 pages.

---

### S2 — Explicit negative cases (where convergence should NOT correlate with $T_2$)
**Add**
- A short paragraph listing scenarios:
  - when decoherence dominated by extrinsic fabrication noise unrelated to electronic manifold,
  - when convergence artifacts are introduced by poor measurement statistics,
  - when active space misses the true coupling orbitals.

**Benefit**
- Increases credibility by showing you know limits.

**Page contribution**
- +0.3 to +0.5 pages.

---

### S3 — Visualization strategy (if figures are used to drive pages cleanly)
**Add**
- A figure plan with captions drafted (see below).
- A short “data products” paragraph.

**Page contribution**
- depends on figure count (likely +0.5 to +1.0 pages text around figures).

---

## 3) Explicit “do not add” list (content that will weaken the proposal now)

### D1 — Do not attempt quantitative $T_2$ prediction
- Avoid fitting formulas or claiming a numeric mapping from convergence to seconds.
- Keep $T_2$ as qualitative ordering and mechanism-consistency check.

### D2 — Do not add extra materials systems unless they introduce a new mechanism tier
- More systems without new conceptual leverage will look like scope creep.
- Keep exactly three tiers in main text.

### D3 — Do not include deep Hamiltonian parameter dumps in the main text
- Parameter tables belong in appendix or supplementary.
- Main text should focus on what is computed and what is compared.

### D4 — Do not over-focus on qubit counts as a selling point
- Qubit count is implementation detail; reviewers care about scientific diagnostic value.

---

## 4) Figure-driven expansion strategy (how to add pages without diluting)

### F1 — “Three-tier diagnostic ladder” schematic (1 figure)
**What it proves**
- The proposal is structured, not scattered.
- Each tier tests a distinct mechanism.

**Where**
- End of Background / start of Proposed Work.

**Text it enables**
- 0.5–0.8 pages describing the ladder and what each tier contributes.

---

### F2 — “Perturbation response” conceptual plot: $E(\lambda)$ for NV vs SiV (1 figure)
**What it proves**
- Symmetry protection shows up as near-zero susceptibility slope.
- Makes intermediate tier instantly legible.

**Where**
- Intermediate tier subsection.

**Text it enables**
- 0.4–0.7 pages of protocol + interpretation.

---

### F3 — “Convergence texture panel” (residual curves + Ritz stabilization) comparing two regimes (1 figure)
**What it proves**
- Convergence is not a scalar; it has signatures.
- Visual evidence of near-degeneracy vs isolation.

**Where**
- Taxonomy section or Advanced case.

**Text it enables**
- 0.6–1.0 pages mapping signatures to physics.

---

### F4 — “Manifold collapse” diagram for Cr:SiC: unquenched vs quenched (1 figure)
**What it proves**
- The advanced case is about low-energy manifold dimensionality/connectivity.
- Shows why gap alone is insufficient in correlated shells.

**Where**
- Advanced case subsection.

**Text it enables**
- 0.7–1.2 pages.

---

### F5 — “Workflow + outputs” diagram (ablation loop + logged metrics + ranking) (1 figure)
**What it proves**
- Feasibility and deliverables are concrete.

**Where**
- End of Proposed Work or Validation Strategy.

**Text it enables**
- 0.4–0.8 pages.

---

## 5) Final structural target (how to reach 10–15 pages organically)

### Target page allocation (text only; figures excluded from count)
1) **Executive Summary**: 0.8–1.0 pages  
   - keep as-is, lightly sharpen claims and deliverables.

2) **Background & Motivation**: 2.5–3.5 pages  
   - expand with:
     - limitations of energy-centric methods,
     - why decoherence is a coupling-structure problem,
     - why “diagnostic observables” are missing today,
     - insert Figure F1.

3) **Conceptual Innovation + Taxonomy**: 2.0–3.0 pages  
   - expand P1 heavily:
     - operational definitions,
     - mapping signatures → channels,
     - include Figure F3.

4) **Proposed Work**: 4.5–6.0 pages  
   - Tier 1: short pointer + explicitly “Appendix calibration” (0.3–0.6 pages)
   - Tier 2: perturbation test protocol (1.0–1.5 pages) + Figure F2
   - Tier 3: Cr:SiC mini-proposal (2.0–3.0 pages) + Figure F4
   - Implementation + ablation automation + outputs (1.0–1.5 pages) + Figure F5

5) **Validation Strategy**: 0.8–1.2 pages  
   - tighten: what gets compared, what constitutes success, what would falsify.

6) **Supporting Models + Future Work**: 0.8–1.2 pages  
   - keep Anderson impurity and divacancy here, intentionally brief.

7) **Conclusion**: 0.4–0.6 pages  
   - reiterate contribution + deliverables.

**How this reaches 10–15 pages**
- Conservative total (text): ~12 pages if P1–P5 are executed with discipline.
- Figures: 5 figures minimum already satisfied by F1–F5.

### Why the narrative stays tight
- Single central claim: convergence observables diagnose coherence-relevant isolation mechanisms.
- Three tiers only; everything else is supporting or future.
- Controls prevent the “numerics” critique.
- No overreach on $T_2$ quantification.

---

## 6) Immediate execution checklist (what to write next, in order)
1) Write P1 taxonomy expansion + build expanded table structure.
2) Write P4 controls/failure-modes subsection.
3) Rewrite intermediate NV/SiV subsection into a protocol with $\chi$ and $\lambda$ sweep.
4) Rewrite Cr:SiC subsection into a mini-proposal with deliverables and success criteria.
5) Add feasibility/timeline block.
6) Insert figure placeholders + captions and let figures drive clean expansion.
