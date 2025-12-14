# Reconstruction of Proposal Development: Logic, Decisions, and Constraints

## 0) Ground rules that silently shaped everything
### Non-negotiable constraints (course + reviewer realism)
- **Grant-proposal posture**: must read like fundable research, not a class essay; must be evaluable on novelty, feasibility, controls, and risk management.
- **Length and structure**: 10–15 pages text + figures + references; explicit sections: Executive Summary, Background, Proposed Work.
- **Avoid overclaiming**: no quantitative $T_2$ prediction promise without a validated mapping; treat $T_2$ as **ordering / consistency check**.
- **Reviewer skepticism anticipated**:
  - “You just rediscovered the gap” (donor case risk).
  - “Convergence is numerical artifact” (needs basis choice + controls).
  - “Toy model evidence” (Anderson 4-qubit risk).
  - “Scope creep” (too many systems dilutes).

### Central narrative pressure
- Proposal must **explain why this is new** without pretending it is a substitute for established decoherence modeling.
- Proposal must **show a path to results** with clear measurements (algorithmic observables) and clear calibration logic.

---

## 1) Intellectual trajectory (how the idea evolved, step-by-step)

### 1.1 Initial dissatisfaction: energy-centric electronic structure is misaligned with coherence questions
- Starting observation: defect qubits show **order-of-magnitude $T_2$ variability** even in nominally similar materials.
- Existing toolchain is optimized for:
  - ground-state energies,
  - orbital occupations,
  - static descriptors,
  - energetic correctness.
- But coherence is governed by **coupling structure** to environments (spin bath, charge noise, phonons) and by whether the qubit state forms a **well-isolated subspace**.
- Key “why this matters”: current workflows often treat decoherence as an *add-on* after electronic structure, while in practice coherence feasibility is a **selection constraint**.

### 1.2 Conceptual inversion: treat algorithmic convergence as a physically meaningful observable
- Pivot: instead of seeing convergence issues as numerical annoyances, reinterpret them as signals of:
  - near-degeneracy,
  - manifold dimensionality,
  - coupling connectivity,
  - non-separability between “qubit” and “environment” subspaces.
- Explicit statement that emerged:
  - **Algorithmic convergence stability** is a proxy for **electronic separability/isolation**.
  - Isolation/separability is what determines susceptibility to decoherence channels.
- This inversion redefines “what we measure”:
  - not only energies, not only orbitals,
  - but **how hard it is for a Krylov solver to isolate the state**.

### 1.3 Why Krylov methods (and why SKQD specifically)
- Krylov methods are inherently sensitive to spectral structure because they:
  - build low-dimensional subspaces,
  - converge depending on gaps, degeneracies, intruder states, and mixing.
- They provide a **natural notion of “convergence texture”**: not just output value, but convergence dynamics.
- SKQD enters as the forward-looking method because:
  - it is a quantum-compatible Krylov approach,
  - it supports **sample-based** estimation (shot noise becomes part of the convergence story),
  - it motivates a “hybrid-ready” pipeline for strongly correlated active spaces.
- Critically: Krylov methods allow a uniform framework across:
  - simple donor manifolds,
  - symmetry-protected centers,
  - correlated transition-metal shells.

### 1.4 Why orbital ablation became the organizing principle
- Orbital ablation is the mechanism to turn convergence into a diagnostic:
  - remove/restrict orbitals,
  - watch convergence respond,
  - interpret sensitivity as “orbital importance for maintaining isolation.”
- Analogy adopted (with care): feature ablation in ML.
  - Not claiming ML; using ablation logic to justify “importance via sensitivity.”
- Orbital ablation operationalizes the method into a repeatable protocol:
  - define active space candidates,
  - ablate subsets,
  - record convergence observables,
  - map patterns to decoherence channels via taxonomy.

---

## 2) Decision points (forks, alternatives, and why choices were made)

### 2.1 Early fork: “Start with a simple defect class” vs “jump to realistic complex defects”
**Options considered**
- Start directly with complex defects (e.g., transition-metal defects, divacancies).
- Start with simplest “textbook” qubits for calibration.

**Decision**
- Start with simplest calibration case first.

**Reason**
- The method’s credibility depends on showing:
  - convergence metrics behave sensibly in a regime where physics is well-known,
  - before claiming anything in non-obvious regimes.

**Risk mitigated**
- Avoid reviewer critique: “You have no calibration; convergence could mean anything.”

### 2.2 Donors in silicon: used only for calibration, not as “proof of novelty”
**Why donors were selected initially**
- Clean, well-studied, benchmarkable.
- Provides a controlled environment for linking:
  - spectral separation (valley-orbit splitting),
  - Krylov convergence behavior,
  - known coherence times.

**Why donors were demoted to appendix-only**
- Donor physics is a classic case where:
  - “gap arguments” are dominant and expected.
- If donors are featured prominently in the main narrative, reviewers can dismiss the method as:
  - “You’re just using a solver to rediscover $\Delta E$.”

**Risk mitigated**
- Main-text novelty dilution.
- “Obvious case” perception.

**Resulting placement**
- Donor case becomes an *anchor calibration appendix*:
  - necessary for interpretability,
  - carefully framed as non-novel validation.

### 2.3 NV vs SiV: necessary intermediate tier, but insufficient as a first “capstone”
**Why it was introduced**
- Represents a qualitatively different regime:
  - not dominated by gap alone,
  - dominated by symmetry and electric-field noise coupling.
- Introduces the shift from:
  - static Hamiltonian solution → perturbation/noise operator sensitivity.

**Why it is necessary**
- It demonstrates the framework can:
  - incorporate **noise operators**,
  - interpret convergence/energy response under perturbations,
  - link symmetry (parity/inversion) to “susceptibility slope.”

**Why it is insufficient alone**
- Risk: it can still be framed as “symmetry selection rules are known; you’re confirming selection rules.”
- It does not stress the strong-correlation and manifold-collapse aspects that define the proposal’s “non-obvious” ambition.

**Risk mitigated by how it is framed**
- It is the “symmetry-controlled immunity tier,” not the final proof.
- It demonstrates method extensibility beyond gap calibration.

### 2.4 Advanced-case fork: which “non-obvious” system proves value beyond gap/symmetry?
**Candidates discussed**
- 4-qubit Anderson impurity toy model (hybridization knob).
- SiC divacancy (24 qubits) as scaling + symmetry differences.
- Transition-metal defect in SiC: Chromium (18 qubits) with ligand-field quenching.
- (Implicitly) other transition metals and correlated shells.

**Decision**
- Choose **Chromium in SiC (18 qubits)** as the capstone.

**Reasons**
- It naturally embodies a regime where:
  - coherence depends on **manifold structure** and **orbital quenching**,
  - not simply on a single-particle gap or a parity selection rule.
- It stress-tests SKQD in a meaningful way:
  - multi-electron, multi-orbital correlations,
  - degenerate manifold collapse under ligand field.
- It supports the intended “capstone claim”:
  - convergence stiffness reflects low-energy manifold entropy/connectivity.

**Risk mitigated**
- Avoid toy-model evidence critique.
- Avoid overextension into too many systems.
- Avoid re-centering around known parity rules alone.

### 2.5 Why Anderson impurity (4-qubit) and SiC divacancy (24-qubit) were demoted
**Anderson impurity (4-qubit) demotion**
- Kept only as conceptual bridge:
  - illustrates hybridization inheritance of noise.
- Not used as evidence:
  - toy model is too easily dismissed as trivial/parameter-chosen.
- Risk mitigated:
  - “You can tune V to force the result; it doesn’t validate anything.”

**Divacancy (24-qubit) demotion**
- High value as future scaling target and symmetry-site comparison.
- But it risks scope creep and implementation burden in first submission.
- Risk mitigated:
  - reviewer skepticism about feasibility,
  - dilution of central capstone story.

**Result**
- Both remain as “supporting intuition / future scaling” and not core claims.

---

## 3) The three-tier structure (what each tier proves, and what it deliberately does not claim)

### Tier 1 — Basic: gap-controlled isolation (Si:P vs Si:Bi)
**Purpose**
- Calibration of convergence observables in a regime where the physical driver is understood.

**What it proves**
- Convergence observables respond monotonically to spectral separation in a non-eigen basis.
- “Convergence penalty” is interpretable when nearby states encroach.

**What it does NOT claim**
- It does not claim novelty.
- It does not claim convergence predicts $T_2$ universally.
- It does not claim absolute $T_2$ prediction.

**Strategic role**
- Establishes trust in the metrics, not in the screening engine.

### Tier 2 — Intermediate: symmetry-controlled noise immunity (NV vs SiV)
**Purpose**
- Demonstrate that the framework can incorporate **noise operators** and symmetry constraints.

**What it proves**
- Convergence/energy sensitivity under perturbation differentiates:
  - symmetry-protected vs non-protected active spaces.
- Introduces susceptibility slope / perturbation sweep logic.

**What it does NOT claim**
- Not the final proof of added value beyond known selection rules.
- Not a quantitative $T_2$ model.
- Not a complete decoherence simulation.

**Strategic role**
- Shows the method is not just a gap-detector; it can test immunity under structured perturbations.

### Tier 3 — Advanced: manifold-controlled shielding (Cr in SiC, 18 qubits)
**Purpose**
- First genuinely non-obvious regime: strong correlation and manifold collapse.

**What it proves**
- SKQD convergence stiffness can diagnose:
  - low-energy manifold dimensionality,
  - near-degenerate configuration mixing,
  - “quenching” as manifold collapse under ligand-field coupling.

**What it does NOT claim**
- No absolute $T_2$ prediction.
- No claim that convergence = $T_2$ numerically.
- Uses $T_2$ only as qualitative ordering / consistency check.

**Strategic role**
- Capstone: demonstrates proposed method can identify a coherence-relevant mechanism beyond gap/symmetry.

---

## 4) Appendix strategy (why it exists, what it calibrates, what it avoids, how it protects the main proposal)

### 4.1 Why the donor appendix exists
- It is the “sanity check” that makes convergence observables credible.
- It demonstrates the metric suite (iterations, residual decay, Ritz stabilization) behaves consistently in a known regime.
- It allows the main text to avoid spending precious space on a trivial mechanism.

### 4.2 What it calibrates
- Defines and operationalizes convergence metrics:
  - $N_{\text{iter}}(\|r\|<10^{-6})$
  - $\Delta N_{\text{Krylov}} = N_{\text{full}} - N_{\text{isolated}}$
  - Ritz stabilization: $|E_{\text{Ritz}}^{(k)}-E_{\text{Ritz}}^{(k-1)}|$
  - residual slope: $\log\|r_k\|$ vs $k$
- Shows dependence on:
  - basis choice (valley basis vs eigenbasis),
  - spectral crowding (small valley-orbit splitting).
- Provides a “do not overinterpret” model example.

### 4.3 What it intentionally avoids claiming
- No novelty claim.
- No assertion that donors validate the full framework.
- No implication that “gap explains everything.”
- No attempt to fit $T_2$.

### 4.4 How it protects the main proposal
- It preempts the reviewer line:
  - “Convergence is numerics.”
- It prevents the main body from being criticized as:
  - “a worked example of obvious physics.”
- It frees the main text to focus on:
  - symmetry/noise operator tests,
  - manifold-controlled shielding in correlated systems.

---

## 5) Meta-strategy (the unifying thesis and its disciplined framing)

### 5.1 Unifying thesis (what the whole proposal is actually doing)
- **Treat diagonalization behavior as data.**
- Use **orbital ablation** to perturb the representation and read out sensitivity.
- Interpret **convergence stiffness** as proxy for:
  - separability,
  - robustness,
  - “effective entropy” / dimensionality of low-energy space.
- Map convergence signatures to decoherence channels via a taxonomy.

### 5.2 The disciplined claim boundary (what is promised vs what is intentionally not promised)
- Promised:
  - a screening diagnostic for “coherence viability” and mechanism identification.
  - qualitative ranking and mechanism-consistent ordering with known $T_2$.
  - a reproducible metric suite.
- Not promised:
  - quantitative $T_2$ prediction.
  - a full decoherence simulator replacing CCE or phonon modeling.

### 5.3 SKQD positioned as a “materials robustness scanner”
- The framework generalizes beyond qubits:
  - stiff convergence is good when you want isolation/robustness,
  - sluggish convergence is good when you want degeneracy/mixing (e.g., TADF).
- This reframes SKQD as a tool that can:
  - screen, classify, and guide experimental focus,
  - not just compute energies.

---

## 6) Resulting “final shape” of the proposal (why it ended up structured the way it is)
- **Main narrative**: inversion + taxonomy + three-tier validation culminating in correlated manifold case.
- **Calibration**: donors moved to appendix to avoid novelty dilution.
- **Scope control**:
  - one advanced capstone (Cr:SiC) to concentrate credibility.
  - other systems (Anderson, divacancy) retained as future work for feasibility optics.
- **Defensibility**:
  - clear metrics,
  - clear controls (basis choice, perturbation sweeps),
  - explicit non-overclaiming about $T_2$.
