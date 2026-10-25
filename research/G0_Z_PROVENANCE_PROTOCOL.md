# G0-Z · Provenance before interpretation: avoid false gluing

Date: 2026-10-25. Pre-registered BEFORE G0-Z code/run. Motivation: G0-Y demonstrated information synergy but its likelihood assumes that every sensor event refers to the SAME latent fiber contrast D. That assumption is an epistemic dependency, not a mathematical guarantee.

## Hypotheses and typed organs

Three competing models for scalar orientation-sensitive sensing:
- H_I: shared contrast D~N(0,1), orientation identity. Odd and even probes observe D with Gaussian noise.
- H_P: shared contrast D~N(0,1), orientation transpose. Odd probes observe −D, even probes observe D.
- H_⊥: *independently replaced contrast* D_k~N(0,1) on every sensing event, no stable cross-time fiber identification and hence no meaningful T classification. Predictive observation law N(0,1+σ_k²) each event.

Prior P(H_I,H_P,H_⊥)=(.4,.4,.2). Bayesian model averaging: under H_I/H_P track separate online conjugate Gaussian posteriors over the SAME unknown D; under H_⊥ integrate out a fresh D_k each time. Use explicit likelihood normalization. Expose two distinct inferential objects: orientation T conditional on persistent identity, and continuity class C∈{shared,independent}. Do not collapse into one scalar confidence.

Simulated surrogate organs: scalar sensing/probe parity and quiet/focus gating, finite metabolic ledger and an *auditor* of relation provenance. Current model has no biological neurons, synapses, gut circuitry or independent continuous neural clocks.

## Important new corollary [P conditional]

A sequence of even probes cannot reveal the orientation T under balanced H_I/H_P, but **repeated even probes can reveal persistence C**: shared D induces covariance +Var(D), whereas independent replacements do not. Thus a sensor can be uninformative about the *topological orientation* and simultaneously informative about *identity continuity*. Information value is indexed by the question under investigation.

The symmetric latent model of G0-Y admits zero orientation information from same-parity observations, but a three-hypothesis organism can use such observations to validate or challenge whether gluing is legitimate.

## Frozen observation schedule / resource budget

Same four sensor modes and cost/noise law as G0-Y, fixed six-probe schedule: even_focus, odd_focus, even_quiet, odd_quiet, even_quiet, odd_quiet. With G0-Y energy cap 5.5, floor .22, maintenance .025 per probe, its full cost is within reserve. Same observed stream is fed to:
- **Naive orientation-only filter**, which falsely assumes all D observations share one state.
- **Provenance-aware mixture**, distinguishing shared identity vs complete per-event replacement.

Simulate 128 independent paired seeds in each world: shared identity, shared transpose, and independent replacement (orientation not well-defined). Use independent Gaussian noise for each probe and fixed seeded D per shared episode. No privileged D or T as inputs to either filter.

## Decision gates and metrics

Orientation declaration from provenance filter requires P(shared)≥.70 AND max(P(I|shared),P(P|shared))≥.95. Naive orientation filter declares if its own posterior confidence≥.95. Quarantine/rejection if P(independent)≥.50. Otherwise abstain. Compare orientation accuracy/coverage on shared worlds, false orientation declarations on independent world, calibration/Brier for continuity, and mean quarantine correctness. Report seed-level paired results, not just aggregated claims.

No forced decision is counted correct on independent world: the label T is *unidentifiable by construction* when D re-draws independently every time.

## Required negative and out-of-family cases

1. All-even observations can update continuity while leaving P(I|shared)=P(P|shared), hence orientation unresolved.
2. Exactly one observation cannot distinguish shared/independent models because their marginals coincide.
3. Symmetric hidden D plus only odd or only even probes cannot identify orientation even after many events under a fixed shared model.
4. No negative variance, energy overdraft, posterior normalization failure, or unsupported confidence promotion.
5. Mid-episode change-point D replacement is outside both perfect-persistence and every-event-independent classes. Evaluate separately as exploratory robustness and explicitly report model misspecification.
6. The auditor may reduce false confident gluing but could also refuse too many valid shared states. Preserve the coverage tradeoff and any failures.
7. A conventional three-component real Bayesian state-space model given the identical observations has EXACTLY the same predictions. No special geometric algorithmic advantage, intelligence or hardware power saving is demonstrated.

## Evidence [P/A/M/H/S]

P conditional: covariance/identifiability identities. A: chosen hidden-state and Gaussian sensor laws, prior and thresholds. M: only later bounded remote CPU observations. H: provable safe semantic gluing in adaptive organisms, software/hardware efficiency; S: Worldlet continuity and 'epistemic immune organ' analogies.
