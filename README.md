# ESMF-0 — Mathematical Genesis

**An experimental substrate for endogenous semiotic morphogenesis.**

ESMF-0 asks whether locally adaptive predictive relations, with bounded resources and dynamic topology, can develop reusable distinctions between causally different encounters. A word, class label, or token is **not** the unit of learning. Initially, the organism sees observations and interventions; latent causal rules are hidden and used only by the evaluator.

## Evidence boundary

- **P (conditional mathematics):** bounded projection; finite resource/accounting identities; predictive-equivalence quotient; Gram positive semidefiniteness under the stated assumptions.
- **M (measurement):** reserved for actual recorded runs with seeds, environment, and matched controls.
- **H (hypotheses):** morphology improves held-out prediction; endogenous dynamics generate transferable concepts or useful compositional novelty.
- **S (semantic):** Worldlets, tendrils, and creative organization are design vocabulary, not biological claims.
- The connection-Laplacian Lean results and tenth-discriminant jet-Gram results are *prior mathematics*, not automatically verified theorems about ESMF.

## Current empirical finding

**G0-A was a misleading comparator.** A scripted-edge model appeared better than an empty fixed graph. G0-B then added the missing one-edge fixed control: it slightly **outperformed scripted growth on all eight paired seeds**. The result does **not** establish endogenous learning; it corrects the causal interpretation of the earlier smoke test. See [G0-B's full negative-control record](research/results/G0_B_EDGE_PARITY_2026-10-25.md).

## Developmental surrogate and gauge freedom

**G0-D** now simulates local trophic diffusion, growing tips, exploratory branching, activity-dependent stabilization/pruning, maturity and limited contacts. Six remote CPU runs exhibit both contact formation and pruning—but also high turnover and weak resource pressure. **This is not a realistic brain simulation.**

**G0-E** adds independent local basis transforms with properly transformed metrics and transports. External checks confirm transported edge-energy invariance and the correct Gram determinant transformation. Arbitrary GL chart changes do **not** preserve a raw Gram determinant.

[Development notes](research/G0_D_DEVELOPMENT.md) · [Gauge freedom](research/G0_E_GAUGE.md) · [Measured results](research/results/G0_D_E_2026-10-25.md)

```bash
python -m esmf.development --steps 300 --seeds 0,1,2,3
python -m esmf.gauge
```

## Mechanochemical status: a useful failure

[G0-F](esmf/mechanochemical.py) introduces two reaction-diffusion fields, receptor-dependent stochastic guidance, bounded motility and resource-limited contact growth. The first remote CPU test uncovered **metabolic depletion and contact collapse**. Chemical mass accounting remained numerically consistent; no stable anatomy or semantic benefit has been demonstrated. [Read the numerical audit](research/results/G0_F_2026-10-25.md).

```bash
python -m esmf.mechanochemical --steps 300 --seeds 0,1,2,3
```

## Geometric anti-collapse audit — G0-K

The biological simulator currently uses a 2D periodic mesh and scalar activity. **This is a numerical approximation to external tissue chemistry, not the ontology of cognition.** [G0-K](research/G0_K_GEOMETRIC_AUDIT.md) adds a separate, tested matrix-fiber layer with independent GL(n) frames, a (2,1) indefinite form, hyperboloid distances, gauge-covariant transports, holonomy conjugation, and exact lazy decay for quiescent compartments. [Remote numerical self-check](https://huggingface.co/jobs/jesusvilela/6addd66b1e112550ce7ea222) gave residuals near machine precision.

These are **algebra checks**, not evidence of cognitive efficacy or real energy efficiency. Indefinite mismatch is not a positive Lyapunov energy; applying a dissipative sign assumption to it would be invalid.

```bash
python -m esmf.geometry_k
```

## G0-L · Heterogeneous transport without hidden flattening

[Worldlet correspondence](esmf/remainder.py) permits independent local dimensions, clocks, metrics, forward maps and reconstruction maps. [Gauge-aware residual](research/G0_L_TRANSPORT_REMAINDER.md) is typed and transported rather than silently discarded. A remote CPU [numerical check](research/results/G0_L_2026-10-25.md) passed for a 3D-to-2D lossy correspondence. This is **mathematical infrastructure**, not demonstrated cognition or efficiency.

```bash
python -m esmf.remainder
```

## G0-M · Neutral-signature (2,2) cognitive carriers

[Real (2,2) test geometry](esmf/neutral22.py) replaces planar assumptions **in the new algebraic test layer**, not yet throughout mechanochemical dynamics. Its (O(2,2)) isometries, nonzero null states, and metric-covariant transports are checked remotely on CPU; the [mathematical audit](research/G0_M_NEUTRAL_22.md) documents the distinction between an indefinite signed form and a positive physical resource norm. It does not yet establish semantic, cognitive, or efficiency benefit.

```bash
python -m esmf.neutral22
```

## G0-N · Representation comparisons (preliminary)

A [CPU-only paired benchmark](esmf/representations_n.py) compares equal four-real-parameter Euclidean, neutral (2,2), and complex-block real encodings on the same action-conditioned task. [Eight-seed results](research/results/G0_N_2026-10-25.md): Brier 0.00724437 (Euclidean), 0.01271116 (neutral), 0.00724437 (complex-block). **This is not a gauge-fair comparison:** Euclidean SGD and L2 weight decay change behavior under nonorthogonal O(2,2) boosts. The observed difference is a conditioning result, not a cognitive verdict.

```bash
python -m esmf.representations_n --seeds 0,1,2,3,4,5,6,7
```

## G0-P · Bundle atlas and nonflat neutral connection

[G0-P](esmf/bundle_p.py) implements constant O(2,2) charts, consistent transition cocycles, metric-compatible connection generators, noncommuting curvature and loop holonomy. A [remote CPU audit](research/results/G0_P_2026-10-25.md) caught and corrected an orientation-sign error; the corrected implementation passes gauge and small-loop convergence checks. **This is a trivial bundle testbed with nonflat connection**, not yet a fully general bundle, adaptive cognition, or biological embodiment.

```bash
python -m esmf.bundle_p
```

## G0 test ladder

1. **G0-A:** prove/check state bounds, degree/resource limits, deterministic replays, and numerical stability.
2. **G0-B:** compare fixed and dynamic topology on held-out interventions, with appearance nuisances to prevent superficial classification.
3. **G0-C:** test withheld causal compositions and damage recovery; require measured gains beyond equal-budget controls.

The first simulator is a deliberately small **engineering baseline**, not proof of concept emergence. Never infer meaning from graph shape, oscillation, or an attractive plot.

## Reproduce (CPU-only, no packages)

```bash
python -m unittest discover -s tests -v
python -m esmf.experiment --seeds 0,1,2,3 --episodes 600
python -m esmf.controls --seeds 0,1,2,3,4,5,6,7 --episodes 600
```

Run defaults use standard Python, no GPU, no LM Studio, no remote LLM, no background services, and no filesystem write. The Kaggle machine is not used by Hugging Face-hosted Jobs.

See [G0 protocol](research/G0_PROTOCOL.md) for fixed predictions, controls, falsifiers and what must be recorded before any claim promotion.

## Architecture constraint

No global spectral oracle or determinant maximizer inside the learner. The learner accesses only local observations, messages, interventions and bounded plasticity signals. Connection Laplacian and observation Gram are **external diagnostics**. Z+ is treated as a prospective two-channel possibilistic/probabilistic construction, **not** an implemented Zadeh-faithful uncertainty calculus in the first baseline.

## Related mathematics

- [Connection Laplacian formalization](https://github.com/jesusvilela/connection_laplacian_lean)
- [Tenth discriminant trace-Gram formalization](https://github.com/Dojo-1/tenth-discriminant)

## Safety and research discipline

No autonomous local execution, GPU access, paid inference or Kaggle process interference. Do not silently upgrade an experiment to a proof. Negative and inconclusive results belong in the record.
