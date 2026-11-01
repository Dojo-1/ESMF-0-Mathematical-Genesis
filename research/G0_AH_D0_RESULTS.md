# G0-AH D0 — The first compartmental-neurite comparison

**Executed on GitHub Actions, 1 November 2026.** [Workflow 41325478388](https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis/actions/runs/41325478388), commit `374e9dc18829aacd2b7b8aa6e17da9ae1c0f3036`, full per-seed JSON in Actions artifact **g0-ah-d0-compartmental-development** (ID 13090774724). The source JSON checksum reported by CI:

```
SHA-256  4c87858393622cf30fced16016a0c1708858f8ea98629929866ac750986c8191
```

Protocol [G0_AH_DESIGN_AND_PROTOCOL.md](G0_AH_DESIGN_AND_PROTOCOL.md) was committed before this executed test. All 14 declared development seeds (8300–8313) ran without filtering. The original G0-AG1 held-out seeds remain untouched.

The important observation is **negative**: a smaller, ordinary, fully cue-conditioned memory performed *better* than the branched organism in this task. This architecture has not yet established a cognitive, geometric or biological advantage. Unlike G0-AF/AG, the new AH reference is a discrete selected-operator model, not a continuous-time Markov generator.

## Measured synthetic future-response prediction

Four contexts × four questions × four two-bit cue combinations; six witnessed training outcomes per cell, then 32 new outcomes per cell, drawn **after predictions were fixed**. Thus 384 training observations and 2,048 future tests per seed.

| Model | Mean Brier ↓ | Descriptive 95% interval | Role |
|---|---:|---|---|
| **Ordinary (context, question, cue) counter** | **0.176993** | [0.173503, 0.180483] | Primary, strong semantic baseline |
| G0-AH with paid branches and glial consolidation | 0.185233 | [0.182245, 0.188221] | Full compartment model |
| G0-AH with slow consolidation disabled | 0.185233 | [0.182245, 0.188221] | Resource-matched glial ablation |
| Question-conditioned but cue-pooled counter | 0.250054 | [0.249948, 0.250161] | Weaker information-loss control |

The paired **full G0-AH minus ordinary cue table** was **+0.00823969 Brier** with a descriptive normal-approximation interval [0.00742625, 0.00905313] across the 14 development streams. Positive means G0-AH was worse. This is not a confirmatory significance claim and does not imply the two algorithms perform similarly on other tasks.

The two AH arms received **exactly the same observations, branch growth sequence, 64 final branches, glial pulse schedule and integer resource expenditure**. They did not differ in scored predictions. Although their internal *fast versus consolidated* count arrangements can differ, this short, stationary task does not expose a benefit of that distinction.

The conventional estimator received the same observed labels and question/cue contexts, and uses **128 bounded witness-counter slots**. AH allocates **256 fast/slow witness slots**, plus 64 actual branch objects, quaternionic phases/connection states, four glial territories and paid material. Computational and representational capacity are **not** matched between AH and that control, and the conventional estimator does not pay the model's synthetic ATP. Do not claim an energy-adjusted comparison.

## What did and did not happen

- **Implemented and tested:** 16 somatic cells, 64 paid 3D tree branches, four glial territories, local nonlinear cue conjunction, question-indexed witness counts, distinct fast/slow clocks, typed loss on memory saturation and pruning, gauge-covariant live quaternion phase updates, directed delayed excitatory/inhibitory synapse operators and closed synthetic energy/material ledgers.
- **Measured in the behavioral task:** tree growth, witness accumulation, paid glial pulses and slow consolidation, followed by fresh future-response predictions.
- **Not exercised in that task:** the axonal inhibitory-delay circuit, adaptive synapse acquisition, pruning as a way to improve prediction, independently discovered contexts, or biological physiology. Unit tests of these operators are *mechanistic contracts*, not tests of cognitive benefit.
- **No source-of-truth leap:** quaternionic phase arithmetic is equivalent to a suitable real-coordinate multiplication. Tree edges do not establish nontrivial cycle holonomy. The 3D chart is nonperiodic but is not measured neural anatomy.

## Decision

G0-AH D0 did not justify the claim that neuritic branching brings a predictive advantage on a conditionally independent Bernoulli table. **Retire that specific advantage claim**, retain its implemented mechanisms, and test a different task only after specifying an independent, necessary temporal conjunction or routed local signal that a properly matched ordinary sequence model cannot already exploit at the same compute and memory budget.

A useful next falsifier is a **time-lagged, partially observed cue** arriving at distinct branches. Test local plateaus, inhibitory veto, branch specialization, glial eligibility windows and costly repair *together*, against a conventional recurrent model with the same signal history and equal resource-spending controls. If that model ties or wins, do not grow anatomy or introduce another algebra to rescue the hypothesis.

**Evidence map:** P (conditional ledger and gauge identities), A (observation-to-quaternion adapter), M (bounded synthetic D0 outcomes and structural tests), H (usefulness of growing local dendrites), S (neuron/glia biological analogy), R (claims that branching alone improves this independently scored conditional-table task).
