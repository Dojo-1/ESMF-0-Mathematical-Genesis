# G0-AG D1a — Question-conditioned memory: observed development result

**Executed 1 November 2026**, GitHub Actions [run 41297577245](https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis/actions/runs/41297577245), code from draft PR #20. The complete raw JSON (observed training events, predictions committed before outcomes, future test responses and per-cell scores) is artifact **g0-ag-d1a-memory-results**, ID 13080041522. Extracted file g0_ag_d1a_results.json is 125,939 bytes; SHA-256:

    b173dd490a8e739ddc2ebd0f88f328c934fb642669a6ca4db1c5ab956f43e3be

The prospective [D1a configuration](G0_AG_D1_BEHAVIORAL_PROTOCOL.md) was committed **before code and execution**. Seeds **8150–8163** are development-only; original G0-AG1 held-out seeds 18100–18227 remain untouched.

## Verdict — useful negative result

**M (bounded synthetic task measurement).** The current G0-AE Distinction memory pools responses across Lullian interrogative/question modes. A conventional mode-conditioned memory with the **same eight capped integer slots**, given **exactly the same 24 observed training encounters**, predicts new mode-specific response changes more accurately in this artificial four-cell environment.

This is direct evidence against the claim that the existing pooled memory is sufficient to retain *all task-relevant question-conditioned behavioral distinctions*. It is **not** evidence of superiority of a hypercomplex tissue or of an AI learning its own context boundaries. There is no biological physiology, quantum processing or endogenous sensory choice in the offline D1a measurement.

## What was measured

- 14 paired development streams, each with six observed training encounters in each of the four (context, question mode) cells.
- 32 independent *future* response changes/unchanged events per cell. Predictions were fixed **before** those outcomes were generated.
- A context-pooled memory using the actual G0-AE Distinction update, capped at 3 seen/live witnesses per context.
- A question-conditioned ordinary Bernoulli memory, with two capped counters per (context, mode).
- A question-label-shuffled conventional memory as a negative control.
- The same Beta(1,1)-smoothed predictive probability for all three models.

Each model has eight integer slots in its memory representation. This compares **the organization and retained relation type**, not the total computational overhead of the full G0-AE/G0-AG architecture.

### Primary future-response Brier scores (lower is better)

| Memory | Future-outcome Brier | Analytic expected Brier |
|---|---:|---:|
| Current context-pooled G0-AE | **0.242629** | 0.244420 |
| Ordinary question-conditioned | **0.208190** | 0.208130 |
| Question-label-shuffled control | **0.228841** | 0.230173 |

Paired across 14 seeds, **pooled minus question-conditioned = +0.034440 Brier** (descriptive normal-approximation 95% interval **[+0.025956, +0.042923]**). Question-shuffled minus question-conditioned = **+0.020652** ([+0.012527, +0.028777]). All **14/14** development streams favored the question-conditioned model on the *four-cell aggregate*. These are descriptive paired development measurements, **not** confirmatory population inference.

### Do not hide the mode-specific trade-off

| Context:question | P(change) in evaluator | Pooled Brier | Conditioned Brier | Shuffled Brier |
|---|---:|---:|---:|---:|
| 0:0 | 0.85 | 0.233520 | **0.166568** | 0.192789 |
| 0:1 | 0.55 | **0.247019** | 0.258891 | 0.256298 |
| 1:0 | 0.15 | 0.242438 | **0.156348** | 0.211106 |
| 1:1 | 0.45 | **0.247540** | 0.250952 | 0.255172 |

The conventional estimator wins primarily on *strongly differentiated* question modes (0:0 and 1:0), but has **higher** empirical loss on the more ambiguous modes (0:1 and 1:1). No universal per-context superiority is claimed. Regularization, caps and small data can change local trade-offs.

## An independent known-answer counterworld

Give one context six changed encounters under mode 0 and six unchanged encounters under mode 1. The existing Distinction saturates its same/different counters at 3 and assigns *the same* smoothed probability **0.5** to both probes. A separately indexed but equally sized ordinary memory retains **0.8** for mode 0 and **0.2** for mode 1.

This is a **P (conditional elementary)** counterexample to representational sufficiency: a mode-ignorant state cannot encode both conditional distinctions in this case. The D1a development measurement verifies that the omission can matter on a finite stochastic forecasting task. It does not imply an information-theoretic lower bound on all conceivable pooled networks or a theorem of true semantic meaning.

## Controls, resources and remainder

- **Equal observed data and prediction timing:** yes; predictions were committed before independent future test draws.
- **Equal integer slot count:** yes, eight per model; different allocations of seen/live versus question-specific evidence. Not equality of runtime, weight complexity, anatomical costs or expressive neural capacity.
- **Independent reference:** analytic Bernoulli expected Brier computed in the evaluator agrees in direction with finite future-response Brier. This is a separate calculation but shares the same author-chosen artificial distribution, not independent external replication.
- **Negative control:** question-label shuffling degrades performance relative to retaining mode labels on the four-cell aggregate, as preregistered.
- **Held-out task:** independent future **responses** under fixed synthetic contexts, but no genuinely unseen contexts or full organism world dynamics.
- **Missing biological budget:** this is deliberately offline; glial and neurite energy expenditure were *not evaluated*.
- **Semantic remainder:** changed/unchanged binary outcomes capture a task-specific behavioral relation, not the full meaning of a Worldlet; context IDs are supplied rather than discovered.

## Decision and evidence governance

**Retire (R) the sufficiency claim for context-only pooled behavioral memories** under tasks with genuine mode-dependent observations. Do not attempt to rescue that failure by adding a new quaternion, octonion or dimension.

The constructive next integration is *typed, query-indexed* local relational memory with provenance and dissipation maintained separately, while retaining the prior G0-AE data model as a frozen control. A succeeding, independently scored **embodied** test must determine whether that memory is useful when ATP, glial contact, neurite growth and observation opportunities are *actually coupled and resource-matched*. Until then, no evidence of full ESMF-0 semantic/biological integration is promoted.

**P:** exact counterexample and conditional algebraic predictions. **M:** D1a development scores on the specified synthetic evaluator. **A:** artificial before/after response relation as a proxy for meaning. **H:** useful emergent recognition, endogenous context growth or biological advantage. **R:** pooled-per-context Distinction is sufficient for all question-conditioned tasks.
