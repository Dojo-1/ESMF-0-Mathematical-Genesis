# G0-AG D1 — A behavioral test before any further anatomical or algebraic lift

**Status: preregistration proposal, not implemented, not executed and not frozen. Evidence H.** The G0-AG D0 negative/inconclusive measurement remains intact. This is a *different* test, not a new horizon fitted to rescue D0.

## A missing observable

The current G0-AG program can increase/decrease glial support, change quaternions and grow paid branches. It has no independently scored terminal behavior establishing that a new representation improves the organism's decisions. Developmental complexity, contact lifetime, and lower geometric discrepancy are **not** interchangeable with competence. The next comparator must observe a decision the organism is asked to make *before* the evaluator reveals its outcome.

Start with a deliberately modest **context-conditional invariance forecasting** task. The organism receives a context token c, an interrogative/probe mode q, and past before/after *observable responses*. It does **not** receive the hidden response probability or the held-out answer. It must predict whether a future encounter under that same context and probe mode will change its observed response.

This evaluates acquired task-relative distinctions but **not endogenous discovery of the context IDs themselves**. That stronger objective remains a separate unsatisfied requirement.

## Environment and blind scoring

Use two declared synthetic contexts and two Lullian probe modes already present in G0-AE's external test environment. The evaluator knows a conditional distribution P(change | c,q), samples new independent held-out before/after responses and reveals them only *after* the prediction. No prediction method may read hidden environment probabilities, test labels, event indices other than those it actually experienced, or an unobserved future state.

Synthetic task reference probabilities are based on the previously committed example_world specification:

| Context | Question mode | P(observed change) |
|---|---|---:|
| 0 | 0 | 0.85 |
| 0 | 1 | 0.55 |
| 1 | 0 | 0.15 |
| 1 | 1 | 0.45 |

This is synthetic, not human-validated semantic truth. The score is **Brier loss** for a future observed binary change, plus calibrated coverage and false confident invariance/contrariety declarations. Report individual context/probe modes, not only a pooled scalar.

## Equal-information memory controls

- **Current ESMF relational memory:** for context c, G0-AE Distinction maintains seen_same and seen_different (with live salience), pooled across interrogative modes. Derive a smoothed predictive rate from those recorded counts, no hidden generator probabilities.
- **Question-conditioned ordinary estimator:** two capped counts (same/different) for each (context, question) cell. With two contexts and two question modes this uses eight bounded count slots, compared to the current two-context G0-AE memory's eight seen/live count slots. Both see **the same ordered observed encounters**; the baseline cannot inspect unobserved sensory outcomes or semantic ground truth.
- **Question-shuffled null:** the same capped memory and same observed encounter budget, but scramble the question assignments without changing the observed response sequence.
- **Equivalent real arithmetic:** any transport-dependent prediction must be run in an algebraically identical four-real-coordinate Hamilton representation as a representation-invariance control.
- **Geometry and glial ablations:** when later tested end-to-end, keep the same event observation stream and *realized* external resource inputs; compare inference effects separately from differences in sensing opportunities.

Do not claim full architecture capacity parity from matching only memory count slots: neurons, quaternion sections and glial pools add compute and state. Report all such mismatches explicitly.

## Ordering and validation gates

**D1a (offline first):** feed an identical predeclared encounter trace to the current pooled memory and the question-conditioned control. Evaluate predictions on fresh held-out outcomes from the four-mode evaluator. Use only development seeds 8150–8163 for instrument tests; reserve the original G0-AG1 heldout seeds 18100–18227 and do not inspect them during the development gate. Track Brier loss, coverage, false equivalences and the raw number of observations per mode.

**D1b (embodied only if D1a is valid):** allow the G0-AG generator to select and pay for its sensory encounters and structural edits, then evaluate at a fixed common external resource *spending* budget. Compare full, no-glia-rate, no-geometry-rate, fixed anatomy and the same-capacity conventional learner on an independent held-out task. Equal random seeds are insufficient if spending or observation histories differ; report realized parity and do not adjust scores to force equality.

**Causal tests:** evaluate whether the additional geometry changes *what the organism can predict or do*, not just its exposure time to a contact. A low Brier score from observing a label is invalid: record the prediction and time before sampling or revealing its test outcome. Cap and report class imbalance and abstention.

**Stop/kill rule:** if a simple question-conditioned estimator with the same evidence and lower compute is at least as competent, the stronger claim of special hypercomplex semantic learning must be retired. If a richer manifold or cell type becomes desirable, demand a measured representational obstruction in D1 first. Never introduce a new algebra merely because D0 was underpowered.


## D1a development configuration — frozen before instrument implementation

This subsection **fixes the D1a development probe before its first run**. It does not freeze or execute D1b or the original G0-AG1 held-out programme.

- Seeds: **8150–8163 inclusive** (14 distinct development streams), never 18100–18227.
- Training: **six observable encounters per (context, question mode)** cell; 24 training encounters per stream, in a predeclared balanced 0/1 context × 0/1 question schedule. Outcomes are independent Bernoulli draws from the external evaluator's table above. Input events expose context, question, and actual changed/unchanged response, but never the probability.
- Testing: **32 independent, future outcomes per cell** per stream; 128 held-out-from-training observations per stream. Test outcomes are generated **after the predictors have committed their predictions**. This is a development-split score, not the untouched G0-AG1 holdout.
- Two prediction rules: G0-AE's actual Distinction raw seen_changed / seen_same capped at **3** per context, pooled across modes, versus a conventional question-conditioned estimator with two counters capped at **3** per (context, mode) cell.
- Both use the same **Beta(1,1) smoothed change probability**, so difference reflects retained question information rather than a privileged readout. G0-AE has two contexts × (seen_same, seen_changed, live_same, live_changed) = **8 capped integer memory slots**; the conventional estimator has two contexts × two modes × (same, changed) = **8 capped integer slots**. This equalizes slot count but not other architectural overhead.
- Negative control: same estimator and exact observed training sequence, with question labels permuted *within each context* by a separate fixed RNG. Must not use test outcomes or hidden probabilities to permute.
- Primary observable: **mean future-outcome Brier score**, averaged with equal weight over four context/mode cells. Paired differences across 14 streams are descriptive with normal-approximation 95% intervals; no confirmatory p-value.
- Secondary: expected Brier under the independent evaluator's reference distribution (analytic known answer), conditional per-cell Brier, number of confident declarations at thresholds <=0.25 or >=0.75, and total memory slots. Preserve raw predictions and outcomes.
- Full placebo: rerun each predictor on the same source trace; identical output required. Do not use a hidden task probability in either prediction implementation.
- A better conventional estimator is a **negative result for any claim that the current pooled memory is already sufficient to recognize probe-dependent relationships**. It does not establish that glial/quaternionic geometry is ineffective, because D1a is an offline memory-only diagnostic.
- Kill gate: if the ordinary estimator ties or wins at the same observations and eight integer slots, do **not** add another algebra to rescue pooled semantic memory; identify and explicitly repair the lost contextual distinction first. D1b cannot begin until D1a's comparator validity and provenance pass.
