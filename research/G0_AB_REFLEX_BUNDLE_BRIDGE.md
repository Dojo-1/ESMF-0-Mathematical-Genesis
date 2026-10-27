# G0-AB reflex / bundle bridge

This is a new, unvalidated experimental extension. It does not revise the
frozen G0-AB held-out results in
[G0_AB_2026-10-25.md](results/G0_AB_2026-10-25.md), and its smoke tests are not
held-out evidence.

## Implemented path

`esmf/epistemic_circuits_ab.py` couples a finite-state alternating X/Y probe
circuit to the charted rank-four neutral bundle and partial sensor interface
from G0-AC0. The environment transports full four-component fiber sections
with local chart frames, transformed covectors, the neutral metric, and the
orientation-reversing cylinder seam; two noisy linear sensor readings are
available per transport. Observation delivery is delayed by one fast tick.
After the 60 action opportunities, a separately accounted no-action tick
delivers the final sample so no observation or audit is processed beyond the
resource ledger.
The experimental `nonlinear_drift` world adds a nonlinear coefficient
trajectory; `noise_mismatch` adds shared-channel noise and occasional
outliers.

The fast policy alternates its action axis without enumerating Bayesian
experiment plans. A three-model stable/drift/reset likelihood audit runs only
on the first observation, a high standardized prediction error, ambiguous
mode evidence, or missing provenance. A mesoscopic hysteretic gate updates
precision every two fast events. The ordinary-real recursive least-squares
connection estimator receives each delivered observation in all policies.
Audit likelihoods compare a three-mode Gaussian residual covariance inflation
scale (1, 4, or 25) against the full bivariate output-space predictive
covariance, including shared coefficient uncertainty.
Controls are fixed-focus sensing and posterior-threshold auditing. The
`gaussian_state_space` arm uses constant-forgetting Gaussian RLS and audits
every delivered event; the other arms use the existing adaptive RLS update
with their named audit policy. All use two real coefficient states and share
the resource cap, supply, sensing-cost law, and horizon; realized spend may
differ. Evaluation error uses a separately sampled, noise-free probe section
after the burn-in, not the training probe.

The ledger accounts for supply, overflow, sensor and audit charges, and a
positive reserve floor. `benchmark()` reports wall time and Python-tracked
peak memory; `joules` is deliberately `None`. No watt or energy-efficiency
claim is supported. Torsion is not defined by this vector-bundle connection
alone; curvature is a separate, existing G0-AC0 quantity. Positive resource
accounting is never derived from the indefinite neutral form.

Run the focused checks:

```bash
python -m unittest discover -s tests -p 'test_epistemic_circuits_ab.py' -v
python -m esmf.epistemic_circuits_ab --seeds 8 --worlds static,abrupt_step,nonlinear_drift,noise_mismatch
```

The second command is a local comparison/smoke runner, not a frozen
held-out evaluation. Its per-policy output includes mean prediction error,
audit/focus counts, abstract resource spend, wall time, traced peak memory,
and `joules: null`.

## Still open

This bridge does **not** implement evidence-gated Worldlet creation/merging or
timed return-map remainders; a provenance posterior over change boundaries;
agent-selected experiments with separate `I(T;Y)`, `I(C;Y)`, `I(K;Y)` and
history-conditional information; independent causal-transfer promotion;
calibrated abstention on held-out data; or an electrochemical/developmental
organism. It does not compare against an equally informed conventional
recurrent/state-space family beyond its matched RLS controls. The bundle is a
mathematical testbed, not physical tissue. The simulation crosses the G0-Q
orientation-reversing reflection seam; that does not yet compose G0-W's exact
Mat₂ transpose operator with its left/right action conjugacy in this learner.
The noise-mismatch world is a falsification case; the audit does not learn an
unknown sensor-correlation matrix.
These gaps must be addressed with a new frozen protocol and untouched
evaluation data before making broader claims. In particular, do not claim
semantic intelligence, learned topology, biological embodiment, or hardware
efficiency from these surrogate runs.
