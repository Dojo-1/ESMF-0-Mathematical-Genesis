# G0-AC0 · Adaptive neutral bundle connection — pre-registered

Date 2026-10-25. Written *before* G0-AC0 implementation or remote execution. G0-AB held-out test is complete; this is a DISTINCT geometric-calibration experiment with a newly frozen design.

## Objective

Test whether a **true rank-four neutral `(2,2)` connection** can adapt to changing transport laws while remaining consistent under independent, position-dependent local frames. Avoid inferring an advantage over an equivalent real state-space estimator simply from bundle notation.

## Actual fiber bundle object

Inherit the explicitly charted G0-Q bundle over `M=S1×R` with rank-four real fiber, local neutral form eta=diag(+,+,−,−), three position-dependent nonorthogonal GL4 local frames, and nonorientable seam monodromy R=diag(−1,1,1,1). The two dynamics generators

`X=E_{1,2}+E_{2,1}`, `Y=E_{1,3}+E_{3,1}`

are in o(2,2), commute with R, and have nonzero commutator [X,Y] (indices zero based). For time-varying real coefficients a_t,b_t:

`A_t = a_t X dtheta + b_t Y dz`, `F_t=dA_t+A_t^2 = a_t b_t [X,Y] dtheta∧dz`

on each *instantaneous spatial base slice*, because a_t,b_t vary in external developmental time, not as functions of base theta,z. Time variation defines a family of spatial connections; no theorem about a connection over the extended time manifold is claimed.

A probe `v` injected at a source local chart is transported on a short theta- or z-segment:

`U_ref=exp(−h*a_t*X)` or `exp(−h*b_t*Y)`.
`U_local=B_target U_ref B_source^{-1}`.

Measurements are ONLY two partial *linear sensor functionals* on transported v, not a full revealed transport matrix. Each target local sensor functional is transformed contragrediently `c_local=B_target^{-T} c_ref`. This ensures the physically observable scalar is chart invariant. Intrinsic neutral bilinear form and conjugated connection curvature must remain consistent when coefficients change.

The global topological twist remains fixed (nonorientable): we adapt the *connection*, not the isomorphism class of the bundle. No context-dependent change of rank or topology is asserted.

## Environment / frozen hypotheses

Time horizon 60 transport opportunities. True connection evolves in five worlds:
1. STATIC: a=.4,b=−.32.
2. SMOOTH_DRIFT: time sinusoidal changes.
3. ABRUPT_STEP at t=30.
4. SIGN_REVERSAL at t=30.
5. RAPIDLY_VARIABLE coefficients sampled from independent bounded distributions (negative control; smooth learners should struggle).

External simulation independently samples the input test sections and measures a low-dimensional projection with Gaussian noise std .02. Local source/target charts are randomized from valid overlapping charts; base remains near theta=pi where all three charts have valid lifts. New, independent held-out sections are sampled for evaluation and never used in parameter updates.

Four matched TWO-PARAMETER learners share the same observation streams and fixed input representation:
- ADAPTIVE: online recursive least squares (RLS) with bounded variance inflation after large innovations and normal background drift allowance.
- CONSTANT_FORGETTING: online RLS with fixed positive process variance.
- NO_FORGETTING: online RLS with zero process variance.
- FROZEN_AFTER_CALIBRATION: update only first 12 pulses then freeze.
Additional ORACLE true coefficients (privileged upper bound) and EQUIVALENT_REAL identical algorithm are diagnostic controls, NOT competitors for practical gain.

The adaptive policy should win against no-forgetting/frozen controls when connection law genuinely changes; it **need not** beat them in STATIC where forgetting causes variance and bias. If a simple constant-forgetting RLS ties or beats adaptive, the fancy 'surprise-triggered' rule has no demonstrated value and should be simplified.

## Pre-declared metrics and kill conditions

- Noise-free held-out-section mean squared transport prediction error, measured after first 12 training steps (t=12..59).
- True and learned a,b, mean absolute coefficient error.
- Curvature `a*b [X,Y]` gauge law; local metric compatibility, endpoint representation covariance, winding-monodromy sign when connection varies slowly in time.
- Resilience to independent chart choices and finite-precision numerical changes.
- Per-run CPU wall time and model-state count; NO physical joule measurement.
- Equal two real trainable coefficients; RLS covariance hyperparameters are fixed by protocol and count as design complexity.
- Independent finite-difference check of partial-observation gradients, deliberate wrong-metric/flat-transport negative control.
- At least 32 paired seeds, with seed-cluster bootstrap for competing losses.

Kill if covariance is violated, if nonflat curvature is mislabeled from a gauge artifact, if the adaptive law diverges, if any chart reparameterization changes physically observable prediction, or if partial observations do not permit identification of the parameters. No success may be claimed for emergence of semantics, biology, polyvisceral cognition or superior energy use.

## Status authority

[P] conditional identities for metric-compatible connection, noncommutative curvature and chart covariance; [A] proposed time-varying connection and Gaussian sensors; [M] only later verified remote runs; [H] benefits beyond conventional coordinate-aware RLS. A connected structural body is not yet assembled: G0-AB and G0-AC0 are orthogonal tested components of a future coupled organism.

## Frozen evaluation appendix (recorded after three-seed smoke, before held-out)

The first successful three-seed diagnostic is available at https://huggingface.co/jobs/jesusvilela/6ade17211e112550ce7ebcad . It verified chart covariance, metric transport and nonzero curvature, while showing that constant-forgetting RLS can beat the more sophisticated surprise-triggered update in abrupt/sign-reversal settings. **Do not tune the controller against those results and then re-label them validation.** The implementation's hyperparameters (`SD=.02`, `HORIZON=60`, `BURN_IN=12`, adaptive q=.0008+.04·burst, threshold 6, burst 5; constant q=.01) remain frozen before the next run.

Hold-out seeds: **100 through 131 inclusive (32 distinct seeds)**, all five worlds and all five policy labels. Evaluate noise-free independent probe MSE on t=12..59, mean coefficient and curvature errors, per-seed paired bootstrap (2,000 seed resamples), and exact equivalence against the ordinary real estimator. The three-seed smoke is development data, not held-out evidence. No new policy tuning after seeing seeds 100..131 may count as validation on the same seeds.
