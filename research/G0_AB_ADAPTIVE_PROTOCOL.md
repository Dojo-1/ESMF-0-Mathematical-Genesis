# G0-AB · Adaptive organ ecology — pre-registered streaming experiment

2026-10-25. Registered BEFORE any G0-AB source implementation or remote run. Do not retroactively replace predictions after observing results.

## User requirement

Adaptation must change the organism's *subsequent state-transition policy and resource allocation*, not merely its posterior P(T,D|Y). Inception-stage minimal mechanistic surrogate: fast sensorimotor decision, mesoscopic autonomic precision gate, slow homeostatic reset-hazard plasticity and typed local-context budding, all under a finite external resource ledger. No neuron/glia/gut equivalence; no demonstrated consciousness, intelligence, or actual hardware energy saving.

## Mathematical structure and topology

Hidden global bundle orientation T∈{+1,-1}, fixed per trial, affects odd-winding observation. Local source contrast D_t is one measurable quotient of a 4D neutral Mat2(R) carrier Q(F)=det F (signature 2,2), with transpose monodromy preserving Q. We may embed D_t in SL2(R) as F_t=[[1,D_t],[0,1]], yielding det F_t=1 for every latent state, *but this is only a 1D unipotent orbit inside 4D*, not full four-dimensional learned cognition. Context drift induces relative transport F_t F_(t−1)^−1; under identity monodromy it is a shear, under transpose it is the conjugate opposite shear. Endogenous adaptation here is in the probabilistic filter and policy, not yet in a learned connection/metric.

Four synthetic hidden source regimes (all withheld from the inference engine):
- STABLE: D_t = D_(t−1) + ε, small process noise (variance ≈0.0008).
- DRIFT: D_t = 0.96 D_(t−1) + ε, noise variance ≈0.045.
- JUMPS: D_t mostly stable, but reset to an independent N(0,1) source at two predetermined **evaluation-only** times.
- DIFFUSE: a new independent N(0,1) source each event. Orientation T is statistically nonidentifiable, even if the simulator internally generates a bit.
- SWITCH: stable initial phase, drift later (OOD temporal switching).

Probe event y_t=sign(T,parity_t) D_t + Gaussian noise with quiet/focused precision. The same neutral-seam parity convention as G0-W/X is used. Realistic dynamics and hidden physical state access are explicitly *not* asserted.

## Filter / local learning law

An interacting multiple-model (IMM) Kalman mixture with six Gaussian local hypotheses: orientation T × continuity regime r∈{stable,drift,reset}. A Markov transition matrix couples regimes. Each component predicts its local D and covariance, then updates a Gaussian likelihood; mixture weights preserve uncertainty over orientation. Mixture-to-mixture mode transport is moment matched, with explicit information-loss remainder [R]_transport (not exact Bayesian inference). A reset candidate imposes D∼N(0,1).

Controls will include the same filter with a FIXED transition hazard, allowing attribution of slow hazard learning separately from sensing policy. The hazard update is an explicitly labeled heuristic using past reset evidence, not an exact posterior update: once every four fast events adjust prior reset probability within a bounded range, *for use in subsequent predictions only*. No double reuse of the current observation in the same filtering step.

Three distinct clocks must have observable effects:
- FAST every event: motor/sensor choice, resource supply/consumption, Bayesian state prediction and observation.
- MESO every two events: autonomic gate changes quiet/focus based on normalized predictive surprise with hysteresis and reserve.
- SLOW every four events: plastic reset hazard alters future transitions and may emit a typed temporary Worldlet split when sequential posterior reset evidence reaches a threshold.

A resource ledger includes delivered external nutrient supply, maintenance, sensing and focal-precision charges, cap overflow and remaining reserve. All policies share the same physical budget and sensor-noise laws. Energy is abstract simulated resource, **never watts**.

## Preregistered opponents

1. FIXED_FOCUS: strict parity alternation; always focused when affordable.
2. FIXED_QUIET: strict parity alternation; always quiet.
3. ADAPTIVE: fast alternating parity, mesoscopic surprise-activated precision, slow reset-hazard plasticity with identical six-state IMM.
4. ADAPTIVE_NO_SLOW: same adaptive sensing gate but fixed reset-hazard transition.
5. RANDOM: random parity/precision given affordability, same IMM.
6. EQUIVALENT_REAL: identical posterior, policy and state equations, only no bundle metaphor. Predictions must tie ADAPTIVE.
7. Optional simplified posterior-only/reflex baseline. If the simplest policy dominates, record that failure.

These variants have equal inference state dimension and model-family capacity except for the explicitly ablated slow parameter update. Policy is evaluated under the same opportunity, episode horizon, available probe menu, initial prior, sensory calibration and resource ledger.

## Targets, observables, and kill gates

- Posterior orientation Brier, logarithmic score, selective-accuracy and coverage at confidence≥0.95; **on DIFFUSE, any confident orientation decision is unsupported**.
- Mean resource spent, ledger closure, focus/quiet mix, posterior reset mass and learned hazard trajectory.
- Change recovery (time to gate after a jump), false Worldlet budding in STABLE, true bud rate in JUMPS, uncontrolled budding per episode.
- Hysteresis prevents mode-switch chattering; no NaN, infinite/negative variance, reset probability outside [0.005,0.30], or overspending.
- Metamorphic parity swap and odd-only identifiability check under symmetric source prior.
- Equal seed/orientation sampling, independently seeded potential observation noise for each probe choice, at least 64 paired seed trials per world and mode; paired seed bootstrap uncertainty for meaningful gaps.
- Compare wall-clock, event count, state count, and sensor-energy measures separately. No claims about real joules unless measured.

Predictions prior to run: ADAPTIVE may improve budget usage in mixed/switch worlds but will not automatically beat FIXED_FOCUS on stationary worlds; the latter may remain the economical default. DIFFUSE should induce calibrated uncertainty and abstention; if it does not, model misspecification/heuristic feedback is a failure to document. Hazard learning may destabilize nominal calibration; an ablation distinguishes this. The use of a scalar projected sensor means **no four-dimensional fiber advantage** can be inferred.

## Authority

P: determinant/transpose transport identities under assumptions. A: chosen environment, priors, probabilities and mode transitions. M: only bounded reproducible runs. H: improvement in plastic developmental intelligence/energetic efficiency. S: autonomic/spinal/enteric analogies. [R]_id for continuity; [R]_transport for IMM Gaussian moment matching. No self-promotion from an internally generated experiment.
