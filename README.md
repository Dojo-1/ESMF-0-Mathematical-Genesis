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

**Historical G0-A/G0-B were weak comparators.** A scripted-edge model appeared better than an empty fixed graph. The 2026-10-25 G0-B run added a fixed one-edge control and it slightly **outperformed scripted growth on all eight paired seeds**. This result does **not** establish endogenous learning and did not match contact exposure. See [the historical result and current implementation status](research/results/G0_B_EDGE_PARITY_2026-10-25.md).

The current [G0-B runner](esmf/controls.py) pairs a fixed 0↔1 edge, seeded random one-edge, scripted-edge, and residual-driven topology. All conditions have one active contact per update and matched update, exploration, logical-operation, and parameter-state budgets. The readout is local and receives only explicit messages. Held-out tests span AND/XOR, observation and `do(A)`/`do(B)` interventions, and appearance recoding. Signed-Laplacian and observation-Gram diagnostics run externally after learning. **The revised protocol has not been measured; it is not evidence of semantic emergence.** See the [frozen protocol and limits](research/G0_PROTOCOL.md).

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
python -m esmf.mechanochemical --steps 300 --seeds 0,1,2,3 --suite
python -m esmf.mechanochemical --steps 300 --seeds 3 --continuum
```

The follow-up runner reports paired chemistry, reversed-chemotaxis, nutrient-ablation
and random-navigation controls. `--continuum` compares timestep halving, a CFL-stable
2x grid refinement, and no-flux versus periodic boundaries at fixed physical time.
It reports separate nutrient/ATP and material ledgers, actual paid branch length,
edge lifespans, maturity, and validity residuals. These are surrogate bookkeeping
checks; they do not establish thermodynamic consistency or biological viability.
See the [G0-F follow-up audit](research/results/G0_F_2026-10-25.md#follow-up-audit-2026-10-27).

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

## G0-Q · Globally twisted neutral bundle

[Executable construction](esmf/global_bundle_q.py) advances beyond local gauge checks and the earlier trivial bundle: a non-orientable rank-four neutral (2,2) real vector bundle over a cylinder, with three position-dependent charts, triple-overlap transitions, a compatible nonflat connection, explicit global sections, curvature and winding holonomy. [Remote test and failures ledger](research/results/G0_Q_2026-10-25.md) records both the corrected successful run and the initial metric-derivative bug. **This is an explicit, nontrivial bundle instance, not yet a general theory of cognitive fiber bundles or a learning organism.**

```bash
python -m esmf.global_bundle_q
```

## G0-R · Honest variable-rank interface

[Rank-stratified restrictions](esmf/strata_r.py) provide separate local spaces and typed boundary comparison; [smoke test](https://huggingface.co/jobs/jesusvilela/6adde50d046eb82b1ece4e5d) distinguishes compatible and incompatible gluing and identifies invisible kernel states. This is a presheaf-like test instrument, **not** a globally fixed-rank vector bundle or a proved sheaf descent system. See the [G0-Q expanded audit](research/results/G0_Q_2026-10-25.md).

```bash
python -m esmf.strata_r
```

## G0-S/T/U · First causal learning across bundle charts

**New measured bridge, not cognition proved:** an independently simulated, reversible, area-preserving two-action world on S¹×R supplies short noisy sensorimotor traces. [G0-S](esmf/causal_bundle_s.py) learns two causal-action coefficients and predicts longer held-out intervention sequences. [G0-T](esmf/context_bundle_t.py) infers two nonlinear observation-chart parameters from 32 paired sensor sightings and transfers predictions among three changing local charts. [G0-U](esmf/associated22_u.py) explicitly lifts physical Jacobians to Mat₂(R), a real four-dimensional **(2,2) neutral associated fiber**, with an appropriately transformed bilinear metric.

On 32 paired synthetic seeds, mean held-out loss in the noncommuting world was 0.136080 (flat translation), 0.001496 (linear shear), and 0.00000507 (correctly specified nonlinear shear). **The nonlinear model matches the generative family by design.** The ranking reverses in the commuting-world negative control: flat 0.00000433 versus nonlinear 0.374192. Learned chart calibration retained noncommuting loss 0.00000516; intentionally chart-blind processing scored 0.05614. All numerical gauge tests and independent tangent finite-difference checks passed.

**Critical comparator:** the same calibrated nonlinear recurrence implemented with ordinary real arithmetic and no explicit bundle lift obtains the same predictions. Therefore we have established *consistent causal learning with geometric transport*, **not an advantage of bundle geometry over an equally informed conventional model**, and no measured efficiency or biological cognition. The G0-Q nonorientable global twist remains a separate bundle instance, not yet coupled to this learner.

[Complete G0-S/T/U results, failures and bounds](research/results/G0_S_T_U_2026-10-25.md) · [Pre-registered causal protocol](research/G0_S_CAUSAL_PROTOCOL.md)

```bash
python -m esmf.causal_bundle_s --seeds 0,1,2,3,4,5,6,7
python -m esmf.context_bundle_t --seeds 0,1,2,3,4,5,6,7
python -m esmf.associated22_u
```

## G0-W · A topology obstruction and two commuting regulatory factors

An important constraint emerged from the [measured causal associated-bundle experiment](research/results/G0_S_T_U_2026-10-25.md): on Mat₂(R), every LEFT GL₂ action has positive determinant in its induced four-dimensional representation, so this action family alone cannot reproduce the nonorientable global twist of G0-Q. [G0-W](esmf/orientation_seam_w.py) implements an orientation-reversing **transpose** isometry that preserves the (2,2) determinant quadratic form while conjugating left SL₂ actions into right ones. Two factor actions commute across left/right roles, but can be noncommutative within a role. Seam-equivariant transport passes remote tests; a chart-blind operator fails.

This suggests—but does not establish the usefulness of—a dual-organ, multi-clock algebraic surrogate with factor-exchange transitions. [G0-W evidence and limitations](research/results/G0_W_2026-10-25.md). No seam-adaptive learner or cognitive/energy advantage has been measured.

```bash
python -m esmf.orientation_seam_w
```

## G0-X · Learning one global bundle transition from evidence

[G0-X](esmf/monodromy_x.py) advances beyond checking preselected local/global structure: given ten noisy odd-winding fiber observations and previously learned causal dynamics, it chooses between identity and orientation-reversing transpose monodromy. Across 32 seeds per world it chose the correct candidate **32/32 times in each**; it correctly abstained when no odd winding was observed, and double winding restored parity. A third, out-of-family neutral twist was rejected in 32 exploratory cases using a posthoc threshold. [Protocol](research/G0_X_MONODROMY_PROTOCOL.md) · [Full results and critical boundaries](research/results/G0_X_2026-10-25.md).

**Evidence boundary:** this is constrained binary model selection on a simulation built from those candidate laws, using faithful frame sensors, not learned arbitrary topology, semiotic understanding, or better efficiency than an equivalently informed real recurrent model.

```bash
python -m esmf.monodromy_x --seeds 0,1,2,3,4,5,6,7
```

## G0-AC · Calibrated reflex versus auditor

[G0-AC](esmf/reflex_auditor_ac.py) tests a budgeted, hysteretic credible-consequence gate between compiled partial-sensor circuits and a costly learned return-map auditor. The hidden state is an arbitrary full Mat₂(R) matrix with noncommuting left/right SL₂ transport families, three overlapping rank-two local sensors, and causal forward/inverse/rest interventions. Reversible transport, reset, gradual drift, dissipation and independent replacement are separate tasks. No learner sees true states, injected test sections, chart adapters or reset labels.

The bounded real recurrent ensemble supports fast continuity, slower hazard updates and temporary model-slot budding/pruning with provenance and reserved undo costs. FIXED_FOCUS, FIXED_QUIET, random, equal-capacity real recurrent/IMM, no-slow, no-budding, blinded-sensor-ID and exact equivalent-real controls share sensors and budget caps. Temperature calibration uses only development seeds 512–527; the [pre-implementation registration](research/G0_AC_PREREGISTRATION.json) reserves fresh seeds 4096–4127.

Every post-burn-in tick is scored, including abstentions and energy exhaustion. Reports separate nonnegative simulated metabolic cost, RLS arithmetic proxies, wall/CPU time and **unmeasured hardware joules (`null`)**. Common energy caps are not equal actual spend: frontier comparisons require no greater comparator spend/RLS work, and promotion additionally requires no greater measured comparator CPU cost. Return maps are lossy diagnostics, not topological gluing certificates; “false confident gluing” measures local predictive-continuity declarations across real resets. The gate's value is a conservative risk proxy, not an identified causal value of information.

From the repository root (artifacts refuse overwriting; freeze calibration before evaluation):

```bash
python -m unittest discover -s tests -p 'test_reflex_auditor_ac.py'
python -m research.run_g0_ac_heldout calibrate --output /tmp/g0-ac-calibration.json
python -m research.run_g0_ac_heldout smoke --calibration /tmp/g0-ac-calibration.json --output /tmp/g0-ac-smoke.json
python -m research.run_g0_ac_heldout evaluate --calibration /tmp/g0-ac-calibration.json --output /tmp/g0-ac-heldout.json
```

This is computational morphology, not physical growth, a reconstructed adaptive neutral connection, or coupling to G0-Q's nonorientable cylinder. **No sentience, biology, unique geometric advantage or physical energy-efficiency claim.** The extra controller remains exception-triggered unless the registered predictive/calibration/compute comparisons pass.

### Frozen held-out outcome

[Frozen calibration](research/results/G0_AC_CALIBRATION.json) was committed before [evaluation](research/results/G0_AC_HELDOUT.json): 32 fresh seeds × 5 worlds × 10 policies × 3 caps = **4,800 episodes**, with exact equivalent-real agreement. At cap 12 both controllers actually spent 12 simulated units; all-tick Brier (lower is better) was:

| World | Adaptive | Fixed focus |
|---|---:|---:|
| Reversible | 0.140868 | 0.120074 |
| Reset | 0.159021 | 0.157063 |
| Drift | 0.141009 | 0.121869 |
| Dissipation | 0.169159 | 0.154138 |
| Independent | 0.277950 | 0.280884 |

**Kill decision: no promotion; retain exception-triggered control only.** At caps 20 and 32 adaptive spent 16.68 versus fixed focus's 16.32 without a general predictive gain. Adaptive triggered **zero audits and zero buds across every held-out budget**, so this run supplies no empirical validation of structural growth/undo: bud precision is undefined and reset-task recall is zero. Unit tests exercise evidence-driven undo; random sensing exercises the actual auditor. Do not reinterpret a dormant gate as successful adaptation. Calibration/thresholds were not retuned after evaluation.

The JSON includes every policy/cap, selective coverage/accuracy, Brier/logloss/MSE, local false-gluing counts, bud precision/recall, paired seed intervals, costs and limitations. The complete held-out evaluation, including comparisons/bootstrap, took **36.41 s wall / 36.40 s CPU** on the recorded environment. A development-only calibration replay reproduced the frozen artifact exactly and took **48.40 s wall / 48.39 s user+system CPU** (`/usr/bin/time -p`): approximately **84.81 s wall / 84.79 s CPU** for the two measured stages, excluding development attempts. These are computer costs, not joules or an efficiency result. The full 118-test suite and CodeQL passed (zero alerts). G0-Q/AC0 connection coupling and physical morphological growth remain open; this result does not justify closing those research requirements.

## G0-AC0 · A genuinely adaptive neutral connection (new)

[G0-AC0](esmf/adaptive_connection_ac0.py) lifts adaptation from G0-AB's scalar observation quotient into a **rank-four real neutral (2,2) vector bundle connection** over a cylindrical base. The connection `A_t=a_tX dtheta + b_tY dz` now learns its two coefficients from *two partial sensor readings* of injected test sections; it remains compatible with three nonorthogonal position-dependent local frames, the neutral metric, nonzero curvature and the orientation-reversing global seam. [Mathematical and numerical audit](research/results/G0_AC0_2026-10-25.md), [frozen pre-run protocol](research/G0_AC0_ADAPTIVE_CONNECTION_PROTOCOL.md), [replay driver](research/run_g0_ac0_heldout.py).

**Held-out 32-seed result:** continued learning beats a frozen connection when the transport law changes, but **ordinary constant-forgetting recursive least squares outperforms the surprise-triggered adaptive variant** in abrupt-change and sign-reversal worlds. Under stationary transport, no forgetting wins. A mathematically equivalent conventional real estimator ties the adaptive variant. **Connection adaptation is implemented and chart-covariant; an intrinsic geometric learning/energy advantage is not demonstrated.** Eight independent package tests passed on remote CPU.

```bash
python -m unittest discover -s tests -p 'test_adaptive_connection_ac0.py'
python -m research.run_g0_ac0_heldout --start 100 --count 32
```

## G0-AB · Adaptation with a resource and evidence budget

[G0-AB](esmf/adaptive_ecology_ab.py) now implements a multi-clock adaptive surrogate: fast sensory selection and Gaussian filtering, mesoscopic surprise-/provenance-dependent sensing precision, slow reset-hazard feedback, and evidence-gated creation of temporary temporal Worldlet contexts. The organism's positive resource ledger includes external supply, explicit sensing expense, overflow, viability floor and state-change costs. The underlying mathematical carrier is a **one-dimensional shear orbit inside a genuine neutral (2,2) Mat₂(R) representation**, not full learned four-dimensional cognition.

**A critical negative result:** in a frozen held-out comparison (seed IDs 128–191, 128 episodes/world/policy) simple fixed focused sensing won predictive coverage over the adaptive version in stable, drift, two-reset and regime-switch worlds. In STABLE, adaptive confidence coverage was 64.8% vs 84.4% for fixed focus, using mean simulated resources 8.776 vs 8.990; in JUMPS it was 56.3% vs 77.3%, using *slightly more* resource. The adaptive organism generated about 0.383 temporal context buds per two-reset episode, detecting just **19.1%** of known reset events within one tick. A separate no-persistent-source hypothesis reduced false orientation claims but did **not eliminate them** (3 of 128 diffuse episodes). An equivalent real Bayesian implementation ties exactly. **No emergent cognition, geometric advantage or measured joule saving.**

[Held-out results, paired CIs, failures and correction history](research/results/G0_AB_2026-10-25.md) · [Pre-registered protocol](research/G0_AB_ADAPTIVE_PROTOCOL.md) · [Post-pilot corrections frozen before held-out](research/G0_AB1_HELDOUT_AMENDMENT.md) · [Typed architecture manifest](research/G0_AB_ARCHITECTURE_MANIFEST.json) · [Replay driver](research/run_g0_ab_heldout.py).

```bash
python -m unittest discover -s tests -p 'test_adaptive_ecology_ab.py'
python -m research.run_g0_ab_heldout --start 128 --count 64
```

An experimental follow-up now couples a finite-state reflex and novelty-triggered
continuity audit to the full charted rank-four bundle with delayed partial
sensing: [G0-AB reflex/bundle bridge](research/G0_AB_REFLEX_BUNDLE_BRIDGE.md). It is
not held-out evidence and leaves the original G0-AB results unchanged. The
bridge reports software runtime/memory only; it measures no physical joules.

## G0-Y/Z/AA · Complementary questions and evidence-gated continuity

**Most consequential correction:** the old claim “even-winding observation gives no orientation information” only holds with a known/conditioned initial fiber. For an *unknown, shared* antisymmetric fiber contrast `D=F01-F10`, one scalar even observation `Y_e=D+noise` and one odd observation `Y_o=T D+noise` are each individually uninformative about balanced monodromy `T`, but their **joint covariance determines a likelihood for T**. No shared-D provenance, no such inference. [Mathematical corollary and 256-episode resource experiment](research/results/G0_Y_2026-10-25.md).

- [G0-Y partial-observation embodied benchmark](esmf/embodied_y.py) adds delayed noisy sensors, action-conditioned cylinder dynamics, damage/recovery, positive resource accounting, four-parameter predictor controls, chart/seam ablations, and per-seed reproducibility. The protocol and raw 32-seed results are [here](research/G0_Y_EMBODIED_PROTOCOL.md) and [here](research/results/G0_Y_EMBODIED_2026-10-27.md). Its geometry row is an exact chart-aware relabeling control, **not an integrated 4D fiber learner**; resource/event counts are not watts.
- [G0-Y active epistemics](esmf/active_epistemics_y.py): 256 episodes, a two-step information policy reached 77.3% classification coverage using 2.99 simulated resource units on average. A *fixed alternating-focused circuit* did **slightly better** at 79.7% coverage, with 3.14 units; the online policy cost ~**500×** more Python CPU time in this benchmark. An ordinary identically informed real Bayesian model ties the geometric interpretation.
- [G0-Z provenance auditor](esmf/provenance_z.py): a three-hypothesis filter distinguishes one persistent D from independently replaced D; in 128 independent-replacement episodes, unsupported confident orientation claims fell from **71.9% to 10.9%**, with a small loss of confidence coverage on genuinely persistent cases. [Results](research/results/G0_Z_2026-10-25.md).
- [G0-AA temporal repair](esmf/temporal_repair_aa.py): a finite 13-hypothesis filter additionally considers one unknown change boundary and can issue two **temporal Worldlet descriptors** when evidence justifies a split. It recovered the true K=3 boundary in only ~34% of all change-point episodes and sometimes proposed false splits; ~4.3× extra CPU over the simpler provenance filter. [Results and failures](research/results/G0_AA_2026-10-25.md).

**Scope:** These are scalar readouts *derived from* the Mat₂(R) neutral (2,2) bundle architecture, **not** its integrated four-dimensional developmental/embodied dynamics. Synthetic action reserves are not measured electrical energy; Bayesian source inference is not biological sentience or general semantic cognition. Strongest conventional models with the same likelihoods remain equal in these experiments.

```bash
python -m esmf.active_epistemics_y --seeds 32
python -m esmf.provenance_z --seeds 128
python -m esmf.temporal_repair_aa --seeds 128
python -m unittest discover -s tests
```

## G0 test ladder

1. **G0-A:** prove/check state bounds, degree/resource limits, deterministic replays, and numerical stability.
2. **G0-B:** compare fixed and dynamic topology on held-out interventions, with appearance nuisances to prevent superficial classification.
3. **G0-C:** test withheld causal compositions and damage recovery; require measured gains beyond equal-budget controls.

The first simulator is a deliberately small **engineering baseline**, not proof of concept emergence. Never infer meaning from graph shape, oscillation, or an attractive plot.

## Reproduce (CPU-only, no packages)

```bash
python -m unittest discover -s tests -v
python -m research.check_reproduction   # re-run stored G0-AC / G0-Y artifacts and diff
python -m esmf.experiment --seeds 0,1,2,3 --episodes 600
python -m esmf.controls --seeds 0,1,2,3,4,5,6,7 --episodes 600
```

The controls runner can preserve its full paired-seed JSON output with `--output /path/to/g0b-run.json`. Defaults use standard Python, CPU only, no network, no background services, and no filesystem write. Remote replications, if separately authorized, must begin with bounded `cpu-basic`; the local Kaggle machine remains untouched.

See [G0 protocol](research/G0_PROTOCOL.md) for fixed predictions, controls, falsifiers and what must be recorded before any claim promotion.

## Architecture constraint

No global spectral oracle or determinant maximizer inside the learner. The learner accesses only local observations, messages, interventions and bounded plasticity signals. Connection Laplacian and observation Gram are **external diagnostics**. Z+ is treated as a prospective two-channel possibilistic/probabilistic construction, **not** an implemented Zadeh-faithful uncertainty calculus in the first baseline.

## Related mathematics

- [Connection Laplacian formalization](https://github.com/jesusvilela/connection_laplacian_lean)
- [Tenth discriminant trace-Gram formalization](https://github.com/Dojo-1/tenth-discriminant)

## Safety and research discipline

No autonomous local execution, GPU access, paid inference or Kaggle process interference. Do not silently upgrade an experiment to a proof. Negative and inconclusive results belong in the record.
