# G0-AE — Telos-native simultaneous genesis

**Evidence:** S/H architecture, P for the finite-generator and frame-change identities under stated assumptions; code/tests are finite *contracts*, not M cognitive performance. **Status:** proposed joint biological–geometric–semantic interface stacked on G0-AD, not a completion claim.

## From inside the organism, not from the scheduler

ESMF-0's original question is whether a resource-bounded organism, receiving only local encounters and consequences rather than externally named latent concepts, can develop **reusable, revisable distinctions** through its own changes of perception, action, memory and anatomy.

Its operative telos is not the optimization of a universal semantic score. It is continued *viable participation in encounters* while differentiating, maintaining and correcting capacities to act. Distinction is necessary but never sufficient; infinite distinctions without viable memory, sensing and morphology are pathological. Neither survival nor coherence by itself counts as learning.

This forbids a governing pipeline: "biochemistry, then geometry, then semantics, then memory, then growth." A stage order has no privileged cognitive status. Instead all local mechanisms must have transition propensities defined **against the same current organism**, competing and coupling through resource availability, sensed outcomes, remembered distinctions, geometric misfit and anatomy.

A computer may sample those events sequentially. That serialization is a *realization* of a joint stochastic law, not the law itself. There is no physical claim that separated events literally occur at identical timestamps.

### Carriers and local capacities

The finite G0-AE surrogate has a state

    s = (e, m, c, q, theta, D, buds, last)

with:
- e: nonnegative currently available energetic resource;
- m: separately accounted structural material (a second physical resource; not a cognitively meaningful scalar score);
- c: currently accessible observational context, with several distinct Worldlet registers;
- q: a currently selected interrogative mode, drawn from G0-AD's typed Lullian question generator;
- theta[c]: one of TWO explicit candidate transports on Mat2(R)'s four-real-coordinate fiber: identity or transposition;
- D[c]: an independently stored witnessed-response and currently salient-response record;
- buds[c]: a paid temporary additional context compartment;
- last: index of an already observed local sensor pair; it is never prefilled with hidden environment truth.

The current laboratory environment offers a finite set of observable before/after responses and **full four-coordinate synthetic paired sightings**. This is substantially stronger sensory access than the partial, delayed observations in G0-AC/G0-Y; no biological sensory equivalence is claimed. The **external generator** has probabilities for which sightings occur. The organism only gets the realized observation. Its repair, learning and budding propensities do not inspect those probabilities or any hidden classification oracle.

The *complete* ESMF-0 ecology is not this fixed two-context laboratory. Rather this is a single small, executable chart on the larger proposed ecology. G0-F chemical fields, G0-AC delayed sensing and G0-Q's true nonorientable cylinder are **not yet jointly implemented** with this generator.

### A native joint dynamical law

For any bounded observable f of the organism's state, define

    (L f)(s) = sum_{a in A(s)} r_a(s) [f(F_a(s)) - f(s)].

Every allowed local channel a has nonnegative hazard r_a(s), and its target F_a(s) must satisfy the viability constraints. The list order of the channels is mathematically irrelevant. The associated finite CTMC Q-matrix has nonnegative off-diagonals and rows summing to zero. The law is

    d p_t / dt = p_t Q,

and in particular conserves total probability while remaining in the declared finite viable state space.

This is classical chemical-master-equation mathematics, not a new theorem. Its point is to choose the *correct object* before choosing an implementation.

Channels in the present surrogate:
- supply of external energy and structural material (separate, positive ledgers);
- maintenance and context changes;
- interrogation, which costs resource and changes which sensory experiment can be carried out;
- sensing, which consumes resource and updates the current context's behavioural record;
- independent slow salience dissipation, with witness counts preserved;
- transport repair, enabled only by disagreement between two locally observed, explicitly compared correspondence candidates;
- paid Worldlet budding, requiring **simultaneously** active positive and negative response witnesses, nontrivial geometric mismatch, and sufficient material and energy;
- conditional pruning, which salvages only one of the two originally paid material units without erasing witness history.

Budding requires at least **two currently salient witnesses of each opposing response type** plus the geometric and resource guards; one contradictory observation cannot itself authorize a bud. This remains vulnerable to repeated sensor noise and is not a source-continuity proof.

The channels are coupled in both directions. A bud changes the rate of future local encounters **and** increases maintenance demand; maintenance can prevent an interrogation; an observation changes the geometric-repair and budding opportunities; questions change observation probabilities. Nothing in the generator requires the organism to finish "semantic processing" before chemical activity can resume.

### The three nontrivial guards

1. **Positive resources:** the model cannot sense, interrogate or bud without paying its declared energetic or material costs. Indefinite quadratic forms are never interpreted as metabolism.
2. **No silent semantic absorption:** contradictory observed responses do not turn into a mean similarity score. Witness counts are retained separately from fast-fading salience; they remain revisable evidence, not immutable truth. The finite CTMC uses saturating counts, while the audit trace retains event history up to its configured horizon.
3. **Transport-aware but not transport-obsessed:** two explicit geometry candidates are compared using G0-L loss-aware correspondence under a positive observational metric. A smaller geometric defect proposes a repair, but does not prove behavioural equivalence. Geometry alone is insufficient to bud.

The organism never receives a label saying "the geometric symmetry is transposition". It may update a candidate only after an actual paired sensory observation. This is **constrained candidate selection**, not the discovery of an arbitrary transformation group.

## Frame and neutral-geometry contract

Let X and Y be independent local four-dimensional real fibers with a positive observational metric H_Y and a candidate T: X -> Y. The target defect after a local observation is

    d_T(x,y) = (y-Tx)^T H_Y (y-Tx).

For invertible re-framings B_X and B_Y, G0-L uses

    T' = B_Y T B_X^{-1},
    H'_Y = B_Y^{-T} H_Y B_Y^{-1},
    (x', y') = (B_X x, B_Y y).

Therefore d_T(x,y) = d_T'(x',y') exactly in algebra, up to floating-point residual in the implementation. The birth/repair hazards computed from this quantity are chart-invariant.

Separately Mat2(R) carries the neutral determinant form of signature (2,2), preserved by the candidate transpose P. The raw determinant in *arbitrary non-volume-preserving charts* is not itself invariant. Signed neutral mismatch is NOT a positive loss or metabolic resource. The numerical tests require both reframe consistency and an honest neutral diagnostic.

## Quantum-amiability, precisely delimited

Any finite classical Markov Q-matrix admits a **diagonal Lindblad lift** on a Hilbert space with basis |s> indexed by all possible organism states. For each positive off-diagonal rate q(s,t), choose

    J_{t,s} = sqrt(q(s,t)) |t><s|.

With Hamiltonian zero, the finite GKSL generator

    G(rho) = sum_{s!=t} [ J_{t,s} rho J_{t,s}^*
                   - (1/2) { J_{t,s}^* J_{t,s}, rho } ]

preserves trace and positivity. On diagonal states

    rho = sum_s p_s |s><s|,

its diagonal agrees exactly with the classical master equation. This is a **standard construction**: a proof follows by taking the diagonal matrix elements of the GKSL equation. A diagonal density matrix is not a coherent superposition, nor does this establish quantum computational advantage.

The desired stronger *quantum* extension would require genuine coherent operations, physically admissible tensor-product locality, instruments for environmental observations, nontrivial off-diagonal evolution, and causal/no-signalling tests. A classical joint process does not supply those.

### Why summation rather than mandatory commutativity?

The implementation includes a conditional generator-bracket diagnostic. At the zero-resource boundary, the observable number of witnessed different responses has

    [L_supply, L_encounter] f = 0.70 × 0.55 × 0.85 = 0.32725

for the specified illustrative query and sensor distribution. Feeding first can enable an encounter; an encounter cannot happen first at the depleted boundary. This nonzero bracket is **classical causal order sensitivity**, not quantum coherence.

There is no requirement that all channel generators commute. Resource-consuming interrogation and resource-consuming growth may not commute, particularly at a viability boundary. Declaring them commutative would erase real causal competition. A globally *specified* joint generator remains meaningful even when conditional local operations do not commute. The proper criterion is causal consistency, not order-independence of every hypothetical intervention.

We should not confuse simultaneous **law** with parallel instruction execution, and should not claim indefinite causal order.

## Exact, inexpensive mathematical targets

Under the finite-environment, bounded-state and admissibility assumptions:

- **P1, standard:** Q has nonnegative off-diagonal rates, and every row sums to zero; no allowed transition leaves the declared viability envelope.
- **P2, standard:** the positive target observation defect and geometric-repair hazard are invariant under the specified independent frame changes.
- **P3, standard:** witness counts are monotone (until saturation) while current salience may decay; no source observation is silently rewritten by dissipation.
- **P4, standard:** the diagonal GKSL lift above reproduces the CTMC on diagonal states.

These are structural properties, not theorems of consciousness, intrinsic intelligence or endogenous semantics. The code's test suite provides bounded numerical/contract checks, not a complete mechanized proof of all finite states.

### The minimal counterworld

If an organism sees alternating opposite outcomes only because of independent response noise, its contradictory witness record can trigger an unnecessary bud. A geometry mismatch may also be caused by sensor corruption, not a missing context. Hence the current conjunction of guards is **necessary by design but not sufficient to infer real latent structure**.

A more elaborate dynamic topology may overfit; a conventional recurrent Bayesian model might handle the same problem better. Neither more compartments nor geometry should be promoted for their own sake.

The telos should permit the organism to **refuse** structural growth, or to prune a mistaken growth episode. But in this surrogate "bud" is still a single paid boolean compartment, not biological neurite extension.

## Interfaces to existing G0 research

- G0-F provides *real* chemical field and structural material accounting ideas; G0-AE currently only uses bounded integer resource surrogates.
- G0-L supplies heterogeneous transport and gauge-aware positive comparison.
- G0-M/U/W supply the neutral Mat2(R) carrier and the admissible transpose candidate, not a quantum state space.
- G0-AB/AC provide richer autonomic and multi-clock filters, which may become separate coupled hazards in a successor.
- G0-AD supplies typed inquiry and the separation of witnessed response difference from geometric equivalence.
- G0-Z/AA supply source-continuity reasoning, still needed before the model can interpret contradictory observations as genuine semantic splits.

No earlier result file, frozen calibration, control ranking or reproduction script is rewritten.

## Tests, controls, next empirical gate

The branch adds contract tests checking: state viability; rate positivity and row closure; absence of a canonical event order; question-dependent observable menus; resource-dependent sensory eligibility; two-way anatomy/sensing cost; witness/salience separation; independent GL(4) reframe covariance; an observed-pair-based geometric repair; paid bud/prune; and positivity/explicit tail of finite-state uniformization.

    python -m esmf.telos_joint_ae
    python -m unittest discover -s tests -p 'test_telos_joint_ae.py' -v

All actual experimental claims are **H until separately measured**. Proposed G0-AE1 must compare this *joint* generator against a carefully budget-matched sequential-splitting approximation to the **same** dynamics, and against a conventional real-valued partially observable state estimator that has access to the same observational history. Merely beating a carelessly staged simulation is not a cognitive discovery.

Use stochastic-seed pairing, analytic Q-state checks, operator-order permutations, step-size convergence, compute/memory/resource accounting, false-bud rates, useful distinctions and abstention. Predefine a kill rule: no promotion if the matched conventional learner equals or wins in predictive utility/viability, if extra morphological events mostly reflect noise, or if gains disappear as the sequential approximation converges.

## Meaning of success

Not a magical speedup, consciousness, nor "quantum" because the notation resembles density operators. A success worth preserving would be a demonstrably viable dynamic ecology in which **questioning, sensing, geometry, memory and physical growth reciprocally change each other's opportunities**, with no privileged stage ordering—and in which the resulting distinctions improve downstream behavior under strong controls.

Our available result so far is an explicit *hypothesis-bearing* joint mathematical structure, with limited executable contracts. An integrated biological field solver, truly learned local context boundaries, online nontrivial bundles and quantum coherent implementation remain open.
