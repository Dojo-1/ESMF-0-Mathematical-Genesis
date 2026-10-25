# G0-S pre-registered experiment — causal transport on a neutral associated bundle
Date: 2026-10-25. **Protocol written before experiment execution.**

## Mathematical carrier [A/P conditional]

External body-environment state lives on the cylinder M=S^1×R, local variables angle q and p. Two reversible interventions:
- X(delta): (q,p) -> (q+a delta tanh(p) mod 2pi,p).
- Y(delta): (q,p) -> (q,p+b delta sin(q)).

Each is area-preserving: its Jacobian J belongs to SL(2,R), and the effects generally do not commute. Each episode begins with an initial 2x2 tangent frame F. Its transport F' = J F induces an SL(2,R) action on Mat(2,R), a four-real-dimensional associated fiber carrying determinant quadratic form Q(F)=det(F) of signature (2,2). This makes neutral signature an *outcome of the causal transport representation*, not merely a selected metric. Physical resources still require separate nonnegative accounting.

**Caution:** the Jacobian cocycle of dynamics is path-dependent, but its noncommutativity alone is NOT proof of nonzero Ehresmann or principal-connection curvature. We measure the Lie bracket of action vector fields separately. This associated bundle is topologically trivial; G0-Q already tests a separate globally nonorientable bundle. They are not quietly claimed to be one joint model.

## Hypothesis and preregistered controls

Train from noisy initial-state/action/final-state episodes of paths length 1-3. Test on disjoint length 4-8 paths and on closed intervention commutators. Common paired seeds and identical budgets. Equal **two free scalar coefficients** for all models:
- FLAT: commuting translations, X changes q by a delta; Y changes p by b delta.
- LINEAR: reversible state-dependent shear X changes q by a delta p; Y changes p by b delta sin(q).
- NONLINEAR: exact reversible shear class X changes q by a delta tanh(p); Y changes p by b delta sin(q).

All are ordinary sequential computational models, regardless of geometric interpretation. The exact nonlinear class is the data-generating class by design, so its expected advantage demonstrates *inductive bias on a favorable toy problem*, not an intrinsic advantage of the bundle or a general intelligence result. In particular, the same state recursion with **no explicit frame/Jacobian** is a powerful matched 2-parameter control expected to tie nonlinear predictions exactly.

Negative environment: commuting translations generated under the same fitting/evaluation protocol. Expected outcome is that FLAT wins, preventing any claim of universal superiority.

## Independent tests

- Action-loop displacement: the signed closed-loop residual should scale as O(h²), with the leading Lie bracket of vector fields. Halving h should reduce magnitude by approximately 4.
- Jacobian determinant: ~1 for area-preserving shear actions, but not necessarily for a dissipative control environment.
- Local chart: coordinate map phi(q,p)=(q+k tanh(p),sinh(p)) on a lift, with B=Dphi. Verify F'_chart=B(z') J B(z)^-1 F_chart and that the determinant bilinear form transforms by its induced GL(4) metric.
- A naively untransported tensor should generally fail the metamorphic frame test.
- Compare task losses on random long paths and loop subsets, per seed, plus noise and inference time. Run-time measurements are not energy measurements.

## Kill gates

- Any nonfinite prediction or state, wrong determinant signature, negative instability in loss, broken chart invariance, or failed leading-loop scaling prevents geometric/learning claims.
- If nonlinear does not beat flat on **noncommuting** held-out paths, no positive mechanistic statement.
- If nonlinear beats ordinary identical real recurrence (it should not), treat as likely implementation/comparison bug.
- If nonlinear wins only in its matching generative family, do not claim intrinsic O(2,2) advantage.
- No semantics/consciousness claims from this experiment; causal state prediction is a precursor, not conceptual learning.

## Evidence boundary
P: determinant/SL(2,R) identities under definitions. A: selected toy physical laws and event-observation model. M: only future measured runs. H: future gain in generalization, compute efficiency, morphogenesis, semantic concepts.
