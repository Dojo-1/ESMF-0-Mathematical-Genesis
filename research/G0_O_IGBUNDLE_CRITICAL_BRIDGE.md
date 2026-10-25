# G0-O · Fiber-bundle reconstruction from IGBundle-LLM (critical audit)

Source repository: https://github.com/jesusvilela/IGBundle-LLM
Inspected: `README.md`, `src/igbundle/modules/geometric_adapter.py`, `src/igbundle/geometry/riemannian.py` on the default branch, 2026-10-25. This is a **bounded code inspection**, not a whole-repository mathematical proof audit.

## Surviving architectural intuition [S/H]

- Base manifold M should index **situations/conditions of observation and development**, not merely serve as a decorative latent space.
- Bundle pi:E→M should specify local fibers E_x, local trivialisations, transition maps satisfying cocycle identities, and allowed sections.
- A declared connection ∇ allows comparisons, covariant derivatives and path-dependent parallel transport, including holonomy, rather than coordinate differences.
- Fibers may be typed/heterogeneous; when ranks differ, use a stratified bundle, a sheaf of modules, a categorical fibration or a correspondence rather than falsely claiming an ordinary fixed-rank vector bundle.
- The embodied substrate (physical tissue), the experiential base (sensorimotor histories), and cognitive fibers may have **different** bases related by maps/correspondences, not one universal Poincare ball.
- A neutral (2,2) form can exist *within* suitable fibers while dissipative resource accounting requires a separate positive coercive quantity.

## Mathematical problems found in source

1. In `riemannian.py:RiemannianGeometry.get_metric`, the tensor g=LL^T+eps I is constant in base positions (for each component). In the supplied coordinates its unique torsion-free metric-compatible Levi-Civita connection has zero Christoffels and vanishing curvature. Yet `christoffel_symbols` independently predicts arbitrary position-dependent connection coefficients. These coefficients need not be Levi-Civita for the returned g. If used as a different affine connection, metric compatibility and torsion must be independently specified.
2. `RiemannianMetric.ensure_positive_definite` symmetrizes and adds eps I, which does not guarantee positive definiteness for an arbitrary symmetric matrix; positivity currently follows from the upstream LL^T construction, not that method generally.
3. `riemann_curvature` calculates one finite-difference derivative term while omitting the second derivative and quadratic Christoffel terms of the Riemann tensor. Therefore the returned tensor is not generally a curvature tensor.
4. `sectional_curvature` uses a contraction with repeated v in antisymmetric slots (if the index convention were that of true curvature, such contraction would vanish) and omits a metric contraction on the raised index. Interpretation as sectional curvature is not justified.
5. `GeometricIGBundleAdapter._parallel_transport_update` explicitly implements a fixed 0.1 Euclidean coordinate correction, not a connection-defined parallel transport.
6. `_information_geometric_update` scales by the inverse metric diagonal rather than solving full g^{-1} grad f; it acts on an update-network output not demonstrated to be an objective gradient.
7. `_riemannian_weighted_mean` uses Euclidean differences instead of exp/log maps and hence does not establish an intrinsic Fréchet mean.
8. `FiberBundleLambdaCalculus.fiber_morphism_compose` is an unconstrained MLP on concatenated vectors. Associativity, identity, source/target typing and composability are not enforced. It is not established as categorical composition.
9. The README's 'Poincare base + categorical fiber + sheaf + symplectic' statement is an architecture proposal; it should not be interpreted as a mathematically verified bundle/sheaf/symplectic construction solely from those labels.
10. A categorical probability simplex has constrained tangent structure and may acquire Fisher geometry on its interior; raw K logits and softmax outputs are not automatically a vector bundle with unconstrained categorical composition.

## Proper ESMF mathematical backbone [A / targets]

Define a smooth base M, an open cover {U_a}, a fixed-rank local fiber F when applicable, and transitions g_ab:U_a∩U_b→Aut(F) satisfying
- g_aa=id, g_ab g_ba=id, g_ab g_bc=g_ac on triple overlaps.

A connection is local one-forms A_a with transformation
A_b = g_ab^{-1} A_a g_ab + g_ab^{-1} d g_ab
for the convention e_b=e_a g_ab; confirm sign and conventions in implementation tests.

Curvature F_a=dA_a+A_a∧A_a transforms F_b=g_ab^{-1}F_a g_ab. Parallel transport obeys D_t v=0 and changes under gauge at endpoints. For neutral fibers impose metric compatibility ∇h=0 if a metric-preserving connection is claimed. Keep real physical energy separate from indefinite h(v,v).

For changing fiber dimensions use a *stratified or categorical* extension, not a globally fixed-rank vector bundle; explicitly account for typed transport remainder and gluing failures.

## Required falsifiers

- Triple-overlap cocycle and overlap reconstruction tests (including intentionally broken cocycles).
- Curvature from two paths around infinitesimal loops agrees to leading order with dA+A∧A.
- Gauge transformation of connection and curvature; holonomy conjugacy.
- Explicit separation of Levi-Civita curvature, independent gauge curvature and information-geometric/Fisher curvature.
- On flat constant g, assert Levi-Civita curvature is zero regardless of a neural Christoffel predictor.
- Compare with coordinate-only, flat-connection and zero-curvature controls on matched intervention tasks.
- No `P` promotion until mathematical objects and compatible algorithms are reconstructed; no `M` claim for cognition without controlled run.

## Scope and provenance

The external IGBundle repository remains unchanged. These are static findings in inspected files, not statements that all branches or experimental reports are invalid. The G0-O bundle itself is a specification [A/H], not implemented/validated yet. Use the stronger corrected mathematics as the starting point, preserving IGBundle's architectural intuition without claiming its code establishes differential geometry.
