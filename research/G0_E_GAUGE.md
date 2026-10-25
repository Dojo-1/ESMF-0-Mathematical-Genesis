# G0-E · Basis freedom, conjugation, and developmental geometry

**Principle:** A change of local coordinates must not manufacture a new memory, a new energy, or a novel semantic distinction. This is the model's minimal *gauge hygiene*.

For fiber state u_i, positive definite local metric H_i, and transport U_ij:

E_ij = (u_i - U_ij u_j)^T H_i (u_i - U_ij u_j).

For any invertible local charts B_i, B_j, transform:

u_i' = B_i u_i;
U_ij' = B_i U_ij B_j^-1;
H_i' = B_i^-T H_i B_i^-1.

Then E_ij'=E_ij, a coordinate identity. Conjugation appears for a loop beginning and ending at i: Hol_i' = B_i Hol_i B_i^-1, hence holonomy conjugacy class is gauge-invariant. Independent chart choices cannot be compared by naive coordinate subtraction.

For observation operator O, G=O^T O, changing domain chart by B gives
G'=B^-T G B^-1. Rank and definiteness survive; det(G')=det(G)/(det B)^2. So ordinary Gram determinant is **not** invariant under arbitrary GL charts. For orthogonal or unimodular charts it is preserved; use metric-normalized volumes when general gauge invariance is essential.

The validated code in esmf/gauge.py checks these matrix identities numerically. These checks do **not** establish the full connection-Laplacian Lean theory, neuron-level biological validity, semantic learning, or effective hypercomplex memory.

## Geometry freedoms to explore later

1. **Conjugation**: local loop transports, spectral conjugacy invariants.
2. **Periodic wraparound**: toroidal spatial boundary currently used by the developmental surrogate; it is a numerical boundary condition, not a claim about cortical topology.
3. **Adaptive bases**: local invertible charts with transformed metrics, not coordinate-dependent "creative" signals.
4. **Higher fibers**: 2x2 real blocks first; unitary/complex and quaternionic candidates only with explicit adjoints and matched baselines.
5. **Branching in physical space vs conceptual space**: different operators and clocks; do not confuse short Euclidean distance with semantic nearness.

## Missing bridge

G0-D morphology is scalar activity plus geometric tips. G0-E transport is an external structural instrument. Their coupling is not yet implemented, and growth/coordination advantages have not been measured. Next experiment must quantify effects of gauge-invariant vs raw-coordinate decisions under randomized local chart rotations.
