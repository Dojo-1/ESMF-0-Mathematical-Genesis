# G0-D · Developmental growth surrogate

**Question:** Can a locally generated trophic field plus stochastic exploratory tips, activity-dependent stabilization, and pruning produce bounded and differentiated connectivity *without using node identities or environment labels to choose edges*?

## Biological analogies (not biological reproduction)

- Growing exploratory tips and branch guidance: axon growth, guidance and branching.
- Diffusive, locally generated concentration field: coarse chemoattractant surrogate.
- Coincident activity and survival: oversimplified activity-dependent stabilization.
- Weak-contact removal: synaptic pruning surrogate.
- Rising maturity and decreasing tip mobility: developmental plasticity-to-stability transition.
- Local finite energy: metabolic/resource constraint.

This model does **not** simulate real axons, neurotrophin receptor signaling, glia, electrochemistry, cortical lamination, gene regulation, realistic developmental times, or physical continuous trajectories. Edge pruning in particular is deliberately abstract.

## Rule boundary

No external teacher, global graph optimizer, latent category label, Gram determinant or connection-Laplacian oracle is used inside the developmental loop. The 2D chemical field is spatially local. A candidate edge forms when a tip enters capture radius of a soma. Contact survival depends on local activity and a finite age/resource budget.

**Important artifact limit:** current simulation searches all soma coordinates when finding a nearby target; although its *decision* uses geometric locality, the implementation's neighbor search is O(N²) across tips and somata rather than a spatial index. Next version should use a uniform grid/spatial hash.

## Falsifiers and controls

- Verify determinism and all finite bounds for n=24, k=4 over 0..300 updates.
- Show edge births, pruning and stable contacts actually occur, rather than merely describing developmental morphology.
- Compare against spatial random-edge growth with identical capture opportunities, budget and stochastic draws.
- Ablate trophic gradient, activity-dependent survival, maturation, and pruning separately.
- Vary density, noise and resource refill. Require finite budget and report collapsed/overgrown regimes, not only successful runs.
- Test whether any growth property improves downstream held-out compositional prediction; this is **not** yet measured.

## External sources for biological analogy

- Hilton et al. (2024), Neuronal maturation and axon regeneration, doi:10.1038/s41583-024-00849-3
- Demmings et al. (2025), (Re)building the nervous system, doi:10.1111/jnc.16258
- NCBI Bookshelf, Molecular and Cellular Mechanisms of Axon Morphogenesis, https://www.ncbi.nlm.nih.gov/books/NBK612545/

## Evidence

Status S/H until runs. Runtime invariants are checked, not Lean proved. Reproducibility uses standard library and a deterministic seed. No claims of artificial cortex, subjective experience, or human-like development.
