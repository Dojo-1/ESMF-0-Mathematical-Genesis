# G0-AG development-only causal controls — registered before execution

Status: **D0 developmental instrument**, not the held-out G0-AG1 study; no new ontology, no biological or semantic-advantage claim. This file is committed before the probe implementation. Evidence `H` before execution; conditional code observations may later be `M` for the exact synthetic system only.

## Object and counterfactual

The G0-AG joint generator has (i) an ordered quaternion path-discrepancy factor multiplying physical neurite-extension hazards and (ii) glial calcium/tripartite states that multiply some encounter, transfer and contact-release hazards. Measure whether those mechanisms change the **distribution of synthetic developmental trajectories**, rather than infer that their presence implies usefulness.

We will sample from **the same initial, fully resource-accounted, reachable challenge state**, constructed by a fixed sequence of *legitimate* positive-rate transitions: growth, neuron-neuron contact, glial calcium, tripartite binding, metabolic conversion, sensory encounter, subsequent conversion, phase revision, nutrient resupply, conversion, material resupply. This is a deliberately **conditioned intervention**, not an unbiased natural organism trajectory. No transition outside the allowed generator may manufacture material or sensory evidence.

Each arm retains exactly the same state schema, tissue geometry, available observations and candidate event destinations. There is no extra data or hidden semantic truth supplied to one arm.

## Interventions (factorial)

1. **full:** original G0-AG event rates.
2. **no_geometry_feedback:** keep every event and all quaternion states, but undo *only* the division of physical neurite-extension hazard by `1+0.25 D_path`.
3. **no_glial_rate_feedback:** retain the glial compartment, substrate inflow, storage and resource-transfer channels, but undo *only* calcium/tripartite multipliers of existing rates (contact release, neural sensory encounter, fiber-experience update, calcium-dependent transfer and maintenance surcharge). Tripartite state still costs its original material-proxy and glial ledger. This is **not** a no-glia ablation.
4. **both_disabled:** both rate interventions.
5. **placebo:** no changes; must be exactly identical to full for each common random seed.

Ablations do not magically equalize all **realized** resource consumption: altered rates produce altered paths, including different resource-inflow events. Compare realized resource inputs, ATP-like proxy waste and event counts explicitly and declare any parity limitations. These are information- and parameter-matched *rate* controls, not all the controls in the G0-AG1 protocol.

## Fixed development run and metrics

- Development seeds **8100–8123 inclusive**, 24 per arm; no held-out seed (18100–18227) may be inspected.
- Shared initial challenge state for all arms and seeds.
- One common `random.Random(seed)` stream per arm, used for waiting time and event selection. Shared seeds provide a weak stochastic coupling **not identical counterfactual event histories** once rates diverge.
- Simulated horizon **0.75 arbitrary time units**; event cap 250, with an explicit failure if exhausted.
- Record each sample's *actual* external resource input counts, ATP/material stores and waste, event-class counts, number of physical segments, signed bookkeeping residuals, neurite contact-time, tripartite contact-time, integral of geometric endpoint discrepancy over time, state viability, and effective evidential observations.
- Report per-arm means **and complete per-seed raw outcomes**, plus paired full-minus-control differences with descriptive normal-approximation 95% intervals (not a confirmatory p-value).
- Measure actual CPU runtime with `time.process_time`, but do not interpret small CI timing differences as hardware efficiency.
- No semantic accuracy, predictive return, generalization, valid growth utility, or biological joules are measurable from this instrument. Do **not** substitute growth counts for competence.

## Necessary negative checks

- Placebo is exact equality for each trajectory and summary (exclude CPU time).
- Structural event destinations stay identical when only hazards are counterfactually altered.
- The initial rate ratio is algebraically correct for explicitly active geometry and glial feedback; this is a **constructed known-answer**, not evidence of emergent ability.
- Resource residuals remain zero in every sampled state, independent of control.
- An independent real 4×4 left-action representation of Hamilton multiplication reproduces quaternion products (already tested in G0-AG).
- Unknown arm names and invalid seeds/intervals must fail; no silent truncation.
- All results preserve old CI and historical reproduction artifacts.

## Interpretation and kill gate

If the probe shows no discernible trajectory effect, the mechanism is dynamically marginal at this horizon. If it shows an effect, that demonstrates only **a causal effect on the synthetic generator's trajectories**, not an improvement of ESMF-0's telos. Either outcome forces the same next obligation: a separately preregistered held-out sensory/behavioral task with explicit beneficial and harmful distinctions, equal resource *spending* and a conventional context-aware learner.

**Stop rule:** do not add new hypercomplex algebras, cell types or cosmoi to rescue an inconclusive mechanism probe. The current G0-AG1 performance programme remains unexecuted and unpromoted.
