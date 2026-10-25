# G0-AB1 · Post-pilot correction log and frozen held-out protocol

Date 2026-10-25. This document was committed **after** debugging the first 8/16-seed G0-AB pilots and **before** the held-out seed range below. The original [G0-AB protocol](G0_AB_ADAPTIVE_PROTOCOL.md) remains unchanged. Never call the amended gate/budding results original preregistration.

## First failed adaptation / changes made

1. The initial SIX-component IMM had no *source-independent* top-level model. In the DIFFUSE pilot, ~56% of examples received unsupported confident orientation decisions. This violated G0-AB's identifiability kill condition. Added a seventh global model H_diffuse: on every observation a newly sampled D produces y~N(0,1+sigma²), without a meaningful T. Renormalized persistent models to prior .8 and diffuse prior .2.
2. Using P(persistent)≥.70 **and** P(T|persistent)≥.95 allowed a decision with as little as .665 joint posterior support. Replaced with decision only when P(persistent,T=t)≥.95. This has a coherent posterior decision-theoretic interpretation.
3. Original consecutive-reset budding rule produced zero new Worldlets in initial jump pilots: real reset evidence was often sharp and isolated. Replaced with a single posterior reset event p_reset>.33, p_persistent≥.85, tick≥2 and a 3-step refractory. Bud emits a typed [R]_id event and reweights future model transitions while preserving conditional T belief. It is a heuristic of the postpilot edition, not proven optimal.
4. Under P(diffuse)≥.80 after four ticks, the adapted controller temporarily switches to cheap even/odd continuity rechecks (one odd probe each four opportunities), instead of wasting focused orientation queries. It continues sampling, rather than permanently freezing inference.

## Pilot and suite

First unguarded six-component 8-seed log: https://huggingface.co/jobs/jesusvilela/6ade138a046eb82b1ece5a19 . The initial false-gluing failure is permanently retained.
Null-model repair 8-seed smoke: https://huggingface.co/jobs/jesusvilela/6ade13c1046eb82b1ece5a3e .
Joint-gate 16-seed pilot: https://huggingface.co/jobs/jesusvilela/6ade13f4046eb82b1ece5a4e .
Reset evidence diagnostic: https://huggingface.co/jobs/jesusvilela/6ade14151e112550ce7ebae4 .
Postpilot reset amendment 16-seed smoke: https://huggingface.co/jobs/jesusvilela/6ade1440046eb82b1ece5a8f .
Normal-package G0-AB regression suite: 11/11 passed: https://huggingface.co/jobs/jesusvilela/6ade14a01e112550ce7ebb34 .

## Held-out experiment frozen now

Evaluate **seed IDs 128 through 191 inclusive** (64 paired seed IDs, two orientation classes each) in each of five distinct environments: stable, drift, twice-reset, every-event-independent, and stable-to-drift switch. All models use identical environment potential outcomes, same sensor-cost law, same six-/seven-state posterior capacity except expressly ablated slow metaplasticity, and budget 8.0 initial plus externally delivered 0.16 each of 12 fast ticks (with cap/spill accounting).

Policies:
- adaptive with fast/meso/slow and provenance-aware query shifts;
- fixed_focus, fixed_quiet, random, adaptive_no_slow;
- equivalent_real as an exact-equivalence control (not an independent algorithm).

Report 128 episodes/world/policy (not independent 128 seeds), seed-cluster bootstrap 2,000 draws of paired orientation-averaged contrasts, Brier and coverage/correct rate in worlds with persistent T, and unsupported confidence on fully independent world, mean abstract spend, mean active and focused probes, bud rate and timely jump detection. Include CPU wall-time from a shared environment; not power.

## Specific gates and interpretation

- An adaptive benefit on energy cost must be stated alongside loss of classification coverage or accuracy.
- Fixed-focus superiority on any environment is a meaningful negative result.
- Diffuse false confidence >0 is a remaining calibration failure even if improved from unguarded pilot.
- Bud precision and recall must both be shown. A temporal Worldlet is a typed memory descriptor and a future model-mode reweighting, **not a grown neuron**.
- Abstract source `F=[[1,D],[0,1]]` is a one-dimensional orbit inside 4D neutral Mat2; there is no demonstrated advantage of 4D topology over an equally informed ordinary real state-space filter. The latter is an exact control.
- Slow learned hazard is a heuristic feedback variable, **not** a posterior over a physical hazard.
- Do not tune these rules using the 128..191 experiment and then report improved results as held-out. Further revisions require new registered seed ranges.

Evidence: mathematical invariants conditional [P], surrogate transitions [A], independent remote test outcomes [M], neural/cognitive/real energy advantage still [H].
