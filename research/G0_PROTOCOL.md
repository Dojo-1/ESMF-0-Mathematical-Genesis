# G0 experiment: registered minimal falsification protocol

## What is (and is not) being tested

This first experiment tests bounded local routing and a fixed-versus-growing connectivity comparison in a tiny artificial world. It **cannot** establish semiotic emergence or creative cognition. No experiments have been recorded in this document.

## Frozen generation
- Binary latent causes A and B, independently sampled.
- Binary observable cues at two distinct sites, ten independent distracting sites.
- Outcome Y = A AND B; no label naming A or B is provided to the learner.
- Identical training/holdout samples and model initialization per seed.
- Runs deterministic under Python standard-library PRNG, subject to software environment.

## Primary observable
Held-out Brier score; lower is better. Positive `delta_dynamic_minus_fixed` means dynamic growth **lost**.

## Caveats & confounds
1. Fixed and dynamic models currently have different used-edge counts; they are **not** parameter/compute matched. This is a feasibility smoke test, not an effectiveness comparison.
2. The growing graph's edge proposal is scripted by node identity and event count. **It is not endogenous**. Replace with a strictly local surprise-driven candidate sampler before claiming morphogenesis.
3. The logistic predictor lacks an explicit interaction term and therefore is misspecified for XOR and weak for logical composition. The AND task is only an initial sanity check.
4. A single graph edge is not a non-scalar geometric transport; higher fibers and the connection Laplacian are deliberately deferred.
5. A scalar Brier score cannot demonstrate formation of Peircean signs, interpretants, or autonomous concepts.
6. No true Z+ possibility distribution is implemented in the baseline. A probabilistic output is **not** Z+.
7. The local learner's output sums all node contributions centrally and is therefore not a fully local associative readout. The next iteration must replace it with an explicit output region, local message passing, and bounded local learning.

## Required next iteration
- Replace identity-based edge proposals with residual/counterfactual local sampling.
- Match active edges, learning parameters, total updates and compute.
- Separate intervention from observation; split appearance and causal rule.
- Introduce strict unseen compositions and a held-out causal equivalence evaluation.
- Implement observational Gram and signed connection-Laplacian diagnostics **outside** the learning loop.
- Add resource events, provenance, typed remainder and correction ledger.
- Run controlled ablations: no-growth, random-growth, morphology-only, probability-only, Z+-enriched, excitation versus no excitation.

## Promotion rule
No H-to-M promotion without seeds, raw output, environment versions, parity checks, negative controls and meaningful confidence intervals. No M-to-P promotion from numerics.

## Compute safety
No default GPU, internet, model downloads, local Kaggle interaction or process priority changes. Use a bounded CPU-only remote job after explicit tool confirmation. Nothing is scheduled.
