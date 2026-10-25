# G0-AA · Change-point-aware temporal Worldlet repair (pre-registered)
Date: 2026-10-25. This was written after G0-Z identified a 41.4% unsupported orientation declaration rate on a deliberately misspecified mid-episode-reset world, but BEFORE any G0-AA code execution.

## Mathematical type
Observation history `(y_t,parity_t,sigma_t)` is a family of *partial scalar sections* whose common latent source identity is uncertain. The candidate model family is a FINITE Gaussian-mixture hypothesis space, not a genuine sheaf-descent theorem:

- persistent: T∈{identity,transpose}, one shared latent D throughout;
- exactly one change: T∈{identity,transpose}, unknown boundary K∈{1,2,3,4,5}, independent D_pre and D_post with the SAME orientation T;
- diffuse: each observation comes from a new D_t, with no identifiable global orientation.

Priors: persistent total .45 (equally split), one-change total .35 (equally split over T and K), diffuse .20. Observations use G0-Y scalar projection, sign inversion on odd winding under transpose, and exact online Gaussian conjugate updating for each declared hypothesis.

**One-time finite repair:** if the model's P(change)≥.50 and max_K P(K|change)≥.30, create two *temporal Worldlet descriptors* (before/after K) carrying explicit boundary uncertainty and evidence source. This is a typed inference split, not the physical growth of neurons or an achieved categorical/sheaf gluing structure. No implicit equality between temporary representation split and biological ontogenesis.

## Frozen stimuli and paired controls
G0-Z fixed six-observation sequence and same sensor/resource schedule: even-focus, odd-focus, even-quiet, odd-quiet, even-quiet, odd-quiet, costing 5.22 simulated units from budget 5.5.

Worlds with 128 seeds each:
1. persistent identity;
2. persistent transpose;
3. one-change identity at K=3;
4. one-change transpose at K=3;
5. independent D_t at every observation (orientation unidentifiable).

The SAME observations are fed to (i) naive original two-orientation filter, (ii) G0-Z provenance-only three-state model and (iii) new G0-AA finite-change mixture. Compare orientation declaration/accuracy on four worlds with meaningful T; unsupported declarations on diffuse world; posterior mass on each continuity type; correct/false split rates; K recovery conditional on split; CPU overhead. Keep all priors fixed for initial runs, never tune thresholds after observing test output.

## Required gates
- Normalized positive posterior and finite Gaussian variances.
- One single observation cannot distinguish these 13 candidate hypotheses.
- Reproduce the exact G0-Z world streams and posteriors on shared/independent/control worlds, preventing uncontrolled source differences.
- On all-even data, orientation conditional on persistent/change models remains unidentifiable at balanced priors, while continuity may update.
- Given true stable world, a new Worldlet split is a false-positive event (must be counted).
- Given one-change world, detected boundary must agree with K=3 when claimed, otherwise report a wrong split.
- Under unmodeled multiple change points or gradually drifting D, this one-change model is misspecified and may become overconfident. Report explicitly; no universal proof of identity continuity.
- Compare strongest ordinary real Bayesian model with identical hypothesis list: no claim of exotic representation benefit.
- False gluing reduction is meaningful only at controlled true-orientation coverage, as changing priors/thresholds can trivially force abstention.

## Expected scientific boundaries
P: elementary finite Gaussian mixture and conditional model equations. A: one reset maximum, known parity schedule, given sensor noise/energy laws, selected priors. M: bounded remote runs when available. H: morphogenetic benefit from new memory organ; biological economy; deeper semantic identity. [R]_id = unassigned continuity; [R]_transport distinct from [R]_id.
