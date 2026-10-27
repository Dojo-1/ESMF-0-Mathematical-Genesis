# G0-Y · Partial-observation embodied prediction protocol

Date: 2026-10-27. This protocol freezes the evaluation before the 32-seed
measurement. The implementation and one-seed smoke run preceded this document;
the smoke run is not included as evidence.

## Question and evidence boundary

Does a four-coefficient action-conditioned predictor retain useful held-out
sensor prediction when momentum and resource are hidden behind noisy, delayed
channels, action costs, damage, and recovery? Compare simple recurrent,
Kalman-like, hierarchical, and chart-calibrated real predictors on paired
traces. The fifth `geometric` row is an exact chart-aware relabeling control:
it is required to tie the ordinary real-valued chart-aware predictor and is
not a 4D geometric learner.

The world has hidden cylinder angle, momentum, damage, and positive resource.
The sensor packet contains only noisy local angle, momentum, reserve, damage,
and chart metadata. Angle/damage are delayed by one tick, momentum by two;
reserve and local chart metadata are current. A spinal-loop policy chooses
motor, coast, rest, and repair actions from sensor packets. Enteric-like
accounting charges maintenance, action, and sensing costs; an autonomic-like
gate updates every two fast ticks and can pay for focused sensing. Action,
sensor, and resource state update on the fast event clock. These are
computational analogies, not biological organs or measured physical power.

The target is the next delayed scalar momentum reading—not hidden state,
monodromy, or a full transported frame. The dynamics include noncommuting,
commuting, dissipative, and out-of-family cases. Sensor-chart and seam controls
are scalar observation-interface tests; the benchmark does not integrate a
learned Mat₂(R) neutral (2,2) carrier. G0-Q remains distinct from the
associated construction in G0-U, and this experiment does not bridge them.

## Frozen design

- 32 paired seeds (0–31), 5 training episodes and 3 held-out episodes per
  scenario/seed; each episode is 28 events.
- Every predictor receives identical generated traces, actions, event budgets,
  and resource/noise realizations within a scenario and seed.
- Held-out episodes use an interleaved movement/recovery action composition
  (including scheduled rest interventions) that is not used in training.
- Five predictors have four fitted real coefficients each. Recurrent,
  scalar-Kalman-like, and two-rate hierarchical features differ; chart-aware
  features invert only the sensor-reported positive chart gain. Geometric is
  an exact duplicate control, not an additional capacity or geometry claim.
- Controls: commuting dynamics; dissipative dynamics; out-of-family test
  dynamics; no regulatory organs; no scalar innovation-remainder correction;
  fixed charts; false chart calibration; and training with no odd winding
  followed by a hidden seam change. The no-winding case must report seam
  hypothesis `unknown`.
- The `no_transport_remainder` scenario specifically ablates the scalar
  observation-innovation correction in the predictor; it does not remove a
  gauge-covariant G0-L transport residual.
- Each model uses a four-parameter ridge-stabilized least-squares fit. The
  prediction variance is estimated on training traces and frozen for held-out
  Gaussian NLL and nominal 90% interval coverage.

## Recorded outcomes and falsifiers

Raw per-seed rows include fitted weights, per-episode held-out squared errors,
RMSE, NLL, nominal interval coverage, winding/identifiability, resource
closure, event counts, multiply-add counts, CPU wall time, and an explicitly
approximate feature-matrix memory payload. Event count is not energy. No watts
are available or inferred.

Report seed-level errors and paired RMSE differences against the ordinary
chart-aware baseline with 95% normal intervals. A chart-aware/geometric
difference is an implementation failure, since their features are identical.
Do not claim topology from no-winding training, semantic understanding from
prediction loss, or a geometry/energy advantage from this scalar surrogate.
The finite-difference action-gradient and positive chart-rescaling checks are
independent algebraic checks; the learner has no gradient-based optimizer.

Run and retain raw results with:

```bash
python -m esmf.embodied_y --seeds 32 --start 0 \
  --output research/results/G0_Y_EMBODIED_2026-10-27.jsonl
```

Status beyond this bounded synthetic benchmark remains H (hypothesis). It
does not validate a full recurrent/Kalman/particle-filter implementation,
full 4D neutral-fiber transport, hardware efficiency, or biological semantics.
