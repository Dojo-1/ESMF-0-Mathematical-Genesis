# G0-Y · Partial-observation embodied benchmark — 2026-10-27

## Run and evidence boundary

Ran the frozen protocol in [G0_Y_EMBODIED_PROTOCOL.md](../G0_Y_EMBODIED_PROTOCOL.md):
32 paired seeds × 9 scenarios × 5 matched four-coefficient predictors = 1,440
seed/model rows. Each row aggregates five training episodes and three held-out
movement/recovery composition episodes (28 events each). The reproducible
seed-level output, including fitted weights and per-episode held-out errors, is
[G0_Y_EMBODIED_2026-10-27.jsonl](G0_Y_EMBODIED_2026-10-27.jsonl) (1.35 MB).

This is bounded synthetic measurement [M], not evidence of semantic learning,
biological organs, hardware efficiency, or a full 4D neutral-bundle learner.
The comparator called `geometric` is deliberately the exact calibrated
real-valued chart predictor under a different label.

## Held-out RMSE by scenario

Lower is better. `Chart/geometry` is one identical model, not two independent
systems. The raw file records Gaussian NLL, training-estimated predictive
variance, nominal 90% coverage, and paired seed-level errors for every row.

| Scenario | Recurrent | Kalman-like | Hierarchical | Chart / geometry |
|---|---:|---:|---:|---:|
| Noncommuting | 0.3839 | 0.4267 | 0.4392 | 0.3820 |
| Commuting | 0.3525 | 0.4316 | 0.3997 | 0.3525 |
| Dissipative | 0.3006 | 0.3497 | 0.3908 | 0.3007 |
| Out-of-family dynamics | 0.2142 | 0.2594 | 0.2758 | 0.2121 |
| No regulatory organs | 0.3406 | 0.3828 | 0.4010 | 0.3381 |
| No innovation remainder | 0.3907 | 0.4367 | 0.4527 | 0.3892 |
| Fixed charts | 0.3759 | 0.4197 | 0.4336 | 0.3759 |
| False charts | 0.3839 | 0.4267 | 0.4392 | 0.3949 |
| Hidden seam change | 0.5540 | 0.5573 | 0.4801 | 0.5541 |

Paired recurrent-minus-chart RMSE differences (95% normal intervals) were:

- Noncommuting: +0.0020 [0.0012, 0.0027].
- Commuting and fixed charts: exactly 0 by construction.
- Dissipative: −0.0001 [−0.0002, 0.0000].
- Out-of-family: +0.0021 [0.0013, 0.0029].
- No regulatory organs: +0.0025 [0.0018, 0.0031].
- No innovation remainder: +0.0015 [0.0008, 0.0021].
- False charts: −0.0109 [−0.0142, −0.0077].
- Hidden seam change: −0.0001 [−0.0003, 0.0001].

The strongest simple recurrent predictor was generally close to the ordinary
calibrated chart baseline. Kalman-like and hierarchical variants did not
improve this task. False chart calibration harmed the chart-aware model
relative to the recurrent control. Geometry and chart-aware weights, predictions,
and arithmetic counts tied exactly on every paired seed; no geometric
advantage is demonstrated.

The `no_transport_remainder` label refers only to the scalar sensor-innovation
feature ablation in this benchmark, not to G0-L's typed gauge-covariant
transport remainder.

## Failure and resource audit

All 160 hidden-seam training rows (32 seeds × 5 models) had no odd-winding
training evidence and returned `seam_hypothesis="unknown"`. Held-out hidden seam
changes caused severe uncertainty miscalibration: chart-aware nominal 90%
coverage was 12.9%, and mean Gaussian NLL was 65.1. The hierarchical model had
lower RMSE (0.4801) there but also only 13.4% nominal coverage. The experiment
does not identify the changed seam; this is a distribution-shift failure, not
evidence for a topology learner.

The positive resource ledger closed to at most `7.4e-15`; no episode crossed
the reserve floor. Test resource expenditure averaged 1.54 abstract units per
episode. This accounting is not physical energy. All models processed the same
84 test events per seed/scenario and counted 336 prediction multiply-adds.
Median Python fit-plus-test wall time was about 0.58 ms per seed/model/scenario.
The 32-byte model state and 7,168-byte feature payload are arithmetic
estimates, not profiled process memory; no watts were measured.

## Conclusion

The main positive result is procedural: a partial, delayed sensor stream,
action costs, recovery, competing four-coefficient predictors, chart/seam
controls, and 32 paired seeds are now executable and auditable. Predictive
performance does not support a geometric or regulatory-organ advantage in
this surrogate. Novel seam changes expose an unresolved calibration failure.
The benchmark does not include a full Kalman/particle filter, a trained
hierarchical RNN, genuine connection-curvature ablation, or integrated neutral
Mat₂(R) transport; those remain outside this result's evidence.
