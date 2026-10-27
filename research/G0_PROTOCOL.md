# G0 experiment: bounded falsification protocol

## Evidence boundary

G0-B is a deterministic, standard-library synthetic experiment. It can compare
predictive behavior under specified budgets; it cannot establish semantic
emergence, biological morphogenesis, or an efficiency advantage. The earlier
edge-parity result in `results/G0_B_EDGE_PARITY_2026-10-25.md` is historical:
it compared a fixed edge to a scripted edge with unequal exposure and is not
evidence for the current protocol. The implementation and tests are not
empirical results; only saved run output supports a measured claim.

## Frozen task and paired controls

- Each paired seed supplies the identical AND/observational training stream and
  independent held-out streams to every condition. Observational causes share
  a latent common cause; randomized `do(A)` / `do(B)` assignments break that
  dependence while retaining the other cause's natural mechanism.
- The held-out factorial includes AND and XOR outcomes, observation and `do(A)`
  / `do(B)` interventions, and both appearance values. Latent causes and
  composition names are evaluator-only; appearance independently recodes both
  local causal observations.
- All four conditions have exactly one undirected active contact for every
  training update, the same nine-parameter local predictor, update count,
  periodic proposal count, candidate-scan count, and recorded state size:
  1. `fixed_informative`: fixed 0↔1 contact.
  2. `random_one_edge`: one seeded random fixed contact.
  3. `scripted_edge`: one contact moved on a fixed node/event schedule.
  4. `residual_driven`: one contact proposed from local activity/residual
     correlation at the readout's two physical neighbors, with one bounded
     proposal event per 20 updates. Each neighbor sends its local residual
     correlation as a proposal message; no other region enters this scan.
- The only prediction is at region 1. It uses region 1's local observation and
  appearance plus explicit messages arriving over its incident contact; it
  never sums all region outputs. The label is used for local prediction error,
  but neither latent cause nor the hidden composition is supplied to learning.

The report records paired per-seed Brier scores and per-composition,
intervention, and appearance strata. It also reports class coverage, prediction
variance, message rate, contact/update/proposal budgets, a logical-operation
estimate, and a parameter-state memory estimate. These estimates are matched
accounting measures, not wall-clock FLOPs or physical memory/energy.

## External diagnostics and uncertainty

The observation Gram and signed contact Laplacian are computed only after
training on held-out observations. They are reported as diagnostics and are not
available to topology proposals, parameter updates, or predictions. The
possibility/probability dual representation and specificity constraints
required for Z+ do not exist; Z+ is explicitly deferred and no probability
output is described as Z+.

Retain every paired seed and every stratum, including null, unfavorable, and
degenerate results. Do not select seeds or suppress failures. A performance
claim requires independent replication and paired uncertainty intervals, and
must beat fixed-size and randomized morphology controls. No semantic-emergence
promotion is permitted from this toy benchmark.

## Resource governor

The runner is local standard-library CPU code with bounded seed, episode, and
evaluation limits. It does not schedule remote work, access Kaggle, change
process priority, download models, or use a GPU. For any separately authorized
hosted replication, start with the remote `cpu-basic` profile and bounded
budgets; do not use or interact with the local Kaggle machine. No remote run is
scheduled by this repository.

Reproduce the smoke run and keep its full JSON record:

```bash
python -m unittest discover -s tests -p 'test_controls.py' -v
python -m esmf.controls --seeds 0,1 --episodes 80 --eval-per-stratum 5 \
  --output /path/to/g0b-run.json
```
