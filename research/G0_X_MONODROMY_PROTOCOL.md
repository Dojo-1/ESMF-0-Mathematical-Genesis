# G0-X protocol: infer a global orientation class from causal fiber observations
Date: 2026-10-25. Written before running G0-X training experiments.

## Object and evidence
Base S¹×R, associated neutral Mat₂(R) fiber with Q(F)=det F. A locally acting reversible causal world supplies fiber Jacobian transports. A candidate deck transformation is either I (orientable family) or P(F)=F^T (orientation-reversing family), where P^T H P=H but det P=-1. Test inference of one unknown Z₂ topological monodromy bit, not discovery of arbitrary global topology.

A physical winding number w around S¹ maps an initial fiber state F0 to `P^(twist * (w mod 2)) L(J_path) F0`. At each chart seam, the local action operator must be conjugated; this formula expresses that global equivariance.

## Observations and data
Action parameters a,b are fitted separately from 140 short noisy state-transition traces using the previously frozen G0-S learner. The topology learner then receives a small number of independent fiber start/end observations of paths whose windings are known from action-derived position. Fiber end sensors have additive small independent Gaussian noise. It must choose between two declared monodromy hypotheses I/P.

Two worlds share the same physical action dynamics, with true seam in {identity, transpose}. Train on a handful of single-winding trajectories. Evaluate on distinct paths, both winding directions, no-winding paths, and a double-winding counterfactual.

## Preregistered controls and kill criteria
- Fixed identity-only model.
- Fixed transpose-only model.
- Learned binary selector from training observations.
- Ordinary real-valued recurrence with an identical learned binary parity gate: *identical predictive capability*. This is crucial and not grounds for geometric superiority.
- **No-odd-winding training:** both candidates become indistinguishable; the algorithm must return 'unidentified', not declare a topology.
- Double winding: P²=I should restore the original orientation.
- Independently check neutral form and physical Jacobian preservation after seam gluing.
- Reject if single-winding orientation selection fails systematically, if models separate without odd winding data, or if prediction fails under chart-equivalent representation of the same path.
- Require per-seed reports, common physical parameters, noise levels, and fixed training budgets.

## What remains unproved
A two-class candidate learner trained on a world built from those same candidate classes is an identifiability demonstration [M when run], not a proof that a general AI learns topology or that its cognition/compute efficiency improves. This synthetic observation process contains a faithful frame sensor and externally supplied causal action labels; living organisms do not necessarily have either.
