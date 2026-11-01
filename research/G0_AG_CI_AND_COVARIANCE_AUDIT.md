# G0-AG correction review — CI, chart transport and endogenous growth

Scope: G0-AG draft PR #18, stacked on G0-AF PR #17. This is a correction-preserving audit, **not** a new cognitive or biological result. Historical G0-F/AC/Y numerical artifacts remain untouched.

## What actually failed

Workflows 41295493316, 41295532623 and 41295546993 were red **solely because Ruff lint reported two F401 unused imports** in tests/test_glial_quaternion_ag.py: ContinuousSections and Glia. Python 3.11, 3.12, 3.13 and historical reproduction succeeded in those runs. Commit 85049592bd21667154824eac4637eeaf71b653e4 removed the imports. Later commit 6274b12f4020bfa3b5cf24df69ccb8156ef623bd passed all five checks in [workflow 41295676923](https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis/actions/runs/41295676923), with 205 tests.

Green CI verifies the declared checks **at that revision**, not a semantic, learning, or biological theorem. Subsequent commits need a separate green CI.

## Adversarial representation review

**Defect A: a covariance gap in the learning event.** The initial code tested gauge invariance of quaternionic transport discrepancy in isolation, but its sensory learning operation mapped a G0-AE four-vector directly into the active neurite phase, without co-transforming the local observer. Thus the *learning event* could change under arbitrary local frames even when the isolated distance formula stayed invariant.

**Correction A:** A-tagged observer/motor frame \(C_i\in S^3\). The chosen synthetic projection is \(\Phi_i(y)=C_iy/\|y\|\) for \(y\ne0\). A local left gauge \(g_i\) maps \(C_i\mapsto g_iC_i\), \(q_i\mapsto g_iq_i\), and \(U_{ij}\mapsto g_jU_{ij}g_i^{-1}\). New regression tests frame-transform the learner **before** its resource-paid phase update, and verify both its event hazard and later paid connection repair. These contracts do not cover historical records without chart identifiers.

**Defect B: zero-sight failure.** A zero R4 observation was valid in the synthetic environment but not representable as a unit quaternion. Previously, attempting to normalize it could raise while enumerating unrelated biological hazards. The new typed adapter abstains on the unrepresentable observation, with a diagnostic flag. It does not pretend to recover its missing direction.

**Defect C: motor action was not gauge-equivariant.** Applying a fixed quaternionic direction \(R\) to a phase \(q\) in an arbitrary local frame is not an intrinsic operation. Under left gauge \(g\), \(R(gq)\ne g(Rq)\) generally; the elementary counterexample is \(ij=k\) versus \(ji=-k\). The corrected local motor action is \(\widehat R=C_iR C_i^{-1}\). The paid new neurite phase and its transport now co-transform under the chosen live gauge, while structural material and ATP accounting remain unchanged. This is tested by a metamorphic whole-growth-event comparison, not simply a quadratic identity.

**Open noninjective projection:** Mat2(R) and H both have real dimension four, but are *not algebra-isomorphic*. Their chosen coefficient identification is an assumption, not automatic meaning transport. Normalizing an observation identifies positive multiples of the same R4 vector; amplitude is a potentially meaningful lost distinction. The diagnostic exposes a typed observation-norm transport remainder and retains the synthetic event reference. This limitation must be tested rather than ignored.

## Evidence and promotion

- **P (conditional standard algebra):** norm preservation and covariance of correctly conjugated unit-quaternion transports; the physical-resource ledgers are conditional identities of the synthetic token model.
- **M (software-suite outcome only):** dated CI passes on exact commits, not held-out task effectiveness.
- **H:** useful biological/semantic emergence, glial advantages and special hypercomplex learning gains.
- **A/S:** chosen observation-to-quaternion identification and glial/neurite biological analogies.
- **R:** the stronger assertion that an isolated gauge-invariant mismatch automatically makes a complete learning and growth process gauge-invariant.

The current tests are correlated author-written implementation contracts. They do not constitute independent scientific reproduction. The audit requires eventual external controls, altered gauges/observational charts, sensor perturbations and held-out behavior.

**Next falsifiable comparison:** paired, equal-information and equal-resource tests against a four-real-coordinate implementation of Hamilton multiplication, a fixed-geometry learner, and an equally informed conventional estimator. Track conditional false equivalences, coverage, transport errors, incorrect budding, material/ATP expenditure, and memory under representation changes. If the hypercomplex model ties or loses, retain sound mathematics while retiring any claimed distinctive cognitive advantage. Do not automatically merge the stacked PR chain.
