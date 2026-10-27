# G0-B · The edge was useful; growth was not demonstrated

**Date:** 2026-10-25. **Evidence:** M, finite CPU-only paired control. **Claim:** the first apparent dynamic-edge advantage is explained by access to the relevant edge, not by an endogenous developmental mechanism.

HF job: https://huggingface.co/jobs/jesusvilela/6addcdc71e112550ce7e9c69

| Seed | No edge | Fixed informative edge | Scripted growth |
|---:|---:|---:|---:|
| 0 | 0.0402730 | 0.0271582 | 0.0272676 |
| 1 | 0.0409703 | 0.0313854 | 0.0314424 |
| 2 | 0.0375820 | 0.0236377 | 0.0237488 |
| 3 | 0.0345364 | 0.0231494 | 0.0233495 |
| 4 | 0.0376701 | 0.0229377 | 0.0230088 |
| 5 | 0.0468913 | 0.0317171 | 0.0318524 |
| 6 | 0.0422977 | 0.0316590 | 0.0317384 |
| 7 | 0.0406344 | 0.0247173 | 0.0249712 |

Fixed informative edge beats scripted growth on every seed. This null/negative result supersedes any earlier suggestion that G0-A supported morphological advantage.

**Remaining confound:** fixed model has access to edge throughout training, while growth only gains it on schedule. This is an edge-budget matched *endpoint*, not a FLOP-, edge-time-, exploration-, or parameter-matched experiment. We should not reverse the claim and conclude fixed topology is inherently superior.

**Status of main hypotheses:** H1 and H2 untested. No real local graph growth, no concepts, no Z+, no independent semantic representation, no creative generalization. The experiment's value is the successful remote pipeline and detection of a weak comparator.

**Next design decision:** The next baseline should force signal mediation via a local readout region (rather than a global linear sum), and select candidate tendrils without privileged node IDs or information about the hidden cause. Establish path discovery separately from latent concept composition.

## Follow-up implementation status

The G0-B runner has since been replaced by a paired, local-readout protocol
covering fixed 0↔1, seeded random one-edge, scripted-edge, and residual-driven
topology with matched active-contact exposure, update, proposal, and accounting
budgets. Its independent held-out matrix covers AND/XOR, observation/`do(A)`/
`do(B)`, and appearance strata. See [the current protocol](../G0_PROTOCOL.md).

This is an implementation update, **not a new measured result**. No new remote
run or semantic-emergence claim is recorded here. Preserve every result,
including null or negative outcomes, when the protocol is run.
