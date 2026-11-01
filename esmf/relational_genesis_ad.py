"""G0-AD: contextual relational learning and a typed Lullian question interface.

An EXPLORATORY, bounded interface. Behavioural feedback is an observed
response relation, not access to a hidden latent semantic label. Evidence
from different Worldlet contexts is never pooled implicitly.

Uses G0-L Correspondence for geometric transport diagnostics. A small
geometric remainder does not certify a semantic equivalence; neither the
question grammar nor the local learner establishes cognitive emergence,
a Noether conservation law, or superiority over a real-valued comparator.
"""

from dataclasses import dataclass
import json
import math

from .geometry_k import bilinear
from .remainder import Correspondence, sub, Worldlet


RELATIONS = ("concordance", "difference", "contrariety", "mediation")
MODES = ("whether", "what", "why", "how")
DECISIONS = ("preserve", "distinguish", "contested", "abstain")


@dataclass(frozen=True)
class Question:
    context: str
    transformation: str
    mode: str
    relation: str
    text: str
    evidence: str = "S"

    def __post_init__(self):
        if self.mode not in MODES or self.relation not in RELATIONS:
            raise ValueError("untyped interrogative")
        if self.evidence != "S":
            raise ValueError("a generated question is not verified evidence")


def questions(context, transformation):
    """Lullian-inspired GENERATOR, not a syllogistic truth engine.

    Difference, concordance and contrariety are different question types:
    a changed behaviour is not automatically a logical contradiction.
    """
    if not context or not transformation:
        raise ValueError("context and transformation required")
    return (
        Question(context, transformation, "whether", "concordance",
                 f"Does {transformation} preserve the relevant response in {context}?"),
        Question(context, transformation, "what", "difference",
                 f"What response differences arise after {transformation} in {context}?"),
        Question(context, transformation, "why", "contrariety",
                 f"What observation would contradict a claimed invariance under "
                 f"{transformation} in {context}?"),
        Question(context, transformation, "how", "mediation",
                 f"Which typed translation could connect the responses under "
                 f"{transformation} in {context}, with explicit loss?"),
    )


@dataclass(frozen=True)
class Encounter:
    """One observed response comparison, with no hidden-world class labels."""

    context: str
    transformation: str
    response_before: str
    response_after: str
    source_event: str

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (
                self.context, self.transformation, self.response_before,
                self.response_after, self.source_event)):
            raise ValueError("encounter fields must be nonempty strings")

    @property
    def preserved(self):
        return self.response_before == self.response_after


@dataclass
class Evidence:
    effective_preserve: float = 0.0
    effective_change: float = 0.0
    seen_preserve: int = 0
    seen_change: int = 0

    @property
    def contested(self):
        return self.seen_preserve > 0 and self.seen_change > 0


class ContextualSymmetry:
    """Bounded online, two-clock behavioural-relation learner.

    An observation is externally observable task feedback. No classes,
    geometric oracles, source continuity, or candidate symmetry truths
    are supplied. Contrary examples in the same declared context forbid
    silently announcing a universal symmetry: they require abstention
    and may warrant a separately authorized context split.

    Fast effective weights dissipate. Raw witness counts and bounded event
    provenance remain, so decay cannot erase a counterexample by fiat.
    """

    def __init__(self, minimum_evidence=2.0, decay=0.8,
                 max_pairs=128, max_events=4096):
        if (minimum_evidence <= 0 or not math.isfinite(minimum_evidence)
                or not 0 < decay <= 1 or not math.isfinite(decay)
                or max_pairs < 1 or max_events < 1):
            raise ValueError("invalid learner bounds")
        self.minimum_evidence = float(minimum_evidence)
        self.decay = float(decay)
        self.max_pairs = int(max_pairs)
        self.max_events = int(max_events)
        self.states = {}
        self.provenance = []
        self.lineage = {}
        self.fast_clock = 0
        self.slow_clock = 0

    def observe(self, encounter):
        if not isinstance(encounter, Encounter):
            raise TypeError("expected Encounter")
        key = (encounter.context, encounter.transformation)
        if key not in self.states and len(self.states) >= self.max_pairs:
            raise RuntimeError("pair capacity exhausted: abstain or authorize growth")
        if len(self.provenance) >= self.max_events:
            raise RuntimeError("provenance capacity exhausted: no silent eviction")
        if any(e.source_event == encounter.source_event for e in self.provenance):
            raise ValueError("duplicate source_event cannot count twice")
        state = self.states.setdefault(key, Evidence())
        if encounter.preserved:
            state.effective_preserve += 1.0
            state.seen_preserve += 1
        else:
            state.effective_change += 1.0
            state.seen_change += 1
        self.provenance.append(encounter)
        self.fast_clock += 1
        return self.decide(*key)

    def dissipate(self, steps=1):
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("nonnegative integer slow steps required")
        factor = self.decay ** steps
        for state in self.states.values():
            state.effective_preserve *= factor
            state.effective_change *= factor
        self.slow_clock += steps

    def decide(self, context, transformation):
        state = self.states.get((context, transformation))
        if state is None:
            return "abstain"
        if state.contested:
            return "contested"
        if (state.seen_preserve and
                state.effective_preserve >= self.minimum_evidence):
            return "preserve"
        if (state.seen_change and
                state.effective_change >= self.minimum_evidence):
            return "distinguish"
        return "abstain"

    def report(self, context, transformation):
        state = self.states.get((context, transformation), Evidence())
        return {
            "context": context,
            "transformation": transformation,
            "decision": self.decide(context, transformation),
            "effective_preserve": state.effective_preserve,
            "effective_change": state.effective_change,
            "witnesses_preserve": state.seen_preserve,
            "witnesses_change": state.seen_change,
            "R_sem": "unvalidated behavioural-to-semantic correspondence",
            "fast_clock": self.fast_clock,
            "slow_clock": self.slow_clock,
        }

    def propose_split(self, context, transformation):
        """Propose, never perform, a Worldlet split after conflicting feedback."""
        state = self.states.get((context, transformation))
        if state is None or not state.contested:
            return None
        return {
            "parent": context,
            "transformation": transformation,
            "status": "H",
            "R_sem": "competing response observations within one declared context",
            "requires": "independent evidence and budgeted split authorization",
        }

    def fork_context(self, parent, child):
        """Explicit new context: do not copy a parent's disputed conclusions."""
        if not parent or not child or parent == child:
            raise ValueError("distinct parent and child contexts required")
        if child in self.lineage or any(k[0] == child for k in self.states):
            raise ValueError("context already exists")
        if not any(k[0] == parent for k in self.states) and parent not in self.lineage:
            raise ValueError("unknown parent context")
        self.lineage[child] = parent


def transport_audit(correspondence, source_state, target_state):
    """External G0-L audit, keeping coordinate and semantic remainders apart.

    Signed quadratic form is NOT named a nonnegative loss under an indefinite
    metric. Chart re-expression co-transforms it under G0-L conventions.
    """
    if not isinstance(correspondence, Correspondence):
        raise TypeError("expected G0-L Correspondence")
    predicted = correspondence.travel(source_state)
    defect = sub(target_state, predicted)
    return {
        "predicted_target": predicted,
        "R_transport_target": defect,
        "R_transport_roundtrip": correspondence.remainder(source_state),
        "signed_target_quadratic": bilinear(
            defect, correspondence.target.metric, defect),
        "semantic_equivalence": "unassessed",
    }


def contrast_audit(correspondence, first, second, known_distinct, tol=1e-12):
    """External observer: detect a declared distinction erased by transport.

    known_distinct is a task witness supplied ONLY to the external auditor,
    not to ContextualSymmetry. This cannot prove semantic truth by itself.
    """
    if not isinstance(correspondence, Correspondence):
        raise TypeError("expected G0-L Correspondence")
    if tol < 0 or not math.isfinite(tol):
        raise ValueError("invalid audit tolerance")
    fa = correspondence.travel(first)
    fb = correspondence.travel(second)
    collapsed = all(abs(a - b) <= tol for a, b in zip(fa, fb))
    return {
        "collapsed_at_target": collapsed,
        "false_merge_of_declared_contrast": bool(known_distinct and collapsed),
        "R_transport_pair": sub(fa, fb),
        "known_distinct_is_external_witness": True,
    }


def selfcheck():
    """Deterministic contract demonstration, NOT an empirical result."""
    learner = ContextualSymmetry()
    for i in range(3):
        learner.observe(Encounter("numeral-reading", "half-turn",
                                  "read-six", "read-nine", f"numeral-{i}"))
        learner.observe(Encounter("physical-object", "half-turn",
                                  "point-to-A", "point-to-A", f"object-{i}"))
    a = Worldlet(3, kind="sensory")
    b = Worldlet(2, kind="response")
    correspondence = Correspondence(
        a, b, ((1., 0., 0.), (0., 1., 0.)),
        ((1., 0.), (0., 1.), (0., 0.)))
    witness = contrast_audit(correspondence, (0., 0., 0.),
                             (0., 0., 1.), known_distinct=True)
    assert witness["false_merge_of_declared_contrast"]
    assert learner.decide("physical-object", "half-turn") == "preserve"
    assert learner.decide("numeral-reading", "half-turn") == "distinguish"
    return {
        "status": "contract-only",
        "numeral": learner.report("numeral-reading", "half-turn"),
        "object": learner.report("physical-object", "half-turn"),
        "question_count": len(questions("numeral-reading", "half-turn")),
        "kernel_contrast_audit": witness,
        "observed_learning_is_not_semantic_certification": True,
    }


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2, sort_keys=True))
