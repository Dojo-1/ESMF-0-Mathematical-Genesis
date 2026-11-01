"""G0-AE: telos-native *joint* biological/geometric/relational dynamics.

A finite continuous-time Markov generator (CTMC), not a sequential pipeline.
All channel hazards are computed from the SAME joint state; the mathematical
object is their sum. A Gillespie trace serializes individual stochastic
events but does not impose a cognitive stage order.

The organism's operational telos is a VIABILITY REGION and revisable local
distinctions, not a hidden semantic oracle or single scalar reward.
A finite classical Markov generator admits an exact diagonal GKSL/Lindblad
embedding (see protocol). This is not a quantum brain or quantum advantage.

Only bounded synthetic contract tests are supported. No empirical claim.
"""
from dataclasses import dataclass, replace
import json
import math
import random

from .associated22_u import neutral_square
from .geometry_k import identity, inverse, matvec
from .orientation_seam_w import P
from .relational_genesis_ad import questions, transport_audit
from .remainder import Correspondence, Worldlet


I4 = identity(4)


@dataclass(frozen=True)
class EncounterView:
    """One locally accessible before/after response and geometric sighting.

    The probability belongs to the *external* environment generator, not to
    the endogenous relation/geometric decision laws.
    """

    before: str
    after: str
    source: tuple
    target: tuple
    probability: float

    def __post_init__(self):
        if not self.before or not self.after:
            raise ValueError("responses must be observable nonempty strings")
        if (len(self.source) != 4 or len(self.target) != 4 or
                any(not math.isfinite(v) for v in (*self.source, *self.target))):
            raise ValueError("Mat2(R) observations need four finite coordinates")
        if not math.isfinite(self.probability) or self.probability <= 0:
            raise ValueError("invalid environment observation probability")

    @property
    def agrees(self):
        return self.before == self.after


@dataclass(frozen=True)
class EncounterWorld:
    """A finite synthetic laboratory; not the organism's learned world model.

    Each local *context* offers two interrogative probes, corresponding to
    Llull-inspired "whether" and "why/contrariety" operations. A probe
    changes which experience the environment may return. Its probability
    kernel is used only by the EXTERNAL joint-process generator.
    """

    contexts: tuple
    probes: tuple
    source_chart: tuple = I4
    target_chart: tuple = I4

    def __post_init__(self):
        if not self.contexts or len(set(self.contexts)) != len(self.contexts):
            raise ValueError("unique nonempty contexts required")
        if any(not isinstance(c, str) or not c for c in self.contexts):
            raise ValueError("invalid context name")
        if len(self.probes) != len(self.contexts):
            raise ValueError("each context requires both question modes")
        for pair in self.probes:
            if len(pair) != 2:
                raise ValueError("two modes required per context")
            for outcomes in pair:
                if not outcomes or not all(isinstance(o, EncounterView) for o in outcomes):
                    raise ValueError("nonempty finite observation menus required")
                if abs(sum(o.probability for o in outcomes) - 1.) > 1e-12:
                    raise ValueError("environment probabilities must normalize")
        if (len(self.source_chart) != 4 or len(self.target_chart) != 4 or
                any(len(row) != 4 for row in (*self.source_chart, *self.target_chart))):
            raise ValueError("charts must be 4 by 4")
        # Require invertibility before relying on G0-L reframe.
        inverse(self.source_chart)
        inverse(self.target_chart)


@dataclass(frozen=True)
class Distinction:
    """Saturating witness counts versus independently dissipating salience."""

    seen_same: int = 0
    seen_different: int = 0
    live_same: int = 0
    live_different: int = 0

    def observe(self, agrees, cap):
        if agrees:
            return replace(self, seen_same=min(cap, self.seen_same + 1),
                           live_same=min(cap, self.live_same + 1))
        return replace(self, seen_different=min(cap, self.seen_different + 1),
                       live_different=min(cap, self.live_different + 1))

    def fade(self):
        return replace(self, live_same=max(0, self.live_same - 1),
                       live_different=max(0, self.live_different - 1))

    @property
    def active_conflict(self):
        return self.live_same > 0 and self.live_different > 0

    @property
    def witnessed_conflict(self):
        return self.seen_same > 0 and self.seen_different > 0


@dataclass(frozen=True)
class Organism:
    """Finite typed state; it is not one globally meaningful latent vector."""

    resource: int
    material: int
    context: int
    question: int
    geometry: tuple
    distinctions: tuple
    buds: tuple
    last: tuple | None = None  # (context, question, observed outcome index)

    @classmethod
    def seed(cls, world, resource=3, material=2):
        n = len(world.contexts)
        return cls(resource, material, 0, 0, (0,) * n,
                   (Distinction(),) * n, (False,) * n)


@dataclass(frozen=True)
class Channel:
    name: str
    rate: float
    destination: Organism


def _replace_at(values, index, value):
    result = list(values)
    result[index] = value
    return tuple(result)


class JointGenesis:
    """One generator: metabolism, interrogation, geometry, memory, anatomy.

    All hazards depend on the present joint state, not on a previously
    executed module. No local learner receives hidden observation
    probabilities, test equivalence labels or an external geometric oracle.
    """

    def __init__(self, world, resource_cap=6, material_cap=4,
                 witness_cap=3):
        if not isinstance(world, EncounterWorld):
            raise TypeError("expected EncounterWorld")
        if (not isinstance(resource_cap, int) or resource_cap < 3 or
                not isinstance(material_cap, int) or material_cap < 2 or
                not isinstance(witness_cap, int) or witness_cap < 2):
            raise ValueError("invalid viability or witness capacity")
        self.world = world
        self.resource_cap = resource_cap
        self.material_cap = material_cap
        self.witness_cap = witness_cap
        w = Worldlet(4, kind="associated-neutral-fiber")
        # Positive metric for observable discrepancy; neutral determinant
        # is tracked separately and is NEVER a positive metabolic energy.
        self.maps = tuple(
            Correspondence(w, w, f, f).reframe(
                world.source_chart, world.target_chart)
            for f in (I4, P)
        )

    def viable(self, s):
        n = len(self.world.contexts)
        if not isinstance(s, Organism):
            return False
        if (not isinstance(s.resource, int) or not 0 <= s.resource <= self.resource_cap or
                not isinstance(s.material, int) or not 0 <= s.material <= self.material_cap or
                not isinstance(s.context, int) or not 0 <= s.context < n or
                s.question not in (0, 1) or
                len(s.geometry) != n or len(s.distinctions) != n or
                len(s.buds) != n or any(g not in (0, 1) for g in s.geometry) or
                any(type(b) is not bool for b in s.buds)):
            return False
        for d in s.distinctions:
            if not isinstance(d, Distinction):
                return False
            values = (d.seen_same, d.seen_different, d.live_same, d.live_different)
            if any(not isinstance(v, int) or not 0 <= v <= self.witness_cap
                   for v in values):
                return False
            if d.live_same > d.seen_same or d.live_different > d.seen_different:
                return False
        if s.last is not None:
            if not isinstance(s.last, tuple) or len(s.last) != 3:
                return False
            c, q, k = s.last
            if not all(isinstance(i, int) for i in s.last):
                return False
            if not (0 <= c < n and q in (0, 1) and
                    0 <= k < len(self.world.probes[c][q])):
                return False
        return True

    def current_question(self, state):
        if not self.viable(state):
            raise ValueError("organism is not viable")
        candidate = questions(self.world.contexts[state.context], "encounter-transformation")
        return candidate[0 if state.question == 0 else 2]

    def observed_pair(self, state):
        """Only the ALREADY OBSERVED pair, never the hidden environment law."""
        if not self.viable(state):
            raise ValueError("organism is not viable")
        if state.last is None:
            return None
        c, q, k = state.last
        return self.world.probes[c][q][k]

    def geometric_defect(self, state, alternative=None):
        """G0-L positive metric mismatch for the last locally observed pair.

        A genuinely neutral (2,2) invariant is available separately through
        neutral_square; it must not be used as a nonnegative loss.
        """
        sight = self.observed_pair(state)
        if sight is None or state.last[0] != state.context:
            return None
        candidate = state.geometry[state.context] if alternative is None else alternative
        if candidate not in (0, 1):
            raise ValueError("unknown transport candidate")
        x = matvec(self.world.source_chart, sight.source)
        y = matvec(self.world.target_chart, sight.target)
        squared = transport_audit(self.maps[candidate], x, y)["signed_target_quadratic"]
        if squared < -1e-9 or not math.isfinite(squared):
            raise ArithmeticError("positive observation metric violated")
        return max(0., squared)

    def neutral_diagnostic(self, state):
        """External algebraic diagnostic; not an endogenous meaning oracle."""
        sight = self.observed_pair(state)
        if sight is None:
            return None
        return neutral_square(sight.target) - neutral_square(sight.source)

    def channels(self, state):
        if not self.viable(state):
            raise ValueError("organism is not viable")
        out = []

        def add(name, rate, destination):
            if rate < 0 or not math.isfinite(rate):
                raise ArithmeticError("invalid CTMC hazard")
            if rate and destination != state:
                if not self.viable(destination):
                    raise AssertionError("transition violates viability envelope")
                out.append(Channel(name, rate, destination))

        c = state.context
        d = state.distinctions[c]
        bud_count = sum(state.buds)
        # All channels are declared together: there is NO canonical order
        # between nutrient influx, sensing, geometrical repair, or growth.
        if state.resource < self.resource_cap:
            add("energy_supply", .70, replace(state, resource=state.resource + 1))
        if state.material < self.material_cap:
            add("material_supply", .23, replace(state, material=state.material + 1))
        if state.resource > 0:
            add("maintenance", .16 + .10 * bud_count,
                replace(state, resource=state.resource - 1))
        if len(state.geometry) > 1:
            add("context_shift", .40,
                replace(state, context=(c + 1) % len(state.geometry)))
        # A question is an embodied, costly action that alters subsequent
        # sampling affordances; it is NOT assigned a semantic truth value.
        if state.resource >= 1:
            add("interrogate", .12 + (.24 if d.active_conflict else 0.),
                replace(state, resource=state.resource - 1,
                        question=1 - state.question))

            for k, sight in enumerate(self.world.probes[c][state.question]):
                newer = d.observe(sight.agrees, self.witness_cap)
                after = replace(
                    state, resource=state.resource - 1,
                    distinctions=_replace_at(state.distinctions, c, newer),
                    last=(c, state.question, k))
                add("encounter:" + str(k),
                    .55 * (1.25 if state.buds[c] else 1.) * sight.probability,
                    after)
        # Evidence ageing is allowed while witness counts remain. Raw
        # witness records live in the external append-only event trace.
        for j, cell in enumerate(state.distinctions):
            faded = cell.fade()
            if faded != cell:
                add("dissipation:" + str(j), .18,
                    replace(state, distinctions=_replace_at(
                        state.distinctions, j, faded)))

        defect = self.geometric_defect(state)
        if defect is not None and state.resource >= 1:
            alternate = self.geometric_defect(state, 1 - state.geometry[c])
            gain = defect - alternate
            if gain > .05:
                add("transport_repair", .9 * gain / (1. + gain),
                    replace(state, resource=state.resource - 1,
                            geometry=_replace_at(
                                state.geometry, c, 1 - state.geometry[c])))

        # The organism cannot bud merely because a scalar surprise peaked.
        # Distinct response witnesses AND presently unresolved geometric
        # mismatch AND the structural material and resource must coexist.
        if (not state.buds[c] and
                min(d.seen_same, d.seen_different,
                    d.live_same, d.live_different) >= 2 and
                defect is not None and defect > .10 and
                state.resource >= 2 and state.material >= 2):
            add("worldlet_bud", .17 + .20 * min(1., defect),
                replace(state, resource=state.resource - 2,
                        material=state.material - 2,
                        buds=_replace_at(state.buds, c, True)))

        # Pruning salvages ONE of the two paid units of structural material;
        # it does not erase witnessed observations or replenish energy.
        for j, planted in enumerate(state.buds):
            if planted and (state.resource <= 1 or
                            not state.distinctions[j].active_conflict):
                add("worldlet_prune:" + str(j), .18,
                    replace(state, material=min(self.material_cap, state.material + 1),
                            buds=_replace_at(state.buds, j, False)))
        return tuple(out)

    def operator_action(self, state, family, observable):
        """Conditional generator on a scalar TEST observable, not a utility.

        Families select a named event or colon-delimited family of events;
        the observed scalar is only an audit probe of the CTMC dynamics.
        """
        if not self.viable(state):
            raise ValueError("invalid operator state")
        initial = observable(state)
        return math.fsum(
            ch.rate * (observable(ch.destination) - initial)
            for ch in self.channels(state)
            if ch.name == family or ch.name.startswith(family + ":")
        )

    def operator_bracket(self, state, first, second, observable):
        """The CTMC generator commutator [L_first, L_second] f at state.

        Interacting channels need NOT commute. This is a classical causal
        order-sensitivity diagnostic, not quantum noncommutativity.
        """
        a = self.operator_action(
            state, first,
            lambda s: self.operator_action(s, second, observable))
        b = self.operator_action(
            state, second,
            lambda s: self.operator_action(s, first, observable))
        return a - b

    def row(self, state, reverse=False):
        """Off-diagonal rate Q(s,t); diagonal = -sum of all off-diagonal.

        Group duplicate destinations because distinct encounter channels may
        induce the same state. The result is independent of channel listing.
        """
        channels = self.channels(state)
        if reverse:
            channels = tuple(reversed(channels))
        rates = {}
        for ch in channels:
            rates[ch.destination] = rates.get(ch.destination, 0.) + ch.rate
        return rates

    def diagonal(self, state):
        return -math.fsum(self.row(state).values())

    def total_rate_bound(self):
        """Provable upper bound on exit rates for ALL viable states."""
        n = len(self.world.contexts)
        return (1. + .70 + .23 + .16 + .10 * n + .40 +
                .36 + .55 * 1.25 + .18 * n + .90 +
                .37 + .18 * n)

    def uniformization_step(self, distribution):
        """One stochastic step of I + Q / Lambda. No module-stage order."""
        lam = self.total_rate_bound()
        output = {}
        for state, p in distribution.items():
            if p < 0 or not math.isfinite(p):
                raise ValueError("invalid input probability")
            if not self.viable(state):
                raise ValueError("unviable distribution state")
            rates = self.row(state)
            used = math.fsum(rates.values())
            if used > lam + 1e-12:
                raise AssertionError("uniformization bound violated")
            output[state] = output.get(state, 0.) + p * (1. - used / lam)
            for target, rate in rates.items():
                output[target] = output.get(target, 0.) + p * rate / lam
        return output

    def distribution(self, initial, duration, tolerance=1e-10,
                     max_terms=72, max_states=50000):
        """Small finite-support diagnostic for exp(t Q) by uniformization.

        The real tail is returned, NOT silently renormalized. This routine
        intentionally stops if finite-state expansion becomes too large.
        """
        if not self.viable(initial):
            raise ValueError("invalid start state")
        if duration < 0 or not math.isfinite(duration) or duration > 2.:
            raise ValueError("diagnostic duration must be in [0,2]")
        if not 0 < tolerance < .01:
            raise ValueError("invalid Poisson tail tolerance")
        lam = self.total_rate_bound()
        omega = lam * duration
        weight = math.exp(-omega)
        cumulative = 0.
        iterate = {initial: 1.}
        result = {}
        for k in range(max_terms + 1):
            for state, p in iterate.items():
                result[state] = result.get(state, 0.) + weight * p
            cumulative += weight
            if 1. - cumulative <= tolerance:
                break
            if k == max_terms:
                raise ArithmeticError("unresolved uniformization tail")
            iterate = self.uniformization_step(iterate)
            if len(iterate) > max_states:
                raise RuntimeError("finite-state expansion exceeded limit")
            weight *= omega / (k + 1)
        return result, max(0., 1. - cumulative)

    def trajectory(self, initial, horizon, seed=0, max_events=500):
        """One Gillespie realization. Discrete event ORDER is a sample,
        while rates all come from the joint process, not a staged pipeline.
        """
        if not self.viable(initial) or not math.isfinite(horizon) or horizon < 0:
            raise ValueError("invalid trajectory")
        rng = random.Random(seed)
        t = 0.
        state = initial
        trace = []
        for _ in range(max_events):
            choices = self.channels(state)
            rate = math.fsum(ch.rate for ch in choices)
            if rate <= 0:
                break
            waiting = rng.expovariate(rate)
            if t + waiting > horizon:
                break
            threshold = rng.random() * rate
            running = 0.
            selected = choices[-1]
            for candidate in choices:
                running += candidate.rate
                if running >= threshold:
                    selected = candidate
                    break
            t += waiting
            state = selected.destination
            trace.append((t, selected.name, state))
        return state, tuple(trace)


def example_world(source_chart=I4, target_chart=I4):
    """Observable scenario pairs; no true symmetry label is given to a learner.

    Two contexts share a nominal transformation, yet their observed
    responses need not share the same relation. The two probes sample
    different frequency menus. This is an illustrative controlled world.
    """
    a = (1., 2., 3., 4.)
    ap = matvec(P, a)
    b = (2., -1., .25, 3.)
    bp = matvec(P, b)
    return EncounterWorld(
        ("sensor-context-0", "sensor-context-1"),
        (
            (
                (EncounterView("six", "nine", a, ap, .85),
                 EncounterView("six", "six", a, a, .15)),
                (EncounterView("six", "nine", b, bp, .55),
                 EncounterView("six", "six", b, b, .45)),
            ),
            (
                (EncounterView("point-A", "point-A", a, a, .85),
                 EncounterView("point-A", "point-B", a, ap, .15)),
                (EncounterView("point-A", "point-A", b, b, .55),
                 EncounterView("point-A", "point-B", b, bp, .45)),
            ),
        ),
        source_chart=source_chart, target_chart=target_chart,
    )


def selfcheck():
    """Finite mathematical-contract smoke; NOT a held-out performance result."""
    model = JointGenesis(example_world())
    start = Organism.seed(model.world)
    assert model.viable(start)
    channels = model.channels(start)
    assert channels and all(ch.rate > 0 and model.viable(ch.destination)
                            for ch in channels)
    assert model.row(start) == model.row(start, reverse=True)
    terminal, trace = model.trajectory(start, .8, seed=7)
    assert model.viable(terminal)
    assert all(model.viable(item[2]) for item in trace)
    return {
        "evidence": "contract-only",
        "base": "telos-native finite joint CTMC",
        "joint_channels_initial": sorted(set(ch.name.split(":")[0] for ch in channels)),
        "initial_rate": -model.diagonal(start),
        "example_trace_events": len(trace),
        "quantum": "diagonal Lindblad embedding exists algebraically; not built",
        "hardware_joules": None,
        "semantic_emergence_proved": False,
    }


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2, sort_keys=True))
