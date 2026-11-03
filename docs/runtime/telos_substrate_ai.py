"""G0-AI: weakly differentiated, locally rewriting synthetic tissue.

This is a bounded discrete-time experimental substrate, not a brain, a CTMC,
quantum computation, or a demonstrated instance of emergent cognition.
No cell is assigned a question, class, astrocyte role, or organ name.
Fast local activation, slower local regulatory dispositions, optional contact
morphogenesis and resource accounting are coupled through observable histories.
Supervised error trains only the external readout, never a designated organ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, tanh, sqrt
from random import Random
from typing import Literal

Mode = Literal['adaptive', 'fixed', 'plastic_no_growth', 'shuffled']


def clip(v: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, v))


def sigmoid(z: float) -> float:
    return 1 / (1 + exp(-clip(z, -30, 30)))


@dataclass
class LocalUnit:
    xyz: tuple[float, float, float]
    sensor: tuple[float, ...]
    activity: float = 0.0
    fast_trace: float = 0.0
    slow_trace: float = 0.0
    novelty: float = 0.0
    regulome: float = 0.0
    rate: float = 0.06
    gain: float = 1.0
    rule_mix: tuple[float, float, float] = (1/3, 1/3, 1/3)
    atp: int = 2100
    material: int = 5


@dataclass
class Contact:
    source: int
    target: int
    weight: float
    age: int = 0
    fitness: float = 0.0


@dataclass(frozen=True)
class Remainder:
    kind: str
    source: int
    target: int
    value: float
    tick: int


@dataclass
class Substrate:
    """Typed finite state with explicit metabolic and structural accounts."""

    units: list[LocalUnit]
    contacts: dict[tuple[int, int], Contact]
    tick: int = 0
    waste: int = 0
    material_input: int = 0
    energy_input: int = 0
    remainders: list[Remainder] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    former_contacts: int = 0
    growth_events: int = 0
    pruning_events: int = 0
    attempts: int = 0
    initial_energy: int = 0
    initial_material: int = 0

    def ledger(self) -> tuple[int, int]:
        energy = (sum(u.atp for u in self.units) + self.waste -
                  self.initial_energy - self.energy_input)
        material = (sum(u.material for u in self.units) +
                    len(self.contacts) - self.initial_material - self.material_input)
        return energy, material


class TelosSubstrate:
    """Same undifferentiated initial conditions and budgets in every arm.

    Different modes ablate ONLY spontaneous correlation-based morphology and
    history-dependent regulatory coefficients. An ordinary conditional-memory
    control is evaluated separately, with its incomparable cost made explicit.
    """

    def __init__(self, *, seed: int, mode: Mode = 'adaptive', n: int = 20,
                 channels: int = 4, max_edges: int = 72,
                 refresh: int = 16, initial_atp: int = 2100,
                 max_remainders: int = 3000):
        if mode not in ('adaptive', 'fixed', 'plastic_no_growth', 'shuffled'):
            raise ValueError('unknown ablation')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        if any(type(v) is not int for v in (n, channels, max_edges, refresh, initial_atp, max_remainders)):
            raise ValueError('integer capacities required')
        if not (8 <= n <= 48 and channels == 4 and n <= max_edges <= n*5
                and 1 <= refresh <= 100 and initial_atp >= 1
                and max_remainders >= 0):
            raise ValueError('invalid capacities')
        self.mode, self.n, self.channels = mode, n, channels
        self.max_edges, self.refresh, self.max_remainders = max_edges, refresh, max_remainders
        rng = Random(seed)
        units = []
        for _ in range(n):
            xyz = tuple(rng.uniform(-1, 1) for _ in range(3))
            # No hardwired sensor organ or named question: each weakly exposed
            # site gets an exchangeable random projection of four raw cues.
            sensor = tuple(rng.uniform(-0.32, 0.32) for _ in range(channels))
            units.append(LocalUnit(xyz=xyz, sensor=sensor, atp=initial_atp))
        candidates = []
        for source in range(n):
            for target in range(n):
                if source == target:
                    continue
                dist = self.distance(units[source].xyz, units[target].xyz)
                if dist <= 1.25:
                    candidates.append((source, target, dist))
        # Geometry constrains possible connections, not the eventual role
        # of any compartment. A small symmetric seed scaffold is shared.
        contacts = {}
        for target in range(n):
            neighbors = sorted((item for item in candidates if item[1] == target),
                               key=lambda v: (v[2], v[0]))[:2]
            for source, _, _ in neighbors:
                if (source, target) not in contacts and units[source].material > 0:
                    contacts[source, target] = Contact(source, target, rng.uniform(-.17, .17))
                    units[source].material -= 1
        self.candidates = tuple(sorted((s, t) for s, t, _ in candidates))
        self.s = Substrate(units=units, contacts=contacts,
                           initial_energy=n * initial_atp,
                           initial_material=n * 5)
        assert self.s.ledger() == (0, 0)
        self._rng_shuffle = Random(seed + 982451653)
        self._rng_morph = Random(seed + 433494437)
        self.readout = [0.0] * (n + 1)
        self.pending: tuple[float, tuple[float, ...]] | None = None
        self.readout_lr = .12
        # Real pair-valued local history, not a predefined cell subtype.
        self.corr = {key: 0.0 for key in self.candidates}
        self._baseline = [0.18] * n
        self._exposure = [0.0] * n

    @staticmethod
    def distance(a, b):
        return sqrt(sum((x-y)**2 for x, y in zip(a, b)))

    def _step_cells(self, cues: tuple[int, ...]):
        if len(cues) != self.channels or any(v not in (-1, 1) for v in cues):
            raise ValueError('untyped or unsupported external stimulus')
        units = self.s.units
        if any(u.atp < 1 for u in units):
            raise RuntimeError('resource exhaustion: atomic event rejected')
        old = [u.activity for u in units]
        recurrent = [0.0] * self.n
        for edge in self.s.contacts.values():
            recurrent[edge.target] += edge.weight * old[edge.source]
        new = []
        for i, u in enumerate(units):
            current = tanh(u.gain * (sum(a*b for a, b in zip(u.sensor, cues)) + recurrent[i]))
            new.append(current)
            delta = abs(current - u.fast_trace)
            u.novelty = 0.96*u.novelty + 0.04*delta
            u.fast_trace = .72*u.fast_trace + .28*current
            if self.s.tick % 5 == 0:
                u.slow_trace = .985*u.slow_trace + .015*current
            if self.mode == 'adaptive' or self.mode == 'plastic_no_growth' or self.mode == 'shuffled':
                # History rewrites each local *learning disposition* and gain,
                # rather than prescribing cell types or operator schedules.
                # Regulatory *weights* themselves evolve by local history,
                # using three available physical observables. This is still
                # a fixed primitive vocabulary, NOT unrestricted rule genesis.
                excess = delta - self._baseline[i]
                neighborhood = recurrent[i] / (1 + abs(recurrent[i]))
                disagreement = abs(current - neighborhood) - self._baseline[i]
                resource_pressure = min(0., (u.atp/2100) - .3)
                cues_for_regulation = (excess, disagreement, resource_pressure)
                # Local predictability improvement supplies a provisional
                # credit signal; no supervised task label enters this rule.
                progress = self._exposure[i] - delta
                self._exposure[i] = .9*self._exposure[i] + .1*delta
                gain = tuple(max(.00001, w*exp(clip(.15*progress*z, -.2, .2)))
                             for w,z in zip(u.rule_mix, cues_for_regulation))
                normalizer = sum(gain)
                u.rule_mix = tuple(w/normalizer for w in gain)
                drift = sum(w*z for w,z in zip(u.rule_mix, cues_for_regulation))
                u.regulome = clip(.97*u.regulome + .03*drift, -.75, .75)
                u.rate = clip(u.rate + .012*u.regulome, .009, .14)
                u.gain = clip(u.gain + .004*(.34 - abs(current)), .58, 1.65)
                self._baseline[i] = .996*self._baseline[i] + .004*delta
            u.activity = current
            u.atp -= 1
            self.s.waste += 1
        for key in self.candidates:
            source, target = key
            # A local, lagged relation; not a supervised label oracle.
            evidence = new[target] * old[source]
            self.corr[key] = .92*self.corr[key] + .08*evidence
        for key, edge in self.s.contacts.items():
            source, target = key
            edge.age += 1
            relation = self.corr[key]
            edge.fitness = .96*edge.fitness + .04*abs(relation)
            if self.mode != 'fixed':
                lr = units[target].rate
                edge.weight = clip(edge.weight + lr*.07*(relation -
                        .5*new[target]**2*edge.weight), -.65, .65)
        return tuple(new)

    def _reorganize(self):
        if self.mode in ('fixed', 'plastic_no_growth'):
            return
        # The substrate supplies *opportunities*, not a clocked instruction
        # to grow. Local novelty and endogenous regulatory dispositions
        # determine the event hazard; the actual target is coactivity-led.
        drive = sum(max(0., u.novelty + u.regulome) for u in self.s.units)/self.n
        chance = clip((.35 + 3*drive)/self.refresh, .0005, .28)
        if self._rng_morph.random() >= chance:
            return
        s = self.s
        available = [(src, target) for src, target in self.candidates
                     if (src, target) not in s.contacts and s.units[src].material > 0]
        if not available:
            return
        # Local activity/history biases the *opportunity*, but not a desired
        # anatomical target nor an externally assigned cell role.
        if self.mode == 'shuffled':
            order = list(available)
            self._rng_shuffle.shuffle(order)
        else:
            order = sorted(available, key=lambda p:(-abs(self.corr[p])*s.units[p[1]].rate,
                                                    p[0], p[1]))
        src, target = order[0]
        s.attempts += 1
        # Dynamic morphology cannot grow from evidence alone without local
        # resource: the same opportunity accounting is used in shuffled arm.
        if s.units[src].atp < 2 or len(s.contacts) >= self.max_edges:
            if len(s.contacts) < self.max_edges:
                return
            if len(s.remainders) >= self.max_remainders:
                raise RuntimeError('typed loss journal full: no silent pruning')
            poorest = min(s.contacts, key=lambda p:(s.contacts[p].fitness, p))
            old = s.contacts.pop(poorest)
            s.units[old.source].material += 1
            s.pruning_events += 1
            s.former_contacts += 1
            s.remainders.append(Remainder('[R]_pruned_contact', old.source,
                                          old.target, old.weight, s.tick))
            s.events.append({'tick':s.tick, 'type':'pruning',
                             'source':old.source, 'target':old.target,
                             'weight_lost':round(old.weight, 6)})
        if s.units[src].material < 1 or s.units[src].atp < 2:
            return
        s.units[src].atp -= 2
        s.waste += 2
        s.units[src].material -= 1
        relation = self.corr[src, target]
        s.contacts[src, target] = Contact(src, target,
                                    .08 if relation >= 0 else -.08)
        s.growth_events += 1
        s.events.append({'tick':s.tick,'type':'growth', 'source':src,
                         'target':target, 'relation':round(relation, 6)})

    def predict(self, cues: tuple[int, ...]) -> float:
        if self.pending is not None:
            raise RuntimeError('must observe previous prediction before next cue')
        if self.s.ledger() != (0, 0):
            raise AssertionError('broken energy/material conservation')
        if len(cues) != self.channels or any(v not in (-1, 1) for v in cues):
            raise ValueError('invalid cue cannot advance the clock')
        if any(u.atp < 1 for u in self.s.units):
            raise RuntimeError('resource exhaustion: atomic event rejected')
        self.s.tick += 1
        features = self._step_cells(cues)
        self._reorganize()
        if self.s.ledger() != (0, 0):
            raise AssertionError('non-conserving state transition')
        inp = (1.0,) + features
        p = sigmoid(sum(w*x for w, x in zip(self.readout, inp)))
        self.pending = (p, inp)
        return p

    def observe(self, changed: bool):
        if type(changed) is not bool or self.pending is None:
            raise ValueError('an externally witnessed boolean outcome is required')
        p, x = self.pending
        err = float(changed) - p
        for i in range(len(self.readout)):
            self.readout[i] = clip(self.readout[i] + self.readout_lr*err*x[i], -4, 4)
        self.pending = None

    def integrity(self) -> dict:
        return {'ledger': self.s.ledger(), 'cells':self.n,
                'contacts':len(self.s.contacts),
                'growth':self.s.growth_events,
                'pruning':self.s.pruning_events,
                'mean_rate':sum(u.rate for u in self.s.units)/self.n,
                'regulator_spread':max(u.rate for u in self.s.units)-min(u.rate for u in self.s.units),
                'energy_spent':self.s.waste,
                'material_retained':sum(u.material for u in self.s.units),
                'mean_gain':sum(u.gain for u in self.s.units)/self.n,
                'rule_weight_spread':sum(max(u.rule_mix)-min(u.rule_mix) for u in self.s.units)/self.n,
                'remainders':len(self.s.remainders)}
