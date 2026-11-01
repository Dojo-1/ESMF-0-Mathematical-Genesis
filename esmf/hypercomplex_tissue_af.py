"""G0-AF: neurite-tissue ecology with variable-rank quaternionic local fibers.

A small JOINT CTMC over chemical fields, nutrient/ATP, structural material,
neurite path sections, quaternions, and the existing G0-AE cognitive process.
NOT a biological tissue model or a measured cognitive/quantum improvement.

A soma carries Mat2(R)'s neutral (2,2) form AND a separate direct sum of
Hamilton-quaternion sections (discretized to its noncommutative subgroup Q8).
Each paid neurite extension ADDS a section and four real carrier dimensions;
pruning deletes that section with an explicitly recorded [R]_transport.
The 2D voxel lattice models *extracellular substrate*, not cognition.
"""
from dataclasses import dataclass, replace
import json
import math
import random

from .associated22_u import neutral_square
from .geometry_k import matvec
from .orientation_seam_w import P
from .telos_joint_ae import JointGenesis, Organism, example_world


# Multiplicative subgroup Q8 of the Hamilton quaternions:
# 1, i, j, k, -1, -i, -j, -k. Its multiplication is noncommutative.
Q8_NAMES = ("1", "i", "j", "k", "-1", "-i", "-j", "-k")
DIRECTIONS_2 = ((1, 0), (0, 1), (-1, 0), (0, -1))
DIRECTIONS_3 = ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1))
DIRECTION_PHASE = (1, 2, 5, 6, 3, 7)


def q8_mul(a, b):
    if type(a) is not int or type(b) is not int or not (0 <= a < 8 and 0 <= b < 8):
        raise ValueError("Q8 elements must be integer indices 0..7")
    x, y = a % 4, b % 4
    sign = (1 if a < 4 else -1) * (1 if b < 4 else -1)
    if x == 0:
        return y + (0 if sign > 0 else 4)
    if y == 0:
        return x + (0 if sign > 0 else 4)
    if x == y:
        return 0 + (4 if sign > 0 else 0)
    positive = {(1, 2): 3, (2, 3): 1, (3, 1): 2}
    if (x, y) in positive:
        axis = positive[x, y]
    else:
        axis = positive[y, x]
        sign = -sign
    return axis + (0 if sign > 0 else 4)


def q8_inv(q):
    if type(q) is not int or not 0 <= q < 8:
        raise ValueError("invalid quaternion phase")
    return q if q % 4 == 0 else (q + 4) % 8


def q8_coords(q):
    if type(q) is not int or not 0 <= q < 8:
        raise ValueError("invalid quaternion phase")
    return tuple((1 if q < 4 else -1) * (1 if k == q % 4 else 0)
                 for k in range(4))


def phase_gap(source, target, forward=0):
    """Positive Q8 representation mismatch, separate from neutral (2,2).

    Squared Euclidean *auxiliary* metric on quaternion coefficient arrays.
    Invariant under independent left-unit frame changes with connection
    U' = g_t U g_s^-1; not by naively rotating states alone.
    """
    a = q8_coords(q8_mul(forward, source))
    b = q8_coords(target)
    return sum((x - y) ** 2 for x, y in zip(a, b))


def q8_reframe(source, target, forward, source_gauge, target_gauge):
    return (q8_mul(source_gauge, source),
            q8_mul(target_gauge, target),
            q8_mul(target_gauge, q8_mul(forward, q8_inv(source_gauge))))


@dataclass(frozen=True)
class Neuron:
    soma: tuple
    path: tuple
    sections: tuple
    neutral: tuple
    substrate: int
    atp: int
    material: int

    @property
    def tip(self):
        return self.path[-1] if self.path else self.soma

    @property
    def fiber_real_dimension(self):
        # Mat2(R) contributes 4 and each quaternionic section 4.
        # A separate block-fiber, not one homogeneous metric.
        return 4 + 4 * len(self.sections)


@dataclass(frozen=True)
class TissueState:
    cognitive: Organism
    cells: tuple
    attract: tuple
    repel: tuple
    links: tuple = ()
    scars: tuple = ()  # (neuron_index, removed_voxel, removed_Q8_section)
    nutrient_input: int = 0
    material_input: int = 0
    waste: int = 0


@dataclass(frozen=True)
class TissueEvent:
    name: str
    rate: float
    destination: TissueState
    typed_remainder: tuple = ()


class HyperNeuriteEcology:
    """Bounded coupled tissue/cognition generator.

    The G0-AE resource is a synthetic cognitive CONTROL CREDIT, distinct
    from physical ATP. Sensing/interrogating/structural cognitive edits
    additionally consume the active tissue neuron's ATP in this adapter.
    Tissue growth uses actual paid material, changes local rank, and alters
    future sensing affordances; salience conflict modulates tissue growth.
    """

    def __init__(self, *, side=4, spatial_dims=3, max_sections=8, max_field=3,
                 max_nutrient_input=16, max_material_input=16, max_scars=16):
        if (type(side) is not int or not 4 <= side <= 8 or
                type(spatial_dims) is not int or spatial_dims not in (2, 3) or
                type(max_sections) is not int or not 2 <= max_sections <= 16 or
                type(max_field) is not int or not 1 <= max_field <= 6 or
                type(max_nutrient_input) is not int or max_nutrient_input < 0 or
                type(max_material_input) is not int or max_material_input < 0 or
                type(max_scars) is not int or max_scars < 0):
            raise ValueError("invalid finite tissue capacity")
        self.core = JointGenesis(example_world())
        self.side = side
        self.spatial_dims = spatial_dims
        self.directions = DIRECTIONS_3 if spatial_dims == 3 else DIRECTIONS_2
        self.max_sections = max_sections
        self.max_field = max_field
        self.max_nutrient_input = max_nutrient_input
        self.max_material_input = max_material_input
        self.max_scars = max_scars
        self.max_atp = 3
        self.max_substrate = 3
        self.max_material = 6
        self.orig_energy = 8    # two neurons, each substrate 2 and ATP 2
        self.orig_material = 4  # two neurons, material 2
        self.soma = ((0,) * spatial_dims,
                     (side // 2,) + (0,) * (spatial_dims - 1))

    def seed(self):
        cells = tuple(
            Neuron(pos, (), (phase,), (1., .5, -.5, 1.), 2, 2, 2)
            for pos, phase in zip(self.soma, (0, 1)))
        count = self.side ** self.spatial_dims
        return TissueState(Organism.seed(self.core.world), cells,
                           (0,) * count, (0,) * count)

    def index(self, point):
        if len(point) != self.spatial_dims or any(
                type(v) is not int or not 0 <= v < self.side for v in point):
            raise ValueError("voxel does not belong to the tissue chart")
        result = 0
        for value in point:
            result = result * self.side + value
        return result

    def site(self, index):
        if type(index) is not int or not 0 <= index < self.side ** self.spatial_dims:
            raise ValueError("voxel index outside tissue")
        values = [0] * self.spatial_dims
        for j in reversed(range(self.spatial_dims)):
            values[j] = index % self.side
            index //= self.side
        return tuple(values)

    def move(self, point, direction):
        shift = self.directions[direction]
        return tuple((x + d) % self.side for x, d in zip(point, shift))

    def touches(self, i, j, cells):
        # This does not claim a synapse: only adjacency of paid branches.
        return (any(self.move(loc, k) == cells[j].soma or loc == cells[j].soma
                    for loc in cells[i].path for k in range(len(self.directions))) or
                any(self.move(loc, k) == cells[i].soma or loc == cells[i].soma
                    for loc in cells[j].path for k in range(len(self.directions))))

    def valid(self, state):
        if not isinstance(state, TissueState) or not self.core.viable(state.cognitive):
            return False
        nvoxels = self.side ** self.spatial_dims
        if len(state.cells) != 2 or len(state.attract) != nvoxels or \
                len(state.repel) != nvoxels:
            return False
        if (type(state.nutrient_input) is not int or
                not 0 <= state.nutrient_input <= self.max_nutrient_input or
                type(state.material_input) is not int or
                not 0 <= state.material_input <= self.max_material_input or
                type(state.waste) is not int or state.waste < 0 or
                len(state.scars) > self.max_scars):
            return False
        if any(type(v) is not int or not 0 <= v <= self.max_field
               for v in state.attract + state.repel):
            return False
        for i, cell in enumerate(state.cells):
            if not isinstance(cell, Neuron) or cell.soma != self.soma[i]:
                return False
            if (len(cell.sections) != 1 + len(cell.path) or
                    len(cell.sections) > self.max_sections or
                    any(type(q) is not int or not 0 <= q < 8 for q in cell.sections)):
                return False
            if (any(not isinstance(loc, tuple) or
                    len(loc) != self.spatial_dims or
                    any(type(v) is not int or not 0 <= v < self.side
                        for v in loc) for loc in cell.path) or
                    len(set(cell.path)) != len(cell.path) or
                    cell.soma in cell.path):
                return False
            before = cell.soma
            for step in cell.path:
                if step not in (self.move(before, k)
                                for k in range(len(self.directions))):
                    return False
                before = step
            if (len(cell.neutral) != 4 or
                    any(not math.isfinite(v) for v in cell.neutral)):
                return False
            if (type(cell.atp) is not int or not 0 <= cell.atp <= self.max_atp or
                    type(cell.substrate) is not int or
                    not 0 <= cell.substrate <= self.max_substrate or
                    type(cell.material) is not int or
                    not 0 <= cell.material <= self.max_material):
                return False
        if (len(state.links) > 1 or
                any(len(link) != 2 or link[0] not in (0, 1) or
                    link[1] != 1 - link[0] or
                    not self.touches(link[0], link[1], state.cells)
                    for link in state.links)):
            return False
        for scar in state.scars:
            if (not isinstance(scar, tuple) or len(scar) != 3 or
                    type(scar[0]) is not int or scar[0] not in (0, 1) or
                    not isinstance(scar[1], tuple) or
                    len(scar[1]) != self.spatial_dims or
                    any(type(v) is not int or not 0 <= v < self.side
                        for v in scar[1]) or
                    type(scar[2]) is not int or not 0 <= scar[2] < 8):
                return False
        remaining_energy = (
            sum(c.atp + c.substrate for c in state.cells)
            + sum(state.attract) + sum(state.repel) + state.waste)
        remaining_material = (
            sum(c.material + len(c.path) for c in state.cells)
            + len(state.links))
        return (remaining_energy == self.orig_energy + state.nutrient_input and
                remaining_material == self.orig_material + state.material_input)

    def channels(self, state):
        if not self.valid(state):
            raise ValueError("invalid joint tissue/cognitive state")
        out = []

        def add(name, rate, dest, remainder=()):
            if not math.isfinite(rate) or rate < 0:
                raise ArithmeticError("bad channel hazard")
            if rate and dest != state:
                if not self.valid(dest):
                    raise AssertionError("transition broke closed ledgers or carrier")
                out.append(TissueEvent(name, rate, dest, remainder))

        def cell_change(i, neuron, **kw):
            cells = list(state.cells)
            cells[i] = neuron
            return replace(state, cells=tuple(cells), **kw)

        c = state.cognitive.context
        c_active = state.cells[c % len(state.cells)]
        d = state.cognitive.distinctions[c]
        conflicted = min(d.live_same, d.live_different) >= 2

        # Cognitive hypotheses and tissue events are simultaneously eligible.
        # Their causal coupling is via real ATP, structural extent and contact.
        for edge in self.core.channels(state.cognitive):
            sensor = (edge.name.startswith("encounter:") or
                      edge.name in ("interrogate", "transport_repair", "worldlet_bud"))
            if edge.name == "worldlet_bud" and not c_active.path:
                continue
            if sensor:
                if c_active.atp < 1:
                    continue
                new_cell = replace(c_active, atp=c_active.atp - 1)
                dest = cell_change(c % len(state.cells), new_cell,
                                   cognitive=edge.destination, waste=state.waste + 1)
                # A physical neurite changes sensor affordances; a virtual
                # cognitive bud is not confused with a material branch.
                rate = edge.rate * (1. + .10 * len(c_active.path)
                                    + .15 * len(state.links))
            else:
                dest = replace(state, cognitive=edge.destination)
                rate = edge.rate
            add("cognitive/" + edge.name, rate, dest)

        for i, neuron in enumerate(state.cells):
            position = self.index(neuron.soma)
            tip = self.index(neuron.tip)
            a_here = state.attract[tip]
            r_here = state.repel[tip]

            if (neuron.substrate < self.max_substrate and
                    state.nutrient_input < self.max_nutrient_input):
                add(f"nutrient_supply:{i}", .30,
                    cell_change(i, replace(neuron, substrate=neuron.substrate + 1),
                                nutrient_input=state.nutrient_input + 1))
            if (neuron.material < self.max_material and
                    state.material_input < self.max_material_input):
                add(f"material_supply:{i}", .21,
                    cell_change(i, replace(neuron, material=neuron.material + 1),
                                material_input=state.material_input + 1))
            if neuron.substrate and neuron.atp < self.max_atp:
                add(f"metabolic_conversion:{i}", .35,
                    cell_change(i, replace(neuron, substrate=neuron.substrate - 1,
                                           atp=neuron.atp + 1)))
            if neuron.atp:
                add(f"maintenance:{i}", .10 + .035 * len(neuron.path)
                    + .05 * len(state.links),
                    cell_change(i, replace(neuron, atp=neuron.atp - 1),
                                waste=state.waste + 1))
            for name, f in (("attract", state.attract), ("repel", state.repel)):
                if neuron.substrate and f[position] < self.max_field:
                    field = list(f)
                    field[position] += 1
                    new = replace(neuron, substrate=neuron.substrate - 1)
                    kw = {name: tuple(field)}
                    add(f"secrete_{name}:{i}", .17 if name == "attract" else .09,
                        cell_change(i, new, **kw))
            if neuron.atp and (a_here or r_here or conflicted):
                # Two different external/local cues select two generators.
                factor = 1 if a_here > r_here else 2
                updated = list(neuron.sections)
                updated[-1] = q8_mul(updated[-1], factor)
                add(f"phase_pulse:{i}", .18 + .03 * (a_here + r_here),
                    cell_change(i, replace(neuron, atp=neuron.atp - 1,
                                           sections=tuple(updated)),
                                waste=state.waste + 1))

            if (neuron.atp and neuron.material and
                    len(neuron.sections) < self.max_sections):
                tip_phase = neuron.sections[-1]
                compass = (tip_phase % 4 + 2 * (tip_phase // 4)) % len(self.directions)
                for direction in range(len(self.directions)):
                    dest_pos = self.move(neuron.tip, direction)
                    if dest_pos == neuron.soma or dest_pos in neuron.path:
                        continue
                    target = self.index(dest_pos)
                    attractive = max(
                        0, state.attract[target] - state.repel[target])
                    propensity = (.035 + .07 * attractive +
                                  (.06 if direction == compass else 0.) +
                                  (.05 if conflicted else 0.))
                    sections = neuron.sections + (
                        q8_mul(tip_phase, DIRECTION_PHASE[direction]),)
                    seam = (neuron.tip[0], dest_pos[0])
                    crosses = seam in ((0, self.side - 1),
                                       (self.side - 1, 0))
                    neutral = (
                        tuple(matvec(P, neuron.neutral)) if crosses else neuron.neutral)
                    new = replace(neuron, path=neuron.path + (dest_pos,),
                                  sections=sections, neutral=neutral,
                                  atp=neuron.atp - 1,
                                  material=neuron.material - 1)
                    add(f"neurite_extend:{i}:{direction}", propensity,
                        cell_change(i, new, waste=state.waste + 1))

            if neuron.path and len(state.scars) < self.max_scars and \
                    neuron.material < self.max_material and not state.links:
                before = neuron.path[-2] if len(neuron.path) > 1 else neuron.soma
                seam = (neuron.path[-1][0], before[0])
                crosses = seam in ((0, self.side - 1),
                                   (self.side - 1, 0))
                neutral = (
                    tuple(matvec(P, neuron.neutral)) if crosses else neuron.neutral)
                scar = (i, neuron.path[-1], neuron.sections[-1])
                new = replace(neuron, path=neuron.path[:-1],
                              sections=neuron.sections[:-1], neutral=neutral,
                              material=neuron.material + 1)
                add(f"neurite_prune:{i}", .05 + (.18 if neuron.atp == 0 else 0.),
                    cell_change(i, new, scars=state.scars + (scar,)),
                    ("[R]_transport", scar))

            if neuron.path and neuron.atp and neuron.material and \
                    not state.links and self.touches(i, 1-i, state.cells):
                target_cell = state.cells[1-i]
                mismatch = phase_gap(neuron.sections[-1],
                                     target_cell.sections[0])
                new = replace(neuron, atp=neuron.atp - 1,
                              material=neuron.material - 1)
                add(f"contact_birth:{i}", .30 / (1. + mismatch),
                    cell_change(i, new, links=((i, 1-i),),
                                waste=state.waste + 1))
            if state.links and state.links[0][0] == i and neuron.material < self.max_material:
                # Explicit relation withdrawal; material is recycled
                # rather than created. Provenance remains in event trace.
                new = replace(neuron, material=neuron.material + 1)
                add(f"contact_release:{i}", .04,
                    cell_change(i, new, links=()),
                    ("[R]_attr", state.links[0]))

        for field_name in ("attract", "repel"):
            field = getattr(state, field_name)
            for ix in range(self.side ** self.spatial_dims):
                    pos = self.site(ix)
                    if not field[ix]:
                        continue
                    for direction in range(len(self.directions)):
                        target = self.index(self.move(pos, direction))
                        if field[target] >= self.max_field:
                            continue
                        values = list(field)
                        values[ix] -= 1
                        values[target] += 1
                        add(f"diffuse_{field_name}:{ix}:{direction}",
                            .065 * field[ix],
                            replace(state, **{field_name: tuple(values)}))
                    values = list(field)
                    values[ix] -= 1
                    add(f"decay_{field_name}:{ix}", .035 * field[ix],
                        replace(state, **{field_name: tuple(values)},
                                waste=state.waste + 1))
        return tuple(out)

    def generator_action(self, state, prefix, observable):
        """Named part of a *joint* generator on a test observable."""
        if not self.valid(state) or not prefix:
            raise ValueError("invalid observation or generator prefix")
        original = observable(state)
        return math.fsum(
            ch.rate * (observable(ch.destination) - original)
            for ch in self.channels(state) if ch.name.startswith(prefix))

    def generator_bracket(self, state, first, second, observable):
        """Classical causal noncommutativity, not quantum coherence."""
        return (self.generator_action(
            state, first, lambda s: self.generator_action(s, second, observable))
                - self.generator_action(
                    state, second, lambda s: self.generator_action(s, first, observable)))

    def row(self, state, reverse=False):
        channels = self.channels(state)
        if reverse:
            channels = reversed(channels)
        rates = {}
        for ch in channels:
            rates[ch.destination] = rates.get(ch.destination, 0.) + ch.rate
        return rates

    def diagonal(self, state):
        return -math.fsum(self.row(state).values())

    def trajectory(self, start, horizon, seed=0, max_events=500):
        if (not self.valid(start) or not math.isfinite(horizon) or
                horizon < 0 or type(max_events) is not int or max_events < 1):
            raise ValueError("invalid simulation specification")
        rng = random.Random(seed)
        t = 0.
        current = start
        history = []
        for _ in range(max_events):
            events = self.channels(current)
            total = math.fsum(e.rate for e in events)
            if total <= 0:
                break
            waiting = rng.expovariate(total)
            if t + waiting > horizon:
                break
            threshold = rng.random() * total
            running = 0.
            selected = events[-1]
            for event in events:
                running += event.rate
                if running >= threshold:
                    selected = event
                    break
            t += waiting
            current = selected.destination
            history.append((t, selected.name, selected.typed_remainder))
        else:
            raise RuntimeError("max_events exhausted; no silent path truncation")
        return current, tuple(history)

    def diagnostics(self, state):
        if not self.valid(state):
            raise ValueError("invalid state")
        return {
            "carrier": "Mat2(R) neutral (2,2) plus variable Q8-subgroup H sections",
            "real_fiber_dimensions": tuple(c.fiber_real_dimension for c in state.cells),
            "total_neurite_sections": sum(len(c.path) for c in state.cells),
            "links": len(state.links),
            "neurite_paths": tuple(c.path for c in state.cells),
            "typed_prune_remainders": len(state.scars),
            "chemical_packets": sum(state.attract) + sum(state.repel),
            "positive_resource_pools": tuple(
                (c.substrate, c.atp, c.material) for c in state.cells),
            "energy_balance_residual": (
                sum(c.substrate + c.atp for c in state.cells)
                + sum(state.attract) + sum(state.repel) + state.waste
                - self.orig_energy - state.nutrient_input),
            "material_balance_residual": (
                sum(c.material + len(c.path) for c in state.cells)
                + len(state.links) - self.orig_material - state.material_input),
            "neutral_signature_invariants": tuple(
                neutral_square(c.neutral) for c in state.cells),
            "cognitive_relations": tuple(
                (d.seen_same, d.seen_different) for d
                in state.cognitive.distinctions),
            "evidence": "contract-only",
            "actual_hardware_joules": None,
        }


def selfcheck():
    """Deterministic smoke, NOT a held-out biological or cognitive result."""
    obj = HyperNeuriteEcology()
    start = obj.seed()
    assert obj.valid(start)
    assert q8_mul(1, 2) == 3 and q8_mul(2, 1) == 7
    assert q8_mul(q8_mul(1, 2), 3) == q8_mul(1, q8_mul(2, 3))
    assert all(e.rate > 0 and obj.valid(e.destination)
               for e in obj.channels(start))
    final, trace = obj.trajectory(start, .4, seed=11)
    assert obj.valid(final)
    return {"contract": "G0-AF",
            "initial": obj.diagnostics(start),
            "after": obj.diagnostics(final),
            "trace_events": len(trace),
            "claims": "no semantic emergence, tissue validity or quantum advantage"}


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2, sort_keys=True))
