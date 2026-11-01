"""G0-AG: continuous quaternionic neurite transport with local glial support.

Bounded-channel synthetic pure-jump Markov process with continuous S^3
state (NOT a mathematically finite-state CTMC). Extends G0-AF's paid
3-D/2-D neurite path, Q8, Mat2 neutral carrier or historical measurements.

The new continuous S^3 unit-quaternion connection is a DISTINCT fiber
regime, not a quantum state. One spatial glial support compartment has
substrate, transferable proxy, calcium-like gating and an explicitly paid
tripartite contact. Neither neuron nor glial costs are silently refunded.

The G0-AE observational menu remains a synthetic two-context laboratory.
This is an executable mathematical contract, not biological validity,
semantic emergence or demonstrated hypercomplex/hyperdim advantage.
"""

from dataclasses import dataclass, replace
import json
import math
import random

from .hypercomplex_tissue_af import (
    HyperNeuriteEcology, TissueState, q8_coords,
)


ONE = (1., 0., 0., 0.)


def qmult(a, b):
    """Hamilton multiplication on R^4; associative and noncommutative."""
    w, x, y, z = a
    v, p, q, r = b
    return (w * v - x * p - y * q - z * r,
            w * p + x * v + y * r - z * q,
            w * q - x * r + y * v + z * p,
            w * r + x * q - y * p + z * v)


def qconj(a):
    return (a[0], -a[1], -a[2], -a[3])


def qlength2(a):
    return math.fsum(v * v for v in a)


def qnorm(a):
    """Normalize a nonzero finite quaternion without conflating +/- states."""
    if len(a) != 4 or any(not math.isfinite(x) for x in a):
        raise ValueError("four finite quaternion coordinates required")
    magnitude = math.hypot(*a)
    if magnitude <= 1e-12:
        raise ValueError("cannot normalize a null quaternion")
    return tuple(x / magnitude for x in a)


def qerror(source, target, forward=ONE):
    """Positive auxiliary mismatch; not metabolism or neutral energy."""
    return math.fsum(
        (x - y) ** 2 for x, y in zip(qmult(forward, source), target)
    )


def qreframe(source, target, forward, at_source, at_target):
    """Independent left-unit gauges: U' = g_t U g_s^{-1}."""
    return (qmult(at_source, source),
            qmult(at_target, target),
            qmult(qmult(at_target, forward), qconj(at_source)))


def qinterpolate(a, b, fraction):
    """Gauge-equivariant normalized blend on S^3, excluding antipodal zeros."""
    if not math.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError("fraction must be in [0,1]")
    return qnorm(tuple((1. - fraction) * x + fraction * y
                       for x, y in zip(a, b)))


def qtransport_chain(forward, source):
    """Typed directed composition U_n ... U_1 q; order is essential."""
    current = source
    for connection in forward:
        current = qmult(connection, current)
    return current


def qchain_discrepancy(fiber):
    """Auxiliary endpoint mismatch; zero does not certify semantics."""
    return qerror(qtransport_chain(fiber.forward, fiber.phases[0]),
                  fiber.phases[-1])


def observation_quaternion(observed_coordinates, observer_frame=ONE):
    """A-typed observation adapter: canonical R4 sensor -> local H phase.

    Identifying four observed real numbers with quaternion coefficients is a
    DECLARED model assumption, not a semantic/theorem transport from Mat2(R).
    The learned fiber's observer_frame changes under its local gauge.
    Zero or tiny input is unrepresentable: return None, never fake a phase.
    """
    if (len(observed_coordinates) != 4 or
            any(not math.isfinite(x) for x in observed_coordinates) or
            len(observer_frame) != 4 or
            any(not math.isfinite(x) for x in observer_frame) or
            abs(qlength2(observer_frame) - 1.) > 1e-9):
        raise ValueError("invalid typed observation or observer frame")
    magnitude = math.hypot(*observed_coordinates)
    if magnitude <= 1e-12:
        return None
    return qmult(observer_frame, tuple(x / magnitude for x in observed_coordinates))


def reframe_sections(fiber, gauges):
    """Independent local S3 gauges, including observation at the tip.

    Does not transport old provenance records: those require recorded chart
    IDs. Use this as a present-state metamorphic transformation only.
    """
    if not isinstance(fiber, ContinuousSections) or len(gauges) != len(fiber.phases):
        raise ValueError("one local gauge per active quaternion section required")
    for gauge in gauges:
        if (not isinstance(gauge, tuple) or len(gauge) != 4 or
                any(not math.isfinite(x) for x in gauge) or
                abs(qlength2(gauge) - 1.) > 1e-9):
            raise ValueError("all section gauges must be unit quaternions")
    phases = tuple(qmult(g, p) for g, p in zip(gauges, fiber.phases))
    links = tuple(
        qmult(qmult(gauges[i + 1], link), qconj(gauges[i]))
        for i, link in enumerate(fiber.forward)
    )
    return replace(fiber, phases=phases, forward=links,
                   observer_frame=qmult(gauges[-1], fiber.observer_frame))


def qdirection(delta, angle=.43):
    """Smooth 3-D directional rotation; 2-D directions embed in R^3."""
    if len(delta) not in (2, 3):
        raise ValueError("expected 2-D or 3-D spatial direction")
    xyz = delta if len(delta) == 3 else (delta[0], delta[1], 0)
    length = math.sqrt(math.fsum(v * v for v in xyz))
    if length <= 0 or not math.isfinite(angle):
        raise ValueError("nonzero direction and finite angle required")
    h = angle / 2
    return (math.cos(h), *(math.sin(h) * v / length for v in xyz))


@dataclass(frozen=True)
class ContinuousSections:
    """Actual variable-rank S^3 phases with a separate connection per segment."""

    phases: tuple
    forward: tuple
    observer_frame: tuple = ONE  # A: typed Mat2(R) observation -> local H chart

    @property
    def real_carrier_dimension(self):
        # Existing neutral Mat2(R) block + variable real quaternion carrier.
        return 4 + 4 * len(self.phases)


@dataclass(frozen=True)
class Glia:
    voxel: tuple
    substrate: int
    transferable: int
    calcium: int
    tripartite: bool = False


@dataclass(frozen=True)
class Ecology:
    tissue: TissueState
    glia: Glia
    fibers: tuple
    glial_input: int = 0
    glial_to_tissue: int = 0
    glial_waste: int = 0
    fiber_scars: tuple = ()
    repairs: tuple = ()
    phase_history: tuple = ()  # (cell, segment, old, new, observed_event)


@dataclass(frozen=True)
class Event:
    name: str
    rate: float
    destination: Ecology
    remainder: tuple = ()


class GlialQuaternionEcology:
    """Joint bounded-channel generator: glial support, continuous S3, AF tissue.

    The organism has *two* physically accountable energy compartments.
    AF's nutrient_input is the inflow to its energy subledger, including
    transfers from glia. The global ledger SUBTRACTS those internal
    transfers so that they never become new external nutrient supply.
    """

    def __init__(self, *, spatial_dims=3, side=4, max_glial_input=24,
                 max_fiber_records=32, max_support=4, **af_params):
        if (type(max_glial_input) is not int or max_glial_input < 0 or
                type(max_fiber_records) is not int or max_fiber_records < 0 or
                type(max_support) is not int or not 1 <= max_support <= 8):
            raise ValueError("invalid glial resource or provenance bounds")
        self.af = HyperNeuriteEcology(
            spatial_dims=spatial_dims, side=side, **af_params
        )
        self.max_glial_input = max_glial_input
        self.max_fiber_records = max_fiber_records
        self.max_support = max_support
        self.glial_origin = (side // 4,) + (0,) * (spatial_dims - 1)
        self.coverage = max(1, side // 4)
        self.initial_glia = 3

    def seed(self):
        substrate = self.af.seed()
        fiber = tuple(
            ContinuousSections(tuple(q8_coords(q) for q in cell.sections), ())
            for cell in substrate.cells
        )
        return Ecology(
            substrate, Glia(self.glial_origin, 2, 1, 0), fiber
        )

    def _near(self, a, b):
        return sum(min(abs(x - y), self.af.side - abs(x - y))
                   for x, y in zip(a, b)) <= self.coverage

    def _energy_balance(self, s):
        n = s.tissue
        base_energy = (sum(c.atp + c.substrate for c in n.cells)
                       + sum(n.attract) + sum(n.repel) + n.waste)
        glial_energy = (
            s.glia.substrate + s.glia.transferable + s.glial_waste
        )
        return (
            base_energy + glial_energy - self.af.orig_energy
            - self.initial_glia - n.nutrient_input
            + s.glial_to_tissue - s.glial_input
        )

    def valid(self, s):
        if not isinstance(s, Ecology) or not self.af.valid(s.tissue):
            return False
        g = s.glia
        if (not isinstance(g, Glia) or g.voxel != self.glial_origin or
                type(g.substrate) is not int or not 0 <= g.substrate <= self.max_support
                or type(g.transferable) is not int or
                not 0 <= g.transferable <= self.max_support or
                type(g.calcium) is not int or not 0 <= g.calcium <= 2 or
                type(g.tripartite) is not bool or
                (g.tripartite and not s.tissue.links) or
                len(s.fibers) != len(s.tissue.cells)):
            return False
        for x in (s.glial_input, s.glial_to_tissue, s.glial_waste):
            if type(x) is not int or x < 0:
                return False
        if (s.glial_input > self.max_glial_input or
                s.glial_to_tissue > s.tissue.nutrient_input or
                len(s.fiber_scars) > self.max_fiber_records or
                len(s.repairs) > self.max_fiber_records or
                len(s.phase_history) > self.max_fiber_records):
            return False
        for i, fiber in enumerate(s.fibers):
            if (not isinstance(fiber, ContinuousSections) or
                    len(fiber.phases) != len(s.tissue.cells[i].sections) or
                    len(fiber.forward) != len(s.tissue.cells[i].path)):
                return False
            for q in fiber.phases + fiber.forward + (fiber.observer_frame,):
                if (not isinstance(q, tuple) or len(q) != 4 or
                        any(not math.isfinite(x) for x in q) or
                        abs(qlength2(q) - 1.) > 1e-9):
                    return False
        if not all(isinstance(r, tuple) and len(r) == 4
                   for r in s.fiber_scars):
            return False
        if not all(isinstance(r, tuple) and len(r) == 5 for r in s.repairs):
            return False
        if not all(isinstance(r, tuple) and len(r) == 5
                   for r in s.phase_history):
            return False
        return self._energy_balance(s) == 0

    def _put_phase(self, s, i, fiber):
        out = list(s.fibers)
        out[i] = fiber
        return tuple(out)

    def _tissue_paid(self, tissue, i, **kwargs):
        cell = tissue.cells[i]
        if cell.atp < 1:
            raise ValueError("ATP boundary forbids plasticity")
        cells = list(tissue.cells)
        cells[i] = replace(cell, atp=cell.atp - 1)
        return replace(tissue, cells=tuple(cells),
                       waste=tissue.waste + 1, **kwargs)

    def channels(self, s):
        if not self.valid(s):
            raise ValueError("invalid coupled tissue state")
        out = []

        def add(name, rate, target, remainder=()):
            if rate < 0 or not math.isfinite(rate):
                raise ArithmeticError("bad generator rate")
            if rate and target != s:
                if not self.valid(target):
                    raise AssertionError("nonviable tissue or lost provenance")
                out.append(Event(name, rate, target, remainder))

        for e in self.af.channels(s.tissue):
            updated_glia = s.glia
            fibers = list(s.fibers)
            scars = s.fiber_scars
            typed = e.typed_remainder

            if e.name.startswith("neurite_extend:"):
                _, i, direction = e.name.split(":")
                i, direction = int(i), int(direction)
                f = fibers[i]
                delta = self.af.directions[direction]
                rotation = qdirection(delta)
                q_new = qnorm(qmult(rotation, f.phases[-1]))
                U = qnorm(qmult(q_new, qconj(f.phases[-1])))
                fibers[i] = replace(
                    f, phases=f.phases + (q_new,),
                    forward=f.forward + (U,))
            elif e.name.startswith("neurite_prune:"):
                if len(scars) >= self.max_fiber_records:
                    continue   # cannot silently discard lost phases
                i = int(e.name.split(":")[1])
                f = fibers[i]
                scar = (i, s.tissue.cells[i].tip,
                        f.phases[-1], f.forward[-1])
                fibers[i] = replace(
                    f, phases=f.phases[:-1],
                    forward=f.forward[:-1])
                scars = scars + (scar,)
                typed = ("[R]_transport_continuous", scar)
            if updated_glia.tripartite and not e.destination.links:
                updated_glia = replace(updated_glia, tripartite=False)
                typed = ("[R]_attr_glial_support", s.tissue.links)
            target = replace(
                s, tissue=e.destination, fibers=tuple(fibers),
                fiber_scars=scars, glia=updated_glia)
            rate = e.rate
            if e.name.startswith("neurite_extend:"):
                # A locally incoherent continuous transport may make new
                # anatomical extension less affordable before repair. This
                # is a state-derived stabilizer, NOT an evaluator oracle.
                neuron_i = int(e.name.split(":")[1])
                rate /= 1. + .25 * qchain_discrepancy(s.fibers[neuron_i])
            if s.glia.tripartite and e.name.startswith("contact_release:"):
                rate *= .25
            if s.glia.tripartite and e.name.startswith("cognitive/encounter:"):
                rate *= 1.20
            add("af/" + e.name, rate, target, typed)

        g = s.glia
        if g.substrate < self.max_support and s.glial_input < self.max_glial_input:
            add("glia_supply", .27,
                replace(s, glia=replace(g, substrate=g.substrate + 1),
                        glial_input=s.glial_input + 1))
        if g.substrate:
            add("glia_maintenance", .07 + (.03 if g.tripartite else 0.),
                replace(s, glia=replace(g, substrate=g.substrate - 1),
                        glial_waste=s.glial_waste + 1))
        if g.substrate and g.transferable < self.max_support:
            add("glia_convert", .39,
                replace(s, glia=replace(
                    g, substrate=g.substrate - 1,
                    transferable=g.transferable + 1)))
        # A historical last observation is not ongoing activity: salience
        # must still be present, or a current contact must exist.
        activity = (bool(s.tissue.links)
                    or any(d.live_same or d.live_different
                           for d in s.tissue.cognitive.distinctions))
        if g.substrate and g.calcium < 2 and activity:
            add("glia_calcium_pulse", .23,
                replace(s, glia=replace(
                    g, substrate=g.substrate - 1,
                    calcium=g.calcium + 1),
                    glial_waste=s.glial_waste + 1))
        if g.calcium:
            new_calcium = g.calcium - 1
            lost = ("[R]_attr_glial_support", s.tissue.links) if (
                g.tripartite and not new_calcium) else ()
            add("glia_calcium_relax", .10,
                replace(s, glia=replace(g, calcium=new_calcium,
                                        tripartite=g.tripartite and new_calcium > 0)),
                lost)

        if g.transferable and s.tissue.nutrient_input < self.af.max_nutrient_input:
            for i, c in enumerate(s.tissue.cells):
                if (c.substrate >= self.af.max_substrate or not
                        (self._near(g.voxel, c.soma)
                         or self._near(g.voxel, c.tip))):
                    continue
                cells = list(s.tissue.cells)
                cells[i] = replace(c, substrate=c.substrate + 1)
                tissue = replace(
                    s.tissue, cells=tuple(cells),
                    nutrient_input=s.tissue.nutrient_input + 1)
                add(f"glia_transfer:{i}", .24 + .08 * g.calcium,
                    replace(s, tissue=tissue,
                            glia=replace(g, transferable=g.transferable - 1),
                            glial_to_tissue=s.glial_to_tissue + 1))
            if g.calcium:
                loc = self.af.index(g.voxel)
                if s.tissue.attract[loc] < self.af.max_field:
                    field = list(s.tissue.attract)
                    field[loc] += 1
                    tissue = replace(
                        s.tissue, attract=tuple(field),
                        nutrient_input=s.tissue.nutrient_input + 1)
                    add("glia_local_signal", .17,
                        replace(s, tissue=tissue,
                                glia=replace(g, transferable=g.transferable - 1),
                                glial_to_tissue=s.glial_to_tissue + 1))

        # The 3-party relation changes BOTH synaptic stability and sampling.
        # It cannot exist without a paid neuron-neuron contact.
        if (s.tissue.links and g.calcium > 0 and g.transferable > 0 and
                not g.tripartite):
            add("tripartite_bind", .28,
                replace(s, glia=replace(g, transferable=g.transferable - 1,
                                        tripartite=True),
                        glial_waste=s.glial_waste + 1))
        if g.tripartite:
            add("tripartite_release", .055,
                replace(s, glia=replace(g, tripartite=False)),
                ("[R]_attr_glial_support", s.tissue.links))

        c = s.tissue.cognitive.context
        neuron_i = c % len(s.tissue.cells)
        sight = self.af.core.observed_pair(s.tissue.cognitive)
        if sight is not None and s.tissue.cognitive.last[0] == c:
            parent = s.tissue.cells[neuron_i]
            f = s.fibers[neuron_i]
            if parent.atp:
                # External G0-AE values are in its declared canonical R4
                # observation chart; an explicit per-fiber adapter is required.
                observed = observation_quaternion(sight.target, f.observer_frame)
                discrepancy = (qerror(f.phases[-1], observed)
                               if observed is not None else 0.)
                if discrepancy > 1e-8 and len(s.phase_history) < self.max_fiber_records:
                    updated = qinterpolate(f.phases[-1], observed, .30)
                    record = (neuron_i, len(f.phases) - 1,
                              f.phases[-1], updated,
                              s.tissue.cognitive.last)
                    target = replace(s,
                        fibers=self._put_phase(s, neuron_i, replace(
                            f, phases=f.phases[:-1] + (updated,))),
                        tissue=self._tissue_paid(s.tissue, neuron_i),
                        phase_history=s.phase_history + (record,))
                    add("fiber_experience_update",
                        .19 * discrepancy / (1. + discrepancy)
                        * (1. + .2 * g.calcium), target,
                        ("[R]_memory_prior", record))

                # A connection is independently mutable; update does not
                # overwrite the earlier correspondence without an audit.
                if (observed is not None and f.forward and
                        len(s.repairs) < self.max_fiber_records):
                    old = f.forward[-1]
                    mismatch = qerror(f.phases[-2], f.phases[-1], old)
                    if mismatch > 1e-8:
                        corrected = qnorm(
                            qmult(f.phases[-1], qconj(f.phases[-2])))
                        history = (neuron_i, len(f.forward) - 1, old,
                                   corrected, s.tissue.cognitive.last)
                        target = replace(
                            s, tissue=self._tissue_paid(s.tissue, neuron_i),
                            fibers=self._put_phase(s, neuron_i, replace(
                                f, forward=f.forward[:-1] + (corrected,))),
                            repairs=s.repairs + (history,))
                        add("fiber_connection_repair",
                            .14 * mismatch / (1. + mismatch), target,
                            ("[R]_transport_prior", history))

        return tuple(out)

    def row(self, s, reverse=False):
        links = self.channels(s)
        if reverse:
            links = tuple(reversed(links))
        result = {}
        for e in links:
            result[e.destination] = result.get(e.destination, 0.) + e.rate
        return result

    def diagonal(self, s):
        return -math.fsum(self.row(s).values())

    def trajectory(self, initial, horizon, *, seed=0, max_events=150):
        if (not self.valid(initial) or not math.isfinite(horizon) or
                horizon < 0 or type(max_events) is not int or max_events < 1):
            raise ValueError("invalid trajectory parameters")
        rng = random.Random(seed)
        t = 0.
        s = initial
        history = []
        for _ in range(max_events):
            possible = self.channels(s)
            total = math.fsum(e.rate for e in possible)
            if not total:
                break
            wait = rng.expovariate(total)
            if t + wait > horizon:
                break
            threshold = rng.random() * total
            acc = 0.
            selected = possible[-1]
            for e in possible:
                acc += e.rate
                if acc >= threshold:
                    selected = e
                    break
            t += wait
            s = selected.destination
            history.append((t, selected.name, selected.remainder))
        else:
            raise RuntimeError("event bound reached before horizon")
        return s, tuple(history)

    def summary(self, s):
        if not self.valid(s):
            raise ValueError("invalid joint state")
        current = s.tissue.cognitive
        sight = self.af.core.observed_pair(current)
        active_fiber = s.fibers[current.context % len(s.fibers)]
        unrepresentable = (sight is not None and observation_quaternion(
            sight.target, active_fiber.observer_frame) is None)
        return {
            "evidence": "finite-contract-only",
            "organism": "G0-AE + G0-AF + one G0-AG glial support Worldlet",
            "spatial_lattice_dimension": self.af.spatial_dims,
            "carrier_real_dimensions": tuple(
                f.real_carrier_dimension for f in s.fibers),
            "continuous_quaternion_segments": tuple(
                len(f.forward) for f in s.fibers),
            "endpoint_transport_discrepancy": tuple(
                qchain_discrepancy(f) for f in s.fibers),
            "glial_support_state": {
                "reserve": s.glia.substrate,
                "transferable": s.glia.transferable,
                "calcium": s.glia.calcium,
                "tripartite": s.glia.tripartite,
            },
            "fiber_removals_recorded": len(s.fiber_scars),
            "connection_repairs_recorded": len(s.repairs),
            "phase_revisions_recorded": len(s.phase_history),
            "observation_adapter": "A: canonical R4 coefficients to local H, not semantic identity",
            "unrepresentable_active_observation": unrepresentable,
            "R_transport_observation_norm": (
                math.hypot(*sight.target) if sight is not None else None),
            "joint_energy_residual": self._energy_balance(s),
            "af_tissue_balance": self.af.diagnostics(s.tissue)[
                "material_balance_residual"],
            "quantum_state": False,
            "real_tissue_physiology": False,
            "heldout_semantic_advantage": None,
        }


def selfcheck():
    system = GlialQuaternionEcology()
    initial = system.seed()
    assert system.valid(initial)
    assert abs(qlength2(qdirection((0, 1, 0))) - 1.) < 1e-12
    assert qerror(ONE, ONE) == 0.
    assert all(system.valid(e.destination) for e in system.channels(initial))
    current, events = system.trajectory(initial, .15, seed=13)
    assert system.valid(current)
    return {"initial": system.summary(initial),
            "terminal": system.summary(current),
            "events": len(events), "claims": "bounded contracts only"}


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2, sort_keys=True))
