"""G0-AG D0: DEVELOPMENT-ONLY causal-rate controls on the existing joint generator.

Does not execute the proposed G0-AG1 held-out protocol. There is no
independently established cognitive task or biological benefit in this model.
The interventions modify *rates only*, with matched available observations,
state capacity, geometry, initial resources and event destinations.

Run:
    python -m research.g0_ag_development_probe --out /tmp/g0_ag_d0.json

All samples, including negative results, are written to a raw JSON record.
"""
import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path

from esmf.glial_quaternion_ag import (
    Event, GlialQuaternionEcology, qchain_discrepancy,
)

ARMS = (
    "full", "no_geometry_feedback", "no_glial_rate_feedback",
    "both_disabled", "placebo",
)
METRICS = (
    "contact_exposure", "tripartite_exposure", "geometry_defect_exposure",
    "external_nutrient_input", "external_material_input", "tissue_waste",
    "glial_waste", "physical_segments", "cognitive_witnesses",
    "encounters", "extensions", "connection_repairs",
)


def required_transition(model, state, name):
    choices = [e for e in model.channels(state) if e.name == name]
    if len(choices) != 1 or choices[0].rate <= 0:
        raise AssertionError("unreachable registered challenge event: " + name)
    destination = choices[0].destination
    if not model.valid(destination):
        raise AssertionError("challenge broke tissue ledger: " + name)
    return destination


def challenge_state(model):
    """Reach a joint glia+geometry contested mechanism state legally.

    Selection is explicitly conditioned on an event path. It is NOT a
    naturalistic or blind evaluation of the organism.
    """
    s = model.seed()
    path = (
        "af/neurite_extend:0:0",
        "af/contact_birth:0",
        "glia_calcium_pulse",
        "tripartite_bind",
        "af/metabolic_conversion:0",
        "af/cognitive/encounter:0",
        "af/metabolic_conversion:0",
        "fiber_experience_update",
        "af/nutrient_supply:0",
        "af/metabolic_conversion:0",
        "af/material_supply:0",
    )
    for name in path:
        s = required_transition(model, s, name)
    if not s.glia.tripartite or not s.tissue.links:
        raise AssertionError("glial rate feedback was not made active")
    if not s.tissue.cells[0].atp or not s.tissue.cells[0].material:
        raise AssertionError("growth mechanism lacks resources")
    if qchain_discrepancy(s.fibers[0]) <= 1e-9:
        raise AssertionError("geometric growth factor was not made active")
    return s


def control_events(model, state, arm):
    """Four rate interventions and a placebo on identical destinations."""
    if arm not in ARMS:
        raise ValueError("unknown preregistered control")
    geometry_off = arm in ("no_geometry_feedback", "both_disabled")
    glia_off = arm in ("no_glial_rate_feedback", "both_disabled")
    out = []
    for e in model.channels(state):
        rate = e.rate
        if geometry_off and e.name.startswith("af/neurite_extend:"):
            neuron = int(e.name.split(":")[1])
            rate *= 1. + .25 * qchain_discrepancy(state.fibers[neuron])
        if glia_off:
            if state.glia.tripartite and e.name.startswith("af/contact_release:"):
                rate /= .25
            elif (state.glia.tripartite and
                  e.name.startswith("af/cognitive/encounter:")):
                rate /= 1.20
            elif e.name == "fiber_experience_update":
                rate /= 1. + .20 * state.glia.calcium
            elif e.name.startswith("glia_transfer:"):
                rate -= .08 * state.glia.calcium
            elif e.name == "glia_maintenance" and state.glia.tripartite:
                rate -= .03
        if not math.isfinite(rate) or rate < 0.:
            raise AssertionError("negative or nonfinite control rate")
        if rate:
            out.append(Event(e.name, rate, e.destination, e.remainder))
    return tuple(out)


def exposure(state):
    return (
        int(bool(state.tissue.links)),
        int(state.glia.tripartite),
        math.fsum(qchain_discrepancy(f) for f in state.fibers),
    )


def simulate(model, start, arm, seed, horizon=.75, max_events=250):
    """Sample *exact* continuous-time event law, not an Euler pipeline."""
    if (type(seed) is not int or seed < 0 or
            not isinstance(horizon, (float, int)) or
            not math.isfinite(horizon) or horizon <= 0 or
            type(max_events) is not int or max_events < 1):
        raise ValueError("invalid run bounds")
    if not model.valid(start):
        raise ValueError("invalid initial state")
    rng = random.Random(seed)
    s = start
    t = 0.
    time_contact = time_tripartite = time_discrepancy = 0.
    names = []
    for _ in range(max_events):
        choices = control_events(model, s, arm)
        total = math.fsum(e.rate for e in choices)
        if not math.isfinite(total) or total < 0.:
            raise AssertionError("invalid outgoing generator intensity")
        wait = rng.expovariate(total) if total else math.inf
        dt = min(wait, horizon - t)
        contact, tripartite, discrepancy = exposure(s)
        time_contact += dt * contact
        time_tripartite += dt * tripartite
        time_discrepancy += dt * discrepancy
        t += dt
        if wait >= dt and t >= horizon - 1e-14:
            break
        if total == 0.:
            break
        draw = rng.random() * total
        running = 0.
        selected = choices[-1]
        for choice in choices:
            running += choice.rate
            if running >= draw:
                selected = choice
                break
        s = selected.destination
        if not model.valid(s):
            raise AssertionError("sample left the organism's viability region")
        names.append(selected.name)
    else:
        raise RuntimeError("event bound reached before the horizon")
    if model._energy_balance(s) != 0:
        raise AssertionError("combined proxy energy ledger failed")
    if model.af.diagnostics(s.tissue)["material_balance_residual"] != 0:
        raise AssertionError("structural resource ledger failed")
    counts = lambda prefix: sum(n.startswith(prefix) for n in names)
    return {
        "seed": seed,
        "arm": arm,
        "event_count": len(names),
        "contact_exposure": time_contact,
        "tripartite_exposure": time_tripartite,
        "geometry_defect_exposure": time_discrepancy,
        "external_nutrient_input": (
            s.tissue.nutrient_input - s.glial_to_tissue + s.glial_input
        ),
        "external_material_input": s.tissue.material_input,
        "tissue_waste": s.tissue.waste,
        "glial_waste": s.glial_waste,
        "physical_segments": sum(len(c.path) for c in s.tissue.cells),
        "cognitive_witnesses": sum(
            d.seen_same + d.seen_different
            for d in s.tissue.cognitive.distinctions
        ),
        "encounters": counts("af/cognitive/encounter:"),
        "extensions": counts("af/neurite_extend:"),
        "connection_repairs": counts("fiber_connection_repair"),
        "energy_residual": model._energy_balance(s),
        "material_residual": model.af.diagnostics(s.tissue)[
            "material_balance_residual"
        ],
        "state_valid": model.valid(s),
        "sampled_event_names": names,
    }


def descriptive_interval(values):
    if not values:
        raise ValueError("empty measurements")
    avg = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.
    return {
        "mean": avg,
        "normal_approx_95": [avg - 1.96 * sem, avg + 1.96 * sem],
        "n": len(values),
        "qualification": "descriptive, not a confirmatory inference",
    }


def run(seeds=range(8100, 8124), horizon=.75):
    seeds = tuple(seeds)
    if not seeds or any(type(v) is not int or v < 0 for v in seeds):
        raise ValueError("nonempty nonnegative seed sequence required")
    if not 0 < horizon <= 2.:
        raise ValueError("registered short-horizon probe only")
    model = GlialQuaternionEcology()
    start = challenge_state(model)
    audit = {
        "initial_contact": bool(start.tissue.links),
        "initial_tripartite": start.glia.tripartite,
        "initial_geometry_defect": qchain_discrepancy(start.fibers[0]),
        "initial_energy_residual": model._energy_balance(start),
        "initial_material_residual":
            model.af.diagnostics(start.tissue)["material_balance_residual"],
    }
    reference = model.channels(start)
    signature = tuple((x.name, x.destination) for x in reference)
    for arm in ARMS:
        if tuple((x.name, x.destination) for x in control_events(
                model, start, arm)) != signature:
            raise AssertionError("rate ablation changed the event state graph")
    cpu_start = time.process_time()
    records = [
        simulate(model, start, arm, seed, horizon=horizon)
        for seed in seeds
        for arm in ARMS
    ]
    cpu = time.process_time() - cpu_start
    by_key = {(r["seed"], r["arm"]): r for r in records}
    for seed in seeds:
        a = by_key[seed, "full"].copy()
        b = by_key[seed, "placebo"].copy()
        del a["arm"]
        del b["arm"]
        if a != b:
            raise AssertionError("placebo failed paired exact null")
    means = {
        arm: {
            metric: statistics.mean(
                by_key[seed, arm][metric] for seed in seeds
            )
            for metric in METRICS
        }
        for arm in ARMS
    }
    paired = {
        arm: {
            metric: descriptive_interval([
                by_key[seed, "full"][metric] -
                by_key[seed, arm][metric]
                for seed in seeds
            ])
            for metric in METRICS
        }
        for arm in ARMS if arm not in ("full", "placebo")
    }
    return {
        "scope": "development-only conditioned causal-rate probe; not G0-AG1 heldout",
        "status": "synthetic descriptive execution, no semantic/biological advantage",
        "source_git_commit": os.getenv("GITHUB_SHA"),
        "settings": {"seed_first": seeds[0], "seed_last": seeds[-1],
                     "n_seeds": len(seeds), "horizon": horizon,
                     "arms": ARMS},
        "challenge": audit,
        "check": {"placebo_exact": True, "event_destinations_matched": True,
                  "all_states_viable": all(r["state_valid"] for r in records),
                  "all_energy_residuals_zero":
                      all(r["energy_residual"] == 0 for r in records),
                  "all_material_residuals_zero":
                      all(r["material_residual"] == 0 for r in records)},
        "arm_means": means,
        "paired_full_minus_ablation": paired,
        "actual_cpu_seconds_for_simulations": cpu,
        "raw": records,
        "evidence": "M limited to synthetic development-only trajectory statistics",
        "unresolved": [
            "No independent task utility or true semantic discrimination",
            "Same seeds do not enforce identical event histories",
            "Energy and material inflow may diverge across trajectories",
            "No cross-architecture compute or external-resource parity",
            "No physical joules, biological validity, quantum behavior or cognition",
            "The challenge is selected by a designed event sequence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("G0_AG_D0_SUMMARY=" + json.dumps(
        {key: value for key, value in report.items() if key != "raw"},
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
