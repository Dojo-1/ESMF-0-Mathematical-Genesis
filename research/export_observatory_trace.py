"""Export a *recorded* G0-AG state/event replay for the Living Observatory.

The observable is an actual finite sequence of generator transitions.
A reachability prelude deliberately selects valid positive-rate transitions
to expose phenomena; only the suffix is Gillespie sampled. They are never
presented as interchangeable, nor as a biological or held-out experiment.

Run:
    python -m research.export_observatory_trace --out docs/traces/g0_ag_model.json

No external network or LLM is used at any point in trace generation.
"""
import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

from esmf.glial_quaternion_ag import GlialQuaternionEcology, qchain_discrepancy
from research.g0_ag_development_probe import challenge_state

SCHEMA = "esmf-observatory-trace/v1"
PRELUDE = (
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


def canonical_bytes(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def fingerprint(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _sparse_field(model, values):
    return [
        {"voxel": list(model.af.site(index)), "units": quantity}
        for index, quantity in enumerate(values) if quantity
    ]


def snapshot(model, s):
    """Store state, not a decorative guess about its geometric meaning."""
    if not model.valid(s):
        raise AssertionError("cannot publish a nonviable ESMF-0 state")
    material_residual = model.af.diagnostics(s.tissue)[
        "material_balance_residual"
    ]
    energy_residual = model._energy_balance(s)
    if energy_residual or material_residual:
        raise AssertionError("missing conservation or provenance")
    cells = []
    for tissue, fiber in zip(s.tissue.cells, s.fibers):
        cells.append({
            "soma": list(tissue.soma),
            "path": [list(p) for p in tissue.path],
            "atp": tissue.atp,
            "substrate": tissue.substrate,
            "material": tissue.material,
            "real_carrier_dimension": fiber.real_carrier_dimension,
            "tip_quaternion": list(fiber.phases[-1]),
            "transport_defect": qchain_discrepancy(fiber),
        })
    return {
        "cells": cells,
        "glia": {
            "voxel": list(s.glia.voxel),
            "substrate": s.glia.substrate,
            "transferable": s.glia.transferable,
            "calcium": s.glia.calcium,
            "tripartite": s.glia.tripartite,
        },
        "links": [list(link) for link in s.tissue.links],
        "chemical": {
            "attract": _sparse_field(model, s.tissue.attract),
            "repel": _sparse_field(model, s.tissue.repel),
        },
        "cognitive": {
            "context": s.tissue.cognitive.context,
            "question": s.tissue.cognitive.question,
            "last": (list(s.tissue.cognitive.last)
                     if s.tissue.cognitive.last is not None else None),
            "witnesses": [
                {
                    "same": d.seen_same,
                    "different": d.seen_different,
                    "live_same": d.live_same,
                    "live_different": d.live_different,
                } for d in s.tissue.cognitive.distinctions
            ],
            "buds": list(s.tissue.cognitive.buds),
        },
        "provenance": {
            "pruned_sections": len(s.fiber_scars),
            "phase_revisions": len(s.phase_history),
            "connection_repairs": len(s.repairs),
            "neutral_tissue_scars": len(s.tissue.scars),
        },
        "external_inputs": {
            "nutrient": (
                s.tissue.nutrient_input - s.glial_to_tissue + s.glial_input
            ),
            "material": s.tissue.material_input,
        },
        "ledger": {
            "combined_energy_residual": energy_residual,
            "structural_material_residual": material_residual,
        },
    }


def append_frame(frames, model, state, *, phase, event, time, rate=None,
                 total_rate=None, remainder=()):
    represented = snapshot(model, state)
    frames.append({
        "index": len(frames),
        "phase": phase,
        "event": event,
        "time": time,
        "selected_rate": rate,
        "outgoing_rate_sum": total_rate,
        "typed_remainder": (
            {"type": remainder[0], "payload": json.loads(json.dumps(
                remainder[1], allow_nan=False))}
            if remainder else None
        ),
        "state": represented,
        "state_sha256": fingerprint(represented),
    })


def forced_prelude(model, current, frames):
    for name in PRELUDE:
        available = model.channels(current)
        matches = [ch for ch in available if ch.name == name]
        if len(matches) != 1 or matches[0].rate <= 0:
            raise AssertionError("no valid conditioned event: " + name)
        chosen = matches[0]
        current = chosen.destination
        append_frame(
            frames, model, current, phase="conditioned", event=chosen.name,
            time=None, rate=chosen.rate,
            total_rate=math.fsum(e.rate for e in available),
            remainder=chosen.remainder,
        )
    if current != challenge_state(model):
        raise AssertionError("conditioned prefix no longer matches D0")
    return current


def sample_suffix(model, start, frames, *, seed, horizon, max_events=180):
    """Exact Gillespie sampling, independent of illustrative playback speed."""
    rng = random.Random(seed)
    current = start
    elapsed = 0.
    for _ in range(max_events):
        channels = model.channels(current)
        total_rate = math.fsum(ch.rate for ch in channels)
        if not channels or total_rate <= 0:
            return current, elapsed
        if not math.isfinite(total_rate):
            raise ArithmeticError("nonfinite total rate")
        interval = rng.expovariate(total_rate)
        if elapsed + interval >= horizon:
            return current, horizon
        elapsed += interval
        threshold = rng.random() * total_rate
        running = 0.
        chosen = channels[-1]
        for ch in channels:
            running += ch.rate
            if running >= threshold:
                chosen = ch
                break
        current = chosen.destination
        append_frame(
            frames, model, current, phase="sampled",
            event=chosen.name, time=elapsed, rate=chosen.rate,
            total_rate=total_rate, remainder=chosen.remainder,
        )
    raise RuntimeError("event cap exhausted; no silent trace truncation")


def make_run(model, run_id, seed, horizon, conditioned):
    current = model.seed()
    frames = []
    append_frame(
        frames, model, current, phase="initial", event="initial",
        time=None,
    )
    if conditioned:
        current = forced_prelude(model, current, frames)
    terminal, final_time = sample_suffix(
        model, current, frames, seed=seed, horizon=horizon
    )
    if not model.valid(terminal):
        raise AssertionError("terminal state outside viability")
    return {
        "id": run_id,
        "title": ("Conditioned encounter · then stochastic events"
                  if conditioned else "Unconditioned stochastic run"),
        "seed": seed,
        "horizon": horizon,
        "source_process": "G0-AG bounded-channel pure-jump Markov process",
        "conditioned_event_count": len(PRELUDE) if conditioned else 0,
        "sampled_event_count": sum(
            frame["phase"] == "sampled" for frame in frames
        ),
        "final_sampled_time": final_time,
        "frames": frames,
    }


def export_trace(source_sha="unattributed-local"):
    if (not isinstance(source_sha, str) or not source_sha
            or len(source_sha) > 80):
        raise ValueError("invalid source commit identifier")
    model = GlialQuaternionEcology(spatial_dims=3, side=4)
    trace = {
        "schema": SCHEMA,
        "authority": "M: executable synthetic state/event trace ONLY",
        "source_commit": source_sha,
        "source": "esmf/glial_quaternion_ag.py",
        "exporter": "research/export_observatory_trace.py",
        "spatial_dims": model.af.spatial_dims,
        "side": model.af.side,
        "limitations": [
            "Animation is event-by-event; UI playback speed is not simulation time.",
            "The conditioned prefix deliberately selects legal positive-rate transitions; it is not stochastic sampling.",
            "Only suffix events are sampled by the model's Gillespie dynamics.",
            "Three-dimensional voxel positions are numerical coordinates, not biological morphology.",
            "Physical ATP and material values are bounded synthetic proxy units.",
            "A passing resource ledger and correct quaternion law do not validate cognition, biological tissue or quantum advantage.",
            "This is not a held-out D0/D1a efficacy result.",
        ],
        "runs": [
            make_run(model, "conditioned", seed=8100, horizon=.75,
                     conditioned=True),
            make_run(model, "unconditioned", seed=7012, horizon=1.6,
                     conditioned=False),
        ],
    }
    # A checksum over the *exact data as exported* except this checksum itself.
    trace["payload_sha256"] = fingerprint(trace)
    return trace


def write_trace(path, *, source_sha="unattributed-local"):
    target = Path(path)
    data = export_trace(source_sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA",
                   "unattributed-local"))
    args = p.parse_args()
    result = write_trace(args.out, source_sha=args.source_sha)
    print(json.dumps({
        "schema": result["schema"],
        "source_commit": result["source_commit"],
        "payload_sha256": result["payload_sha256"],
        "runs": [{
            "id": r["id"],
            "frames": len(r["frames"]),
            "conditioned": r["conditioned_event_count"],
            "sampled": r["sampled_event_count"],
        } for r in result["runs"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
