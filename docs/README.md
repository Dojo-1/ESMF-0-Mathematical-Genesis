# ESMF-0 / Living Observatory

A source-linked architectural cosmos with **two visibly separate modes**: the original Three.js schematic and a recorded event-by-event replay of the actual **synthetic G0-AG Python generator**. The atlas's breathing/transport particles remain illustrative. In replay mode, the neurite paths, glial proxy, chemical field packets, observed distinctions and resource ledgers come from recorded model states—not browser-generated biology. Neither mode implies actual neural cognition or quantum behavior.

## Local preview

From the repository root:

```bash
python -m research.export_observatory_trace --out docs/traces/g0_ag_model.json
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/`. WebGL requires a browser with GPU/WebGL enabled and access to the pinned Three.js module CDN. If WebGL or external modules fail, the static constellation and searchable source-linked node directory remain available. There is no server, account, tracking, telemetry or user-data upload.

## Provenance-bearing replay

The exporter is [`research/export_observatory_trace.py`](../research/export_observatory_trace.py). It records:

- **One conditioned reachability sequence:** eleven genuine, positive-rate G0-AG transitions selected deliberately to reveal glial contact, paid neurite extension, contextual encounter and continuous quaternion updates. **These are not random draws.**
- **One unconditioned stochastic sample:** a seeded Gillespie realization of the same synthetic generator. Only events marked `sampled` carry sampled holding times.
- **Each state:** 3D numerical voxel paths, physical-proxy ATP/substrate/material, glial gate and transferable pool, sparse chemical packets, cognitive witness counts, continuous carrier dimensions and transport defect, typed remainder when emitted, and zero-valued energy/material ledger residuals.

Each snapshot is fingerprinted with SHA-256. The complete exported file records the **exact source commit SHA** used to generate it, deterministic seeds, model horizon, sampling mode and the export source path. The HTML viewer verifies its shape and ledger closure; Python tests verify hashes, determinism, and the deliberate/sampled distinction. The replay is **not live telemetry** and the 1-second event playback interval is **not physical or Gillespie time**.

On a `main` deployment, GitHub Actions builds the replay JSON using the repository's Python model, validates the exported sequence, commits the resulting `docs/traces/g0_ag_model.json` to `main` and publishes the same file through Pages. The Actions-created data commit does not recursively start another Pages run.

## GitHub Pages

The workflow at `.github/workflows/observatory-pages.yml` publishes the `docs/` folder after validation on `main`. If Pages has not yet been enabled, a repository administrator must set **Settings > Pages > Build and deployment > Source > GitHub Actions** once. GitHub Actions' default token cannot enable Pages automatically.

The expected project URL, once deployed, is:

`https://dojo-1.github.io/ESMF-0-Mathematical-Genesis/`

## Visual semantics and evidence

- Slow breathing: **conceptual** exploratory/consolidation motion, not live state.
- Tendril extension: controlled **illustrative** material morphology; slider is a rendering parameter, not tissue energy.
- Traveling particles: directed, typed dependencies; not measured signal timings.
- Local shells: distinct observation/algebra contexts, without implying a global Euclidean brain.
- Triple connections: higher-order relations, not a proof of irreducible physical three-body effects.
- Evidence tags: P conditional mathematical validity, A explicit assumptions, M limited measurement, H hypothesis, S semantic framing, R retired/superseded (when present).

Three.js is imported at a pinned version via a public CDN; all architecture content is checked into this repository as `docs/architecture.json`. Research claims must be edited from the manifest with source references and tracked through code review. A site animation never authorizes promotion of an evidence label.

## Design boundaries

The G0-AD through D1a modules and the deterministic trace exporter are on **`main`**. An inspector label of **SOURCE ON MAIN** means the code, research note or synthetic study is present in the default branch; it does **not** mean the underlying learning, physiological or semantic claim has been established. H and S evidence tags remain visibly provisional. Historical review and corrections survive in the six merged pull requests. Neither D0 nor D1a proves biological realism or general intelligence. No holographic, topological, hypercomplex or quantum interpretation is inferred from the aesthetic rendering.
