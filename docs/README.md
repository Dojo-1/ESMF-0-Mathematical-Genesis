# ESMF-0 / Living Observatory

A source-linked architectural cosmos with **two visibly separate modes**: the original Three.js schematic and a recorded event-by-event replay of the actual **synthetic G0-AG Python generator**. The atlas's breathing/transport particles remain illustrative. In replay mode, the neurite paths, glial proxy, chemical field packets, observed distinctions and resource ledgers come from recorded model states—not browser-generated biology. Neither mode implies actual neural cognition or quantum behavior.

## Local preview

From the repository root:

```bash
python -m research.export_observatory_trace --out docs/traces/g0_ag_model.json
python -m research.export_g0_ah_microscope --out docs/traces/g0_ah_microscope.json
python -m research.export_g0_ai_film --out docs/traces/g0_ai_film.json
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000/` for the architectural atlas or `http://localhost:8000/neurite.html` for the recorded G0-AH neurite microscope. WebGL requires a browser with GPU/WebGL enabled and access to the pinned Three.js module CDN. If WebGL or external modules fail, the static constellation and searchable source-linked node directory remain available. There is no server, account, tracking, telemetry or user-data upload.

## Provenance-bearing replay

The exporter is [`research/export_observatory_trace.py`](../research/export_observatory_trace.py). It records:

- **One conditioned reachability sequence:** eleven genuine, positive-rate G0-AG transitions selected deliberately to reveal glial contact, paid neurite extension, contextual encounter and continuous quaternion updates. **These are not random draws.**
- **One unconditioned stochastic sample:** a seeded Gillespie realization of the same synthetic generator. Only events marked `sampled` carry sampled holding times.
- **Each state:** 3D numerical voxel paths, physical-proxy ATP/substrate/material, glial gate and transferable pool, sparse chemical packets, cognitive witness counts, continuous carrier dimensions and transport defect, typed remainder when emitted, and zero-valued energy/material ledger residuals.

Each snapshot is fingerprinted with SHA-256. The complete exported file records the **exact source commit SHA** used to generate it, deterministic seeds, model horizon, sampling mode and the export source path. The HTML viewer verifies its shape and ledger closure; Python tests verify hashes, determinism, and the deliberate/sampled distinction. The replay is **not live telemetry** and the 1-second event playback interval is **not physical or Gillespie time**. Unit-quaternion tip states appear as SO(3) frame glyphs via the S³ double cover: the rendering identifies q and −q, and is neither a biological orientation nor a full rendering of quaternionic state.

On a `main` deployment, GitHub Actions generates both model traces from the checked-out Python sources and publishes those generated files in the Pages artifact. It also archives the raw JSON files as a workflow artifact. **Deployment does not commit or push anything back to `main`**—this avoids non-fast-forward races during active research, while each trace retains the exact source checkout SHA. Tracked copies under `docs/traces/` may therefore represent an earlier audited snapshot; the deployed trace and its Actions artifact are authoritative for that published revision.

## G0-AH Neurite Microscope

The separate [3D neurite microscope](neurite.html) depicts **actual selected operations** of [G0-AH](../esmf/compartmental_ah.py). [Its source-bound exporter](../research/export_g0_ah_microscope.py) records one sixteen-stage focused neurite demonstration (branching, question-indexed witnesses, glial slow gate, pruning, a delayed inhibitory contact and fast-clock delivery) and one sixteen-cell/64-branch population demonstration. Unlike G0-AG's Gillespie suffix, **neither G0-AH recording is a stochastic developmental sample**: the operator sequence was selected to exhibit valid behavior. Glial territorial shells depict cell membership, not measured astrocyte positions. The 3D voxels are nonperiodic numerical locations, not anatomy.

The [G0-AH D0 forecasting study](../research/G0_AH_D0_RESULTS.md) was independently scored against an ordinary cue-indexed counter. That conventional baseline achieved **lower Brier loss**, and glial slow consolidation did not change the G0-AH scores. Passing resource conservation and quaternionic gauge tests cannot reverse that negative cognitive result.

## G0-AI: development without assigned organs

[Conditions for Genesis](genesis.html) shows **recorded model-selected** stochastic contact changes of 20 initially role-free local units, with history-adapted regulatory mixtures and exact synthetic energy/material balances. The [film exporter](../research/export_g0_ai_film.py) reproduces the declared development seed 93101, and tracks adaptive, frozen and shuffled-topology variants on the **same** stimulus stream. The site renders subsampled recorded states; its playback speed is not simulation time, and its chart is not biological 3D tissue. No organ types or biological glial classes arise by assertion. The [D0 results](../research/G0_AI_D0_RESULTS.md) showed worse prediction than the simpler fixed control, and the [conditional P1 obstruction](../research/G0_AI_P1_SYMMETRY_OBSTRUCTION.md) identifies a precise odd-symmetry restriction of its local observation algebra. Neither result is evidence for emergent intelligence.

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
