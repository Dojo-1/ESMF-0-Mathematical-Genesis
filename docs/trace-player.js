// ESMF-0 / provenance-bearing trace reader, not a browser simulation.
// The underlying Python generator is executed offline by GitHub Actions.
// A "conditioned" event is a deliberately selected legitimate transition;
// only "sampled" events are drawn by the Gillespie stochastic process.
const $ = selector => document.querySelector(selector);
const REPO = 'https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis';
const TRACE_PATH = './traces/g0_ag_model.json';

function eventOrgan(eventName) {
  const name = eventName.replace(/^af\//, '');
  if (name.includes('neurite_')) return 'neurite';
  if (name.includes('contact_') || name.startsWith('tripartite_')) return 'glia';
  if (name.startsWith('glia_')) return 'glia';
  if (name.startsWith('fiber_')) return 'quaternion';
  if (name.startsWith('cognitive/encounter')) return 'encounter';
  if (name.startsWith('cognitive/interrogate')) return 'questions';
  if (name.startsWith('cognitive/dissipation')) return 'memory';
  if (name.startsWith('cognitive/')) return 'memory';
  if (name.includes('supply') || name.includes('metabolic_')) return 'resources';
  if (name.startsWith('secrete_') || name.startsWith('diffuse_') || name.startsWith('decay_')) return 'tissue';
  return 'tissue';
}

function validateTrace(data) {
  if (!data || data.schema !== 'esmf-observatory-trace/v1' ||
      !Array.isArray(data.runs) || data.runs.length < 1 ||
      !Number.isInteger(data.side) || data.spatial_dims !== 3) {
    throw new Error('The published trace format is unsupported.');
  }
  const ids = new Set();
  for (const run of data.runs) {
    if (!run.id || ids.has(run.id) || !Array.isArray(run.frames) || !run.frames.length) {
      throw new Error('A trace sequence is missing or duplicated.');
    }
    ids.add(run.id);
    let sampledTime = -1;
    for (const [index, frame] of run.frames.entries()) {
      const state = frame.state;
      if (frame.index !== index || !['initial', 'conditioned', 'sampled'].includes(frame.phase) ||
          !state || !Array.isArray(state.cells) || state.cells.length !== 2 ||
          state.ledger?.combined_energy_residual !== 0 ||
          state.ledger?.structural_material_residual !== 0 ||
          !/^[0-9a-f]{64}$/.test(frame.state_sha256)) {
        throw new Error('The trace contains an invalid recorded state.');
      }
      if (frame.phase === 'sampled') {
        if (!(typeof frame.time === 'number' && frame.time > sampledTime) ||
            !(frame.selected_rate > 0 && frame.outgoing_rate_sum >= frame.selected_rate)) {
          throw new Error('The stochastic trace has an invalid clock or hazard.');
        }
        sampledTime = frame.time;
      } else if (frame.time !== null) {
        throw new Error('A conditioned state is mislabeled as a sampled event.');
      }
      for (const neuron of state.cells) {
        if (!Array.isArray(neuron.soma) || !Array.isArray(neuron.path) ||
            neuron.soma.length !== 3 || neuron.real_carrier_dimension !== 8 + 4 * neuron.path.length) {
          throw new Error('Neurite state and local carrier rank disagree.');
        }
      }
    }
  }
  return data;
}

function timeDescription(frame) {
  if (frame.phase === 'initial') return 'INITIAL STATE · t not sampled';
  if (frame.phase === 'conditioned') return 'CONDITIONED · selected legal event, not randomly sampled';
  return 'GILLESPIE DRAW · t = ' + frame.time.toFixed(4) + ' model units';
}

function stateDescription(snapshot) {
  const n0 = snapshot.cells[0], n1 = snapshot.cells[1];
  const g = snapshot.glia;
  const discrepancy = n0.transport_defect + n1.transport_defect;
  return [
    'Cell A · ATP ' + n0.atp + ' · ' + n0.path.length + ' grown segments',
    'Cell B · ATP ' + n1.atp + ' · ' + n1.path.length + ' grown segments',
    'Glial proxy ' + g.transferable + ' · Ca gate ' + g.calcium +
      ' · tripartite ' + (g.tripartite ? 'engaged' : 'absent'),
    'Transport defect ' + discrepancy.toFixed(5) +
      ' · material / energy residuals 0 / 0',
    'Witnesses ' + snapshot.cognitive.witnesses
      .map(d => d.same + '/' + d.different).join(' · ') +
      ' · revisions ' + snapshot.provenance.phase_revisions
  ].join('\n');
}

export async function initializeTracePlayer({ getScene, onNode, onModeChange }) {
  const toggle = $('#trace-toggle');
  const panel = $('#trace-controls');
  const status = $('#trace-load-status');
  const runSelect = $('#trace-run');
  const slider = $('#trace-range');
  const count = $('#trace-position');
  const eventText = $('#trace-event');
  const phaseText = $('#trace-phase');
  const stateText = $('#trace-snapshot');
  const provenance = $('#trace-source');
  const play = $('#trace-play');
  const back = $('#trace-back');
  const next = $('#trace-next');

  let data = null, active = false, run = null, current = 0;
  let running = false, timer = null;

  function stop() {
    running = false;
    clearInterval(timer);
    timer = null;
    play.textContent = 'Play events';
    play.setAttribute('aria-pressed', 'false');
  }

  function show(index) {
    if (!run) return;
    current = Math.min(run.frames.length - 1, Math.max(0, Math.trunc(index)));
    const frame = run.frames[current];
    slider.value = String(current);
    count.textContent = String(current + 1).padStart(2, '0') +
      ' / ' + String(run.frames.length).padStart(2, '0');
    eventText.textContent = frame.event.replaceAll('_', ' ');
    phaseText.textContent = timeDescription(frame);
    stateText.textContent = stateDescription(frame.state);
    const rem = frame.typed_remainder?.type;
    $('#trace-remainder').textContent = rem
      ? 'Typed remainder retained: ' + rem
      : 'No typed remainder emitted by this transition';
    back.disabled = current === 0;
    next.disabled = current === run.frames.length - 1;
    getScene()?.setReplay(frame.state, frame.event);
    if (current > 0) onNode(eventOrgan(frame.event));
    $('#trace-snapshot').setAttribute('aria-label',
      'Recorded state after ' + frame.event);
  }

  function selectRun(id) {
    stop();
    run = data.runs.find(r => r.id === id) || data.runs[0];
    slider.max = String(run.frames.length - 1);
    slider.min = '0';
    $('#trace-run-note').textContent =
      run.conditioned_event_count
        ? run.conditioned_event_count + ' deliberately selected legal transitions, then ' +
          run.sampled_event_count + ' stochastic draws. These phases are not equivalent.'
        : run.sampled_event_count +
          ' stochastic draws from the unconditioned seed. No chosen prelude.';
    show(0);
  }

  function setActive(value) {
    if (!data) return;
    active = !!value;
    toggle.setAttribute('aria-pressed', String(active));
    toggle.textContent = active ? 'Return to atlas' : 'Replay model events';
    panel.hidden = !active;
    $('#morph-control').hidden = active;
    $('#motion-toggle').hidden = active;
    if (!active) {
      stop();
      getScene()?.clearReplay();
    } else {
      selectRun(runSelect.value);
    }
    onModeChange(active);
  }

  try {
    const result = await fetch(TRACE_PATH, { cache: 'no-cache' });
    if (!result.ok) throw new Error('Trace JSON has not yet been published.');
    data = validateTrace(await result.json());
    for (const r of data.runs) {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = r.title;
      runSelect.append(opt);
    }
    if (/^[0-9a-f]{40}$/.test(data.source_commit)) {
      provenance.href = REPO + '/blob/' + data.source_commit + '/' + data.source;
      provenance.textContent = 'Model at ' + data.source_commit.slice(0, 10) + ' ↗';
    } else {
      provenance.href = REPO + '/blob/main/' + data.source;
      provenance.textContent = 'Model source ↗';
    }
    $('#trace-payload-id').textContent = data.payload_sha256.slice(0, 14);
    toggle.disabled = false;
    status.textContent = 'Recorded from Python G0-AG · two deterministic traces · not live telemetry.';
    runSelect.addEventListener('change', () => selectRun(runSelect.value));
    slider.addEventListener('input', () => { stop(); show(Number(slider.value)); });
    back.addEventListener('click', () => { stop(); show(current - 1); });
    next.addEventListener('click', () => { stop(); show(current + 1); });
    play.addEventListener('click', () => {
      if (!active || !run) return;
      if (running) return stop();
      if (current >= run.frames.length - 1) show(0);
      running = true;
      play.textContent = 'Pause replay';
      play.setAttribute('aria-pressed', 'true');
      // Presentation pacing, NOT the stochastic holding times from G0-AG.
      timer = setInterval(() => {
        if (current >= run.frames.length - 1) return stop();
        show(current + 1);
      }, 1000);
    });
    toggle.addEventListener('click', () => setActive(!active));
    selectRun(data.runs[0].id);
    // Keep the default architecture view until the user chooses replay.
    getScene()?.clearReplay();
  } catch (error) {
    console.warn('Recorded event trace unavailable:', error);
    toggle.disabled = true;
    status.textContent = 'Recorded trace unavailable: ' + error.message +
      ' The original 3D atlas remains available.';
  }

  return {
    isActive: () => active,
    destroy: () => stop(),
  };
}
