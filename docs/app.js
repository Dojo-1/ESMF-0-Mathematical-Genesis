// ESMF-0 / Living Observatory
// All motion is a schematic, never a claim of running organism telemetry.
const $ = (selector) => document.querySelector(selector);
const state = {
  activeDomains: new Set(),
  activeEvidence: new Set(['P', 'A', 'M', 'H', 'S', 'R']),
  selected: 'telos',
  motion: !window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  morphology: 72,
  view: 'whole',
  scene: null,
  data: null
};
const evidenceNames = {
  P: 'Conditional proved structure', A: 'Assumption / surrogate',
  M: 'Bounded measurement', H: 'Hypothesis', S: 'Semantic framing',
  R: 'Retired / superseded'
};

function el(tag, className, textValue) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (textValue !== undefined) item.textContent = textValue;
  return item;
}
function repoLink(href, label, className) {
  const a = el('a', className, label);
  if (typeof href === 'string' && href.startsWith('https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis')) {
    a.href = href;
    a.rel = 'noopener noreferrer';
    a.target = '_blank';
  } else {
    a.href = 'https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis';
  }
  return a;
}
function colorFor(domain) { return state.data.domains.find(x => x.id === domain)?.color || '#a6c6cb'; }
function visible(node) { return state.activeDomains.has(node.domain) && state.activeEvidence.has(node.evidence); }
function reconcileSelection() {
  const current = state.data.nodes.find(n => n.id === state.selected);
  if (!current || !visible(current)) state.selected = state.data.nodes.find(visible)?.id || null;
}
function updateVisibility() {
  reconcileSelection();
  state.scene?.setFilters(state.activeDomains, state.activeEvidence);
  state.scene?.highlight(state.selected);
  updateInspector();
  for (const b of document.querySelectorAll('.node-entry')) {
    const node = state.data.nodes.find(n => n.id === b.dataset.id);
    b.hidden = !node || !visible(node);
    b.classList.toggle('is-selected', b.dataset.id === state.selected);
  }
}
function selectNode(id, focus = true) {
  const node = state.data.nodes.find(n => n.id === id);
  if (!node) return;
  state.activeDomains.add(node.domain);
  state.activeEvidence.add(node.evidence);
  for (const b of document.querySelectorAll('.domain')) {
    const on = state.activeDomains.has(b.dataset.domain);
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', String(on));
  }
  for (const b of document.querySelectorAll('.evidence-button')) {
    const on = state.activeEvidence.has(b.dataset.evidence);
    b.classList.toggle('is-on', on);
    b.setAttribute('aria-pressed', String(on));
  }
  state.selected = id;
  updateVisibility();
  if (focus) state.scene?.focusNode(id);
}
function updateInspector() {
  const target = $('#inspector-content');
  target.replaceChildren();
  const node = state.data.nodes.find(n => n.id === state.selected);
  if (!node) {
    target.append(el('p', 'inspector-more', 'All layers are hidden. Activate an organ or an evidence label to inspect its source.'));
    $('#inspector-index').textContent = '-- / --';
    return;
  }
  const idx = state.data.nodes.findIndex(n => n.id === node.id) + 1;
  $('#inspector-index').textContent = String(idx).padStart(2, '0') + ' / ' + state.data.nodes.length;
  target.append(el('p', 'inspector-eyebrow', node.domain.toUpperCase() + ' / ' + node.stage.toUpperCase()));
  target.append(el('h2', null, node.label));
  target.append(el('p', 'inspector-summary', node.summary));
  const meta = el('div', 'inspector-meta');
  meta.append(el('span', 'meta-chip is-evidence', node.evidence + ' / ' + evidenceNames[node.evidence]));
  meta.append(el('span', 'meta-chip', node.stage === 'main' ? 'BASE BRANCH' : node.stage === 'draft' ? 'DRAFT RESEARCH' : 'DESIGN FRAME'));
  target.append(meta);
  target.append(el('p', 'inspector-more', node.detail));
  target.append(repoLink(node.source, 'Read the corresponding research  \u2197', 'source-link'));
}
function renderControls() {
  const root = $('#domain-controls');
  root.replaceChildren();
  for (const domain of state.data.domains) {
    state.activeDomains.add(domain.id);
    const btn = el('button', 'domain is-on');
    btn.type = 'button'; btn.dataset.domain = domain.id;
    btn.style.setProperty('--c', domain.color);
    btn.setAttribute('aria-pressed', 'true');
    btn.setAttribute('title', domain.description);
    btn.append(el('span', 'domain-dot'));
    btn.append(el('span', '', domain.name));
    btn.append(el('span', 'domain-count', String(state.data.nodes.filter(n => n.domain === domain.id).length).padStart(2, '0')));
    btn.addEventListener('click', () => {
      if (state.activeDomains.has(domain.id)) state.activeDomains.delete(domain.id);
      else state.activeDomains.add(domain.id);
      btn.classList.toggle('is-on', state.activeDomains.has(domain.id));
      btn.setAttribute('aria-pressed', String(state.activeDomains.has(domain.id)));
      updateVisibility();
    });
    root.append(btn);
  }
  const evidences = $('#evidence-controls');
  for (const k of Object.keys(evidenceNames)) {
    const btn = el('button', 'evidence-button is-on', k);
    btn.dataset.evidence = k; btn.type = 'button';
    btn.title = evidenceNames[k]; btn.setAttribute('aria-label', evidenceNames[k]);
    btn.setAttribute('aria-pressed', 'true');
    btn.addEventListener('click', () => {
      if (state.activeEvidence.has(k)) state.activeEvidence.delete(k);
      else state.activeEvidence.add(k);
      const on = state.activeEvidence.has(k);
      btn.classList.toggle('is-on', on);
      btn.setAttribute('aria-pressed', String(on));
      updateVisibility();
    });
    evidences.append(btn);
  }
  const list = $('#node-directory');
  for (const node of state.data.nodes) {
    const btn = el('button', 'node-entry');
    btn.type = 'button'; btn.dataset.id = node.id;
    btn.style.setProperty('--entry-color', colorFor(node.domain));
    btn.append(el('i', '')); btn.append(el('span', '', node.label));
    btn.append(el('small', '', node.evidence));
    btn.addEventListener('click', () => selectNode(node.id));
    list.append(btn);
  }
  for (const finding of state.data.findings) {
    const card = el('article', 'finding');
    card.append(el('div', 'finding-title', finding.name.toUpperCase()));
    card.append(el('div', 'finding-value', finding.value));
    card.append(el('div', 'finding-unit', finding.unit));
    card.append(el('p', 'finding-note', finding.footnote));
    card.append(repoLink(finding.source, 'Evidence & conditions \u2197'));
    $('#findings').append(card);
  }
  for (const btn of document.querySelectorAll('.view-btn')) {
    btn.addEventListener('click', () => {
      state.view = btn.dataset.view;
      for (const other of document.querySelectorAll('.view-btn')) {
        const active = other === btn;
        other.classList.toggle('is-current', active);
        other.setAttribute('aria-pressed', String(active));
      }
      state.scene?.view(btn.dataset.view);
    });
  }
  $('#morph-range').addEventListener('input', e => {
    const value = Number(e.target.value);
    state.morphology = value;
    $('#morph-value').textContent = value + '%';
    state.scene?.setMorphology(value / 100);
  });
  $('#motion-toggle').addEventListener('click', () => {
    state.motion = !state.motion;
    state.scene?.setMotion(state.motion);
    setMotionLabel();
  });
  $('#reset-camera').addEventListener('click', () => state.scene?.view('whole'));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      document.querySelector('.directory details').open = false;
      state.scene?.view('whole');
    }
    if (event.key === ' ' && event.target === $('#scene-viewport')) {
      event.preventDefault();
      state.motion = !state.motion;
      state.scene?.setMotion(state.motion);
      setMotionLabel();
    }
  });
  setMotionLabel();
  updateVisibility();
}
function setMotionLabel() {
  $('#motion-toggle').setAttribute('aria-pressed', String(state.motion));
  $('#motion-symbol').textContent = state.motion ? '\u275A\u275A' : '\u25B6';
  $('#motion-text').textContent = state.motion ? 'Pause' : 'Play';
}
function validateManifest(data) {
  if (!data || data.schema !== 'esmf-observatory/v1' || !Array.isArray(data.domains) || !Array.isArray(data.nodes)) {
    throw new Error('Architecture manifest is missing or incompatible.');
  }
  const ids = new Set();
  const domains = new Set(data.domains.map(d => d.id));
  for (const n of data.nodes) {
    if (ids.has(n.id) || !domains.has(n.domain) || !n.position || n.position.length !== 3 || !evidenceNames[n.evidence]) {
      throw new Error('Invalid or duplicated structure: ' + n.id);
    }
    ids.add(n.id);
  }
  for (const l of data.links || []) if (!ids.has(l.from) || !ids.has(l.to)) throw new Error('Missing transport endpoint');
  for (const h of data.hyperedges || []) if (!h.members.every(id => ids.has(id))) throw new Error('Incomplete higher relation');
}
async function start() {
  try {
    const response = await fetch('./architecture.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error('Unable to read architecture manifest');
    state.data = await response.json();
    validateManifest(state.data);
    renderControls();
    updateInspector();
    try {
      const { createScene } = await import('./scene.js');
      state.scene = createScene($('#scene-viewport'), $('#floating-labels'), state.data, selectNode);
      state.scene.setFilters(state.activeDomains, state.activeEvidence);
      state.scene.highlight(state.selected);
      state.scene.setMorphology(state.morphology / 100);
      state.scene.setMotion(state.motion);
      $('#scene-fallback').classList.add('is-hidden');
    } catch (error) {
      console.warn('WebGL unavailable; using accessible architecture index:', error);
      $('#render-state').textContent = '3D unavailable in this browser. The structures and evidence directory remain interactive.';
      document.querySelector('.directory details').open = true;
    }
    $('#footer-model').textContent = 'SCHEMATIC  /  NOT LIVE TELEMETRY  /  ' + state.data.version;
  } catch (error) {
    console.error(error);
    $('#domain-controls').replaceChildren(el('div', 'error-panel', 'Could not load the architecture. Browse the research map from the header.'));
    $('#render-state').textContent = 'The spatial view could not load.';
    $('#footer-model').textContent = 'ARCHITECTURE DATA UNAVAILABLE';
  }
}
start();
