// Architecture is illustrative; the optional voxel overlay replays recorded
// G0-AG Python states. This scene never simulates the organism in the browser.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const TAU = Math.PI * 2;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const point = v => new THREE.Vector3(v[0], v[1], v[2]);
const mix = (a, b, t) => a.clone().lerp(b, t);
function rng(seed) {
  let value = seed >>> 0;
  return () => { value = (1664525 * value + 1013904223) >>> 0; return value / 4294967296; };
}
function makeGlow(color) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 96;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(48, 48, 2, 48, 48, 48);
  gradient.addColorStop(0, 'rgba(255,255,255,.95)');
  gradient.addColorStop(.17, 'rgba(255,255,255,.32)');
  gradient.addColorStop(.48, 'rgba(255,255,255,.08)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 96, 96);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, color: new THREE.Color(color), transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending
  }));
  return sprite;
}
function orbitLine(radius, color, opacity, flattenY = 1) {
  const vertices = [];
  for (let i = 0; i <= 110; i++) {
    const phi = (i / 110) * TAU;
    vertices.push(new THREE.Vector3(Math.cos(phi) * radius, Math.sin(phi) * radius * flattenY, Math.sin(phi) * radius * .3));
  }
  const geom = new THREE.BufferGeometry().setFromPoints(vertices);
  return new THREE.Line(geom, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
}
export function createScene(viewport, labelsRoot, data, onSelect) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#07121a');
  scene.fog = new THREE.FogExp2('#07121a', .020);
  const camera = new THREE.PerspectiveCamera(45, 1, .1, 90);
  camera.position.set(0.0, 1.3, 14.7);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(1.7, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.id = 'webgl-canvas';
  renderer.domElement.setAttribute('aria-label', 'Orbitable schematic of the ESMF-0 architecture');
  viewport.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = .075;
  controls.enablePan = true;
  controls.minDistance = 5.1;
  controls.maxDistance = 25;
  controls.minPolarAngle = .10;
  controls.maxPolarAngle = Math.PI - .12;
  controls.target.set(0, .1, 0);
  controls.update();

  const domainById = new Map(data.domains.map(d => [d.id, d]));
  const nodeById = new Map(data.nodes.map(n => [n.id, n]));
  const visuals = new Map();
  const domains = [];
  const pickMeshes = [];
  const links = [];
  const higher = [];
  const tissue = [];
  const root = new THREE.Group();
  scene.add(root);
  let selected = null, moving = true, morph = .72, activeDomains = new Set(data.domains.map(d => d.id)), activeEvidence = new Set(['P', 'A', 'M', 'H', 'S', 'R']);
  let simulationTime = 0;
  let lastFrame = performance.now();
  let frameId = 0;
  let transition = null;
  let lastLabelRefresh = 0;

  // Independent local-world envelopes. No shared metric or one scalar clock.
  for (let i = 0; i < data.domains.length; i++) {
    const d = data.domains[i];
    const color = new THREE.Color(d.color);
    const group = new THREE.Group();
    group.position.copy(point(d.center));
    const carrier = new THREE.Mesh(new THREE.SphereGeometry(1, 34, 26), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .027, depthWrite: false, side: THREE.BackSide }));
    carrier.scale.set(i === 0 ? 1.90 : 1.75, 1.45, i === 2 ? 1.52 : 1.32);
    group.add(carrier);
    const shell = new THREE.Mesh(new THREE.IcosahedronGeometry(1.7, 1), new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: .067, depthWrite: false }));
    shell.scale.set(i === 0 ? 1.19 : 1, .85, .83);
    group.add(shell);
    const orbitA = orbitLine(1.78, color, .16, .66);
    orbitA.rotation.set(.5 + i * .4, .22 + i * .7, .18 + i);
    group.add(orbitA);
    const orbitB = orbitLine(1.40, color, .095, .64);
    orbitB.rotation.set(.4 - i * .1, 1.10 + i * .65, 1.2);
    group.add(orbitB);
    root.add(group);
    domains.push({ domain: d, group, shell, orbitA, orbitB, speed: [0.07, -.10, .15, -.045][i] });
  }

  // Structural bodies: several independent shells for each local Worldlet.
  const sphere = new THREE.IcosahedronGeometry(1, 2);
  const coreSphere = new THREE.SphereGeometry(1, 22, 18);
  for (const [index, n] of data.nodes.entries()) {
    const color = new THREE.Color(domainById.get(n.domain).color);
    const p = point(n.position);
    const group = new THREE.Group();
    group.position.copy(p);
    const glow = makeGlow(color);
    glow.scale.set(n.size * 6.8, n.size * 6.8, 1);
    glow.material.opacity = .31;
    group.add(glow);
    const orb = new THREE.Mesh(coreSphere, new THREE.MeshBasicMaterial({color, transparent: true, opacity: n.stage === 'draft' ? .38 : .65, depthWrite: false}));
    orb.scale.setScalar(n.size * .65);
    group.add(orb);
    const cage = new THREE.Mesh(sphere, new THREE.MeshBasicMaterial({color, wireframe: true, transparent: true, opacity: n.stage === 'concept' ? .22 : .42, depthWrite: false}));
    cage.scale.setScalar(n.size);
    group.add(cage);
    const loop = orbitLine(n.size * 1.26, color, .38, .48);
    loop.rotation.set(index * .31, .8 + index * .43, .35);
    group.add(loop);
    const pick = new THREE.Mesh(new THREE.SphereGeometry(n.size * 1.35, 14, 10), new THREE.MeshBasicMaterial({visible: false}));
    pick.userData.nodeId = n.id;
    group.add(pick);
    pickMeshes.push(pick);
    root.add(group);
    const label = document.createElement('div');
    label.className = 'node-label';
    label.textContent = n.label;
    label.dataset.id = n.id;
    labelsRoot.append(label);
    visuals.set(n.id, {node: n, group, glow, orb, cage, loop, pick, color, label, origin: p.clone(), phase: index * 1.69});
  }

  // Tendrils: oriented curves with moving packets, not undirected wires.
  for (const [index, link] of data.links.entries()) {
    const a = nodeById.get(link.from), b = nodeById.get(link.to);
    const p0 = point(a.position), p2 = point(b.position);
    const mid = mix(p0, p2, .5);
    const deviation = .45 + (index % 4) * .17;
    mid.z += deviation * (index % 2 ? -1 : 1);
    mid.y += .13 * Math.sin(index * 1.7);
    const curve = new THREE.CatmullRomCurve3([p0, mid, p2], false, 'centripetal', .1);
    const points = curve.getPoints(72);
    const color = domainById.get(a.domain).color;
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({color, opacity: link.type === 'audit' ? .095 : .17, transparent: true, depthWrite: false }));
    root.add(line);
    const packet = new THREE.Mesh(new THREE.SphereGeometry(.037, 7, 7), new THREE.MeshBasicMaterial({color, transparent: true, opacity: .8, depthWrite: false}));
    packet.position.copy(p0);
    root.add(packet);
    links.push({link, line, packet, curve, phase: index * .39, speed: .055 + (index % 6) * .017, base: line.material.opacity});
  }

  // Third-order relationships have their own 3-cells, not scalar pairwise edges.
  for (const h of data.hyperedges) {
    const pts = h.members.map(id => point(nodeById.get(id).position));
    const geometry = new THREE.BufferGeometry().setFromPoints([...pts, pts[0]]);
    const line = new THREE.Line(geometry, new THREE.LineDashedMaterial({color: '#e4caaa', dashSize: .19, gapSize: .20, transparent: true, opacity: .30, depthWrite: false}));
    line.computeLineDistances(); root.add(line);
    const centroid = pts.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / pts.length);
    const orb = new THREE.Mesh(new THREE.OctahedronGeometry(.19, 0), new THREE.MeshBasicMaterial({color: '#e2c2aa', wireframe: true, opacity: .34, transparent: true, depthWrite: false}));
    orb.position.copy(centroid); root.add(orb);
    higher.push({members: h.members, evidence: h.evidence, line, orb});
  }

  // Organic spatial tendrils, whose illustrated extent is scrubbed in the UI.
  const seed = rng(1931);
  const origin = point(nodeById.get('tissue').position);
  for (let i = 0; i < 10; i++) {
    const theta = i * 2.399 + .34;
    const v = new THREE.Vector3(Math.cos(theta), Math.sin(theta) * .65, Math.sin(theta * .8));
    v.normalize();
    const steps = [origin.clone(), origin.clone().addScaledVector(v, .46 + seed() * .29), origin.clone().addScaledVector(v, .86 + seed() * .45), origin.clone().addScaledVector(v, 1.3 + seed() * .9)];
    steps[2].z += (seed() - .5) * .6;
    steps[3].y += (seed() - .5) * .6;
    const curve = new THREE.CatmullRomCurve3(steps);
    const points = curve.getPoints(100);
    const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({color: '#87dbc2', transparent: true, opacity: .32, depthWrite: false}));
    path.geometry.setDrawRange(0, Math.max(2, Math.floor(morph * 98)));
    root.add(path);
    tissue.push({path, points, phase: i * .081, curve});
  }

  // A faint three-dimensional stellar field for depth; seeded and nonsemantic.
  const random = rng(7012);
  const starPositions = [];
  for (let i = 0; i < 800; i++) {
    const r = 7 + random() * 20, theta = random() * TAU, u = random() * 2 - 1;
    const planar = Math.sqrt(1 - u * u);
    starPositions.push(r * planar * Math.cos(theta), r * u, r * planar * Math.sin(theta));
  }
  const stars = new THREE.Points(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3)), new THREE.PointsMaterial({color: '#7caaa8', size: .025, transparent: true, opacity: .4, sizeAttenuation: true, depthWrite: false}));
  scene.add(stars);

  // Noninteractive signed-geometry rings: demonstrate plurality, never proof.
  const geometryRing = orbitLine(.94, '#b2aaf8', .13, .86);
  geometryRing.position.copy(point(nodeById.get('neutral').position));
  geometryRing.rotation.x = .96;
  root.add(geometryRing);

  // A separate, explicitly recorded layer. Positions are voxel-chart
  // projections from G0-AG state snapshots, never newly simulated growth.
  const traceRoot = new THREE.Group();
  root.add(traceRoot);
  const traceOrigin = point(nodeById.get('tissue').position);
  let replayActive = false;
  function tracePosition(voxel) {
    const side = 4; // trace schema's published side, validated in exporter.
    return traceOrigin.clone().add(new THREE.Vector3(
      (voxel[0] - (side - 1) / 2) * .39,
      (voxel[1] - (side - 1) / 2) * .39,
      (voxel[2] - (side - 1) / 2) * .39
    ));
  }
  function traceDot(at, radius, color, alpha = .9) {
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 10, 9),
      new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: alpha, depthWrite: false
      })
    );
    dot.position.copy(tracePosition(at));
    traceRoot.add(dot);
  }
  function traceLine(a, b, color, alpha = .82) {
    const points = [tracePosition(a), tracePosition(b)];
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({
        color, transparent: true, opacity: alpha, depthWrite: false
      })
    );
    traceRoot.add(line);
  }
  function clearTraceObjects() {
    for (const child of [...traceRoot.children]) {
      child.geometry?.dispose();
      child.material?.dispose();
      traceRoot.remove(child);
    }
  }
  function setReplay(snapshot, eventName = '') {
    clearTraceObjects();
    replayActive = true;
    const colors = ['#86edca', '#f4c78f'];
    for (let i = 0; i < snapshot.cells.length; i++) {
      const cell = snapshot.cells[i];
      const color = colors[i % colors.length];
      traceDot(cell.soma, .115, color, .94);
      let previous = cell.soma;
      for (const voxel of cell.path) {
        const wrapped = voxel.some((value, axis) => Math.abs(value - previous[axis]) > 1);
        // Periodic seams are not drawn as physically long neurites.
        if (!wrapped) traceLine(previous, voxel, color);
        traceDot(voxel, .065, color, .80);
        previous = voxel;
      }
      traceDot(previous, .13, color);
    }
    for (const field of ['attract', 'repel']) {
      for (const entry of snapshot.chemical[field]) {
        traceDot(entry.voxel, .042 + .013 * entry.units,
          field === 'attract' ? '#69c9ff' : '#ee9290', .55);
      }
    }
    if (snapshot.links.length > 0) {
      const a = snapshot.cells[0];
      const b = snapshot.cells[1];
      traceLine(a.path.at(-1) || a.soma, b.path.at(-1) || b.soma,
        '#f7dfb5', .55);
    }
    const g = snapshot.glia;
    traceDot(g.voxel, .09 + .025 * g.calcium,
      g.tripartite ? '#f5cf98' : '#8db5fc', .87);
    if (g.tripartite && snapshot.links.length > 0) {
      for (const cell of snapshot.cells) {
        traceLine(g.voxel, cell.path.at(-1) || cell.soma, '#a5abf5', .42);
      }
    }
    traceRoot.visible = activeDomains.has('biology') && activeEvidence.has('M');
    // Event labels remain tied to source event names, not invented signals.
    traceRoot.userData.lastRecordedEvent = eventName;
    for (const v of tissue) v.path.visible = false;
  }
  function clearReplay() {
    replayActive = false;
    clearTraceObjects();
    traceRoot.visible = false;
    for (const v of tissue) {
      v.path.visible = activeDomains.has('biology') && activeEvidence.has('M');
    }
  }

  function isVisible(n) {return activeDomains.has(n.domain) && activeEvidence.has(n.evidence);}
  function setFilters(domainsSet, evidenceSet) {
    activeDomains = new Set(domainsSet); activeEvidence = new Set(evidenceSet);
    for (const d of domains) d.group.visible = activeDomains.has(d.domain.id);
    for (const v of visuals.values()) {
      const active = isVisible(v.node);
      v.group.visible = active;
      v.label.style.display = active ? 'block' : 'none';
    }
    for (const l of links) {
      const a = nodeById.get(l.link.from), b = nodeById.get(l.link.to);
      const yes = isVisible(a) && isVisible(b);
      l.line.visible = yes; l.packet.visible = yes;
    }
    for (const h of higher) {
      const yes = activeEvidence.has(h.evidence) && h.members.every(id => isVisible(nodeById.get(id)));
      h.line.visible = yes; h.orb.visible = yes;
    }
    for (const v of tissue) v.path.visible = !replayActive && activeDomains.has('biology') && activeEvidence.has('M');
    traceRoot.visible = replayActive && activeDomains.has('biology') && activeEvidence.has('M');
    geometryRing.visible = activeDomains.has('geometry') && activeEvidence.has('P');
  }
  function highlight(id) {
    selected = id;
    for (const v of visuals.values()) {
      const yes = v.node.id === id;
      v.label.classList.toggle('is-selected', yes);
      v.orb.material.opacity = yes ? 1 : v.node.stage === 'draft' ? .38 : .65;
      v.cage.material.opacity = yes ? .83 : v.node.stage === 'concept' ? .22 : .42;
    }
    for (const l of links) l.line.material.opacity = (l.link.from === id || l.link.to === id) ? .50 : l.base;
  }
  function setMorphology(value) {
    morph = clamp(value, .1, 1);
    for (const v of tissue) v.path.geometry.setDrawRange(0, Math.max(2, Math.floor(morph * v.points.length)));
  }
  function setMotion(enabled) { moving = !!enabled; }
  function travel(position, target) { transition = {position, target}; }
  function view(name) {
    if (name === 'whole') travel(new THREE.Vector3(0, 1.3, 14.7), new THREE.Vector3(0, .1, 0));
    else if (domainById.has(name)) {
      const domain = domainById.get(name);
      const to = point(domain.center);
      travel(to.clone().add(new THREE.Vector3(1.25, 1.15, 7.9)), to);
    }
  }
  function focusNode(id) {
    const n = nodeById.get(id);
    if (!n) return;
    const to = point(n.position);
    travel(to.clone().add(new THREE.Vector3(1.0, .84, 6.8)), to);
  }
  let pointerOrigin = null;
  renderer.domElement.addEventListener('pointerdown', e => {
    pointerOrigin = [e.clientX, e.clientY];
    transition = null;
  });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!pointerOrigin || Math.hypot(e.clientX - pointerOrigin[0], e.clientY - pointerOrigin[1]) > 7) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster(); ray.setFromCamera(mouse, camera);
    const hit = ray.intersectObjects(pickMeshes.filter(obj => obj.parent.visible), false)[0];
    if (hit && hit.object.userData.nodeId) onSelect(hit.object.userData.nodeId);
  });

  function resize() {
    const width = Math.max(1, viewport.clientWidth), height = Math.max(1, viewport.clientHeight);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(viewport);
  resize();
  function updateLabels(now) {
    if (now - lastLabelRefresh < 65) return;
    lastLabelRefresh = now;
    for (const v of visuals.values()) {
      if (!v.group.visible) continue;
      const projected = v.group.getWorldPosition(new THREE.Vector3()).project(camera);
      const visible = projected.z > -1 && projected.z < 1 && Math.abs(projected.x) < 1.1 && Math.abs(projected.y) < 1.1;
      v.label.classList.toggle('is-visible', visible);
      if (!visible) continue;
      v.label.style.left = (projected.x * .5 + .5) * viewport.clientWidth + 'px';
      v.label.style.top = (-projected.y * .5 + .5) * viewport.clientHeight + 'px';
    }
  }
  function animate(now) {
    frameId = requestAnimationFrame(animate);
    const dt = clamp((now - lastFrame) / 1000, 0, .05);
    lastFrame = now;
    if (moving) simulationTime += dt;
    if (transition) {
      camera.position.lerp(transition.position, .065);
      controls.target.lerp(transition.target, .065);
      if (camera.position.distanceTo(transition.position) < .01 && controls.target.distanceTo(transition.target) < .01) transition = null;
    }
    controls.update();
    if (moving) {
      // Independent clocks illustrate concurrent constitutive dynamics.
      for (const d of domains) {
        d.orbitA.rotation.z += dt * d.speed;
        d.orbitB.rotation.x += dt * d.speed * .41;
        d.shell.rotation.y += dt * d.speed * .13;
      }
      for (const v of visuals.values()) {
        const wave = (Math.sin(simulationTime * (v.node.domain === 'biology' ? 1.33 : v.node.domain === 'semantics' ? 1.71 : .92) + v.phase) + 1) * .5;
        const scale = v.node.size * (1 + .035 * wave);
        v.cage.scale.setScalar(scale);
        v.glow.material.opacity = (.20 + wave * .18) * (v.node.stage === 'concept' ? .7 : 1);
        v.loop.rotation.z += dt * .07;
      }
      for (const l of links) {
        const t = (simulationTime * l.speed + l.phase) % 1;
        l.packet.position.copy(l.curve.getPointAt(t));
        l.packet.material.opacity = .32 + .54 * (Math.sin(t * Math.PI) ** 2);
      }
      for (const h of higher) h.orb.rotation.y += dt * .25;
      geometryRing.rotation.z += dt * .032;
    }
    updateLabels(now);
    renderer.render(scene, camera);
  }
  setFilters(activeDomains, activeEvidence);
  highlight('telos');
  frameId = requestAnimationFrame(animate);
  return {
    setFilters, highlight, setMorphology, setMotion, setReplay, clearReplay, view, focusNode,
    destroy() {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      clearTraceObjects();
      renderer.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.remove();
      labelsRoot.replaceChildren();
    }
  };
}
