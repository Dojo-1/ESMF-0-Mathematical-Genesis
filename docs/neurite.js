// G0-AH microscope: render snapshots exported from the Python reference.
// The browser NEVER advances neural state or manufactures event outcomes.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = selector => document.querySelector(selector);
const REPO = 'https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis';
const TRACE = './traces/g0_ah_microscope.json';
const COLORS = ['#91ebc9','#8fcaf2','#d1aaf0','#ecc69c'];
const $viewport = $('#viewport');
let data, run, current = 0, focusCell = 0, timer = null, playing = false;
let renderer, scene, camera, controls, modelRoot;
let selectable = [];
let chosenBranch = null;

function validate(record) {
  if (record?.schema !== 'esmf-g0-ah-compartmental-microscope/v1' ||
      record.cell_capacity !== 16 || record.branch_cap !== 4 ||
      record.side !== 20 || !Array.isArray(record.runs) ||
      record.runs.length !== 2 ||
      !/^[a-f0-9]{64}$/.test(record.payload_sha256)) {
    throw new Error('Unexpected G0-AH source trace or capacity');
  }
  for (const trajectory of record.runs) {
    if (!Array.isArray(trajectory.frames) || !trajectory.frames.length) {
      throw new Error('No recorded developmental frames');
    }
    trajectory.frames.forEach((frame, index) => {
      if (frame.index !== index || (index && frame.selection !== 'deliberately_selected_operator') ||
          !/^[a-f0-9]{64}$/.test(frame.sha256) ||
          frame.state?.energy_residual !== 0 || frame.state?.material_residual !== 0 ||
          frame.state.cells?.length !== 16 || frame.state.territories?.length !== 4) {
        throw new Error('Recorded state or evidence label failed validation');
      }
    });
  }
  return record;
}

const pos = voxel => new THREE.Vector3(
  (voxel[0] - 9) * .68, (voxel[2] - 2) * .94, (voxel[1] - 9) * .68
);
function material(color, opacity = 1) {
  return new THREE.MeshStandardMaterial({
    color, roughness: .38, metalness: .18, emissive: color,
    emissiveIntensity: .14, transparent: opacity < 1, opacity
  });
}
function addDot(group, voxel, radius, color, id) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 14, 12), material(color));
  mesh.position.copy(pos(voxel));
  mesh.userData = id;
  group.add(mesh);
  if (id) selectable.push(mesh);
  return mesh;
}
function addPath(group, from, to, color, alpha = 1, width = .025) {
  const a = pos(from), b = pos(to), vector = b.clone().sub(a);
  const len = vector.length();
  if (len < 1e-7) return;
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(width, width, len, 8),
    material(color, alpha)
  );
  mesh.position.copy(a).add(b).multiplyScalar(.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0), vector.normalize());
  group.add(mesh);
}
function dispose(node) {
  node.traverse(object => {
    object.geometry?.dispose();
    if (object.material) {
      const list = Array.isArray(object.material) ? object.material : [object.material];
      for (const mat of list) mat.dispose();
    }
  });
  node.removeFromParent();
}
function setCamera(cell = null) {
  if (!controls) return;
  const target = cell === null
    ? new THREE.Vector3(0, .6, 0)
    : pos(run.frames[current].state.cells[cell].soma);
  controls.target.copy(target);
  camera.position.copy(target).add(cell === null
    ? new THREE.Vector3(13, 14, 17) : new THREE.Vector3(4.7, 5.4, 7.0));
  controls.update();
}
function setup3D() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#071823');
  scene.fog = new THREE.FogExp2('#071823', .018);
  camera = new THREE.PerspectiveCamera(43, 1, .1, 120);
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.7));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.45;
  $viewport.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = .08;
  controls.minDistance = 2.2;
  controls.maxDistance = 35;
  scene.add(new THREE.AmbientLight('#9bcee3', 1.75));
  const light = new THREE.PointLight('#a1f8d7', 95);
  light.position.set(4,12,8);
  scene.add(light);
  const glow = new THREE.PointLight('#a18bf7', 55);
  glow.position.set(-8,4,-6);
  scene.add(glow);
  modelRoot = new THREE.Group();
  scene.add(modelRoot);
  const stars = new THREE.Group();
  const random = (() => {
    let n = 137; return () => ((n = (1664525*n + 1013904223) >>> 0) / 2**32);
  })();
  for (let i = 0; i < 230; i++) {
    const star = new THREE.Mesh(new THREE.SphereGeometry(.012 + random()*.019, 5, 4),
      new THREE.MeshBasicMaterial({color:'#5e98ac',transparent:true,opacity:.13 + random()*.3}));
    star.position.set((random()-.5)*34,(random()-.5)*22,(random()-.5)*34);
    stars.add(star);
  }
  scene.add(stars);
  const resize = () => {
    const rect=$viewport.getBoundingClientRect();
    renderer.setSize(Math.max(1,rect.width),Math.max(1,rect.height),false);
    camera.aspect=rect.width/Math.max(1,rect.height);
    camera.updateProjectionMatrix();
  };
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(resize).observe($viewport);
  else window.addEventListener('resize', resize);
  resize();
  setCamera();
  // This loop ONLY animates the user-controlled observer/camera.
  function animate() {
    controls.update();
    renderer.render(scene,camera);
    requestAnimationFrame(animate);
  }
  animate();
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  renderer.domElement.addEventListener('pointerup', event => {
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX-bounds.left)/bounds.width)*2-1,
                -((event.clientY-bounds.top)/bounds.height)*2+1);
    raycaster.setFromCamera(pointer,camera);
    const hits=raycaster.intersectObjects(selectable,false);
    if (!hits.length) return;
    const info=hits[0].object.userData;
    focusCell=info.cell;
    $('#cell').value=String(focusCell);
    chosenBranch=info.branch ?? null;
    inspect();
  });
}

function draw(frame) {
  if (!modelRoot) return;
  for (const child of [...modelRoot.children]) dispose(child);
  selectable = [];
  const state = frame.state;
  // Glial *membership* regions are depicted as an overlay, not a
  // reconstructed astrocytic anatomy or actual extracellular positions.
  for (const t of state.territories) {
    const center = t.members.reduce((acc,id) => acc.add(pos(state.cells[id].soma)),
      new THREE.Vector3()).multiplyScalar(1/t.members.length);
    const shell = new THREE.Mesh(
      new THREE.SphereGeometry(2.0,18,14),
      new THREE.MeshBasicMaterial({
        color:'#9c9de8',wireframe:true,transparent:true,
        opacity:t.calcium ? .18 : .075,depthWrite:false
      })
    );
    shell.position.copy(center);
    modelRoot.add(shell);
  }
  for (const cell of state.cells) {
    const color = COLORS[cell.context % COLORS.length];
    const selected=cell.id === focusCell;
    addDot(modelRoot,cell.soma,selected ? .22 : .15,color,
      {cell:cell.id,branch:null});
    for (const branch of cell.branches) {
      const parent=branch.parent === -1 ? cell.soma
        : cell.branches[branch.parent].voxel;
      addPath(modelRoot,parent,branch.voxel,color,selected ? .95 : .52,
        selected ? .037 : .02);
      const bright=branch.drive < 0 ? '#e8a3ad'
        : branch.eligibility ? '#efe4bc' : '#a4e2e8';
      addDot(modelRoot,branch.voxel,selected ? .10 : .065,bright,
        {cell:cell.id,branch:branch.id});
    }
  }
  for (const edge of state.synapses) {
    const source=state.cells[edge.source].soma;
    const target=state.cells[edge.target].branches[edge.branch]?.voxel;
    if (!target) continue;
    addPath(modelRoot,source,target,
      edge.kind === 'inhibit' ? '#f0a3ae' : '#91dcd1',.40,.012);
  }
  for (const packet of state.pending) {
    const edge=state.synapses.find(x => x.source===packet.source &&
      x.target===packet.target && x.branch===packet.branch);
    if (!edge) continue;
    const a=pos(state.cells[packet.source].soma);
    const b=pos(state.cells[packet.target].branches[packet.branch].voxel);
    const distance=Math.max(1,packet.arrival-state.fast_tick);
    const fraction=1/(1+distance);
    const p=a.lerp(b,fraction);
    const mesh=new THREE.Mesh(new THREE.SphereGeometry(.13,10,8),
      material(packet.amount < 0 ? '#f2a0ab' : '#9deace'));
    mesh.position.copy(p);
    modelRoot.add(mesh);
  }
}

function inspect() {
  if (!run) return;
  const snap=run.frames[current].state;
  const cell=snap.cells[focusCell];
  $('#clocks').textContent=snap.fast_tick + ' / ' + snap.slow_tick;
  $('#branches').textContent=snap.cells.reduce((sum,c) => sum+c.branches.length,0) +
      ' / 64 possible';
  $('#ledger').textContent=snap.energy_residual + ' / ' + snap.material_residual;
  const branch=chosenBranch===null ? null : cell.branches[chosenBranch];
  if (branch) {
    $('#selection').textContent='Dendrite '+cell.id+' / '+branch.id;
    $('#selection-description').textContent='A real local tree node with a parent and its own question-conditioned witnesses; not an autonomous neuron.';
    $('#detail').textContent=[
      'Parent: '+(branch.parent < 0 ? 'soma' : 'branch '+branch.parent),
      'Voxel: '+branch.voxel.join(', '),
      'Context / question: '+cell.context+' / '+cell.question,
      'Cue conjunction: '+branch.cue.join(' & '),
      'Fast [same, changed]: '+branch.fast.join(', '),
      'Long [same, changed]: '+branch.long.join(', '),
      'Local eligibility: '+branch.eligibility,
      'Inhibitory/excitatory drive: '+branch.drive,
      'P(change) from this branch: '+branch.predicted_change.toFixed(4),
      'Quaternionic transport discrepancy: '+branch.transport_defect.toFixed(6)
    ].join('\n');
  } else {
    $('#selection').textContent='Soma '+String(cell.id).padStart(2,'0');
    $('#selection-description').textContent='Local (context, question) worldlet with paid structure and variable-rank dendritic branches.';
    $('#detail').textContent=[
      'Context / question: '+cell.context+' / '+cell.question,
      'Soma voxel: '+cell.soma.join(', '),
      'ATP proxy: '+cell.atp,
      'Available structural material: '+cell.material,
      'Branches: '+cell.branches.length,
      'Branch IDs: '+cell.branches.map(b=>b.id).join(', '),
      'Synapses targeting this cell: '+snap.synapses.filter(x=>x.target===cell.id).length,
      'In-flight packets targeting cell: '+snap.pending.filter(x=>x.target===cell.id).length,
      'Glial territory: '+snap.territories.find(t=>t.members.includes(cell.id))?.id,
      'Total synthetic waste: '+snap.waste
    ].join('\n');
  }
  const recent=snap.remainders.filter(r=>r.cell===focusCell);
  $('#remainder').textContent=recent.length
      ? recent.map(r=>r.kind+' / branch '+r.branch+'\n'+r.reason).join('\n\n')
      : 'No recent typed loss recorded for this cell.';
}

function selectFrame(index) {
  if (!run) return;
  current=Math.max(0,Math.min(run.frames.length-1,index));
  const frame=run.frames[current];
  chosenBranch=null;
  $('#frame').max=String(run.frames.length-1);
  $('#frame').value=String(current);
  $('#progress').textContent=String(current+1).padStart(2,'0')+' / '+run.frames.length;
  $('#event').textContent=frame.event;
  $('#event-type').textContent=frame.selection === 'seed'
      ? 'Initial source state. No selected or sampled transition has occurred.'
      : 'Deliberately selected, legal Python operator · NOT a stochastic event';
  $('#previous').disabled=current===0;
  $('#next').disabled=current===run.frames.length-1;
  draw(frame);
  inspect();
}

function stop() {
  clearInterval(timer);timer=null;playing=false;
  $('#play').textContent='Play';
  $('#play').setAttribute('aria-pressed','false');
}
function chooseRun(id) {
  stop();
  run=data.runs.find(x=>x.id===id)||data.runs[0];
  focusCell=0;
  $('#cell').value='0';
  selectFrame(0);
  setCamera(run.id==='population'?null:0);
}
async function init() {
  const response=await fetch(TRACE,{cache:'no-cache'});
  if (!response.ok) throw new Error('The source-bound model trace is not published yet');
  data=validate(await response.json());
  $('#hash').textContent=data.payload_sha256.slice(0,18);
  if (/^[a-f0-9]{40}$/.test(data.source_commit)) {
    $('#source').href=REPO+'/blob/'+data.source_commit+'/'+data.source;
    $('#source').textContent='Python model at '+data.source_commit.slice(0,10)+' ↗';
  }
  $('#run').replaceChildren();
  for (const r of data.runs) {
    const option=document.createElement('option');
    option.value=r.id;option.textContent=r.title;
    $('#run').append(option);
  }
  $('#cell').replaceChildren();
  for(let i=0;i<16;i++) {
    const option=document.createElement('option');
    option.value=String(i);
    option.textContent='Cell '+String(i).padStart(2,'0')+
      ' · context '+Math.floor(i/4)+' / question '+(i%4);
    $('#cell').append(option);
  }
  try { setup3D(); }
  catch(error) {
    console.warn('WebGL unavailable; showing recorded states as text:',error);
    $('#fallback').hidden=false;
  }
  $('#run').addEventListener('change',()=>chooseRun($('#run').value));
  $('#cell').addEventListener('change',()=>{
    focusCell=Number($('#cell').value);chosenBranch=null;draw(run.frames[current]);inspect();
  });
  $('#previous').addEventListener('click',()=>{stop();selectFrame(current-1);});
  $('#next').addEventListener('click',()=>{stop();selectFrame(current+1);});
  $('#frame').addEventListener('input',()=>{stop();selectFrame(Number($('#frame').value));});
  $('#focus').addEventListener('click',()=>setCamera(focusCell));
  $('#whole').addEventListener('click',()=>setCamera(null));
  $('#play').addEventListener('click',()=>{
    if(playing) return stop();
    if(current===run.frames.length-1)selectFrame(0);
    playing=true;$('#play').textContent='Pause';
    $('#play').setAttribute('aria-pressed','true');
    // One second is a presentation interval, NOT a physiological clock.
    timer=setInterval(()=>{
      if(current===run.frames.length-1)return stop();
      selectFrame(current+1);
    },950);
  });
  chooseRun(data.runs[0].id);
}
init().catch(error=>{
  console.warn('G0-AH microscope could not initialize:',error);
  $('#event').textContent='Recorded state unavailable: '+error.message;
  $('#event-type').textContent='Inspect the GitHub source and preregistered control results instead.';
  $('#fallback').hidden=false;
  $('#fallback').querySelector('p').textContent=error.message;
});
