// G0-AI / replay actual recorded stochastic morphogenesis, never simulate it.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = selector => document.querySelector(selector);
const TRACE = './traces/g0_ai_film.json';
const REPO = 'https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis';
let data, run, current=0, selected=0, timer=null, scene, camera, renderer, controls;
let modelRoot, selectable=[];
const viewport=$('#viewport');
const palette=[new THREE.Color('#69ded0'),new THREE.Color('#b3a6f9'),new THREE.Color('#efd0a2')];

function validate(d){
  if(d?.schema!=='esmf-g0-ai-telos-film/v1' || d.seed!==93101 ||
     d.world_steps!==720 || !/^[a-f0-9]{64}$/.test(d.payload_sha256) ||
     !Array.isArray(d.runs) || d.runs.length!==3)throw Error('Unexpected or outdated G0-AI recorded trace');
  for(const run of d.runs){
    if(!Array.isArray(run.frames)||run.frames.length<20)throw Error('Missing sampled model states');
    for(const [idx,frame] of run.frames.entries()){
      const s=frame.state;
      if(frame.index!==idx || !/^[a-f0-9]{64}$/.test(frame.state_sha256) ||
         s?.energy_residual!==0 || s.material_residual!==0 ||
         s.units?.length!==20 || !Array.isArray(s.contacts))
        throw Error('Trace state or conservation provenance failed');
      if(s.units.some(u=>u.rule_mix?.length!==3 || u.xyz?.length!==3 ||
         u.atp<0 || u.material<0 || Math.abs(u.rule_mix.reduce((a,b)=>a+b,0)-1)>.00001))
        throw Error('Recorded local rule or resource state is invalid');
    }
  }
  return d;
}

function p3(coord){return new THREE.Vector3(coord[0]*4,coord[2]*4,coord[1]*4);}
function localColor(u){
  const color=new THREE.Color('#050505');
  for(let i=0;i<3;i++)color.add(palette[i].clone().multiplyScalar(u.rule_mix[i]));
  return color;
}
function destroy(mesh){
  mesh.traverse(o=>{
    o.geometry?.dispose();
    if(o.material){for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();}
  });
  mesh.removeFromParent();
}
function cylinder(a,b,r,color,opacity=.7){
  const pa=p3(a),pb=p3(b),diff=pb.clone().sub(pa);
  const len=diff.length();
  if(len<.001)return;
  const material=new THREE.MeshBasicMaterial({color,transparent:true,opacity,depthWrite:false});
  const segment=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,7),material);
  segment.position.copy(pa).add(pb).multiplyScalar(.5);
  segment.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),diff.normalize());
  modelRoot.add(segment);
}
function draw(frame){
  if(!modelRoot)return;
  for(const mesh of [...modelRoot.children])destroy(mesh);
  selectable=[];
  const s=frame.state;
  const newContacts=new Set(s.recent_events.filter(e=>e.type==='growth')
                                  .map(e=>e.source+':'+e.target));
  for(const e of s.contacts){
    const a=s.units[e.source].xyz,b=s.units[e.target].xyz;
    const justBorn=newContacts.has(e.source+':'+e.target);
    cylinder(a,b,justBorn?.035:(.011+.025*Math.abs(e.weight)),
             justBorn?'#d1aafb':e.weight>=0?'#82e5c8':'#e8a5a7',
             justBorn?.95:.18+.52*Math.abs(e.weight));
  }
  for(let i=0;i<s.units.length;i++){
    const u=s.units[i];
    const size=.09+.075*Math.abs(u.activity)+(i===selected?.045:0);
    const mesh=new THREE.Mesh(new THREE.IcosahedronGeometry(size,1),
          new THREE.MeshStandardMaterial({color:localColor(u),roughness:.42,
                metalness:.23,emissive:localColor(u),emissiveIntensity:.33}));
    mesh.position.copy(p3(u.xyz));mesh.userData.cell=i;
    modelRoot.add(mesh);selectable.push(mesh);
    if(i===selected){
      const ring=new THREE.Mesh(new THREE.TorusGeometry(size+.085,.012,5,28),
        new THREE.MeshBasicMaterial({color:'#d8fff1',transparent:true,opacity:.8}));
      ring.position.copy(mesh.position);ring.rotation.x=.75;modelRoot.add(ring);
    }
  }
  // Loss is an actual recorded edge, not an invented fracture.
  for(const e of s.recent_events.filter(e=>e.type==='pruning')){
    const a=s.units[e.source]?.xyz,b=s.units[e.target]?.xyz;
    if(a&&b)cylinder(a,b,.014,'#f6a7b1',.26);
  }
}
function inspect(){
  if(!run)return;
  const s=run.frames[current].state;
  const u=s.units[selected];
  $('#clock').textContent=s.tick+' / '+(s.posthoc_world_phase===null?'—':s.posthoc_world_phase);
  $('#contacts').textContent=s.contacts.length;
  $('#growth').textContent=s.growth+' / '+s.pruning;
  $('#ledger').textContent=s.energy_residual+' / '+s.material_residual;
  $('#loss').textContent=s.cumulative_brier_after_warmup===null?'not scored':
      s.cumulative_brier_after_warmup.toFixed(4);
  $('#selection').textContent='Local site '+String(selected).padStart(2,'0');
  $('#selection-description').textContent='A site with no preassigned organ or question. Its current regulatory mixture and connections are consequences of its observed history and generic substrate rules.';
  const incoming=s.contacts.filter(e=>e.target===selected).length;
  const outgoing=s.contacts.filter(e=>e.source===selected).length;
  $('#detail').textContent=[
    'Position (numeric 3D): '+u.xyz.map(v=>v.toFixed(3)).join(', '),
    'Fast activation: '+u.activity.toFixed(4),
    'Slow local disposition: '+u.regulome.toFixed(4),
    'Adaptation rate: '+u.rate.toFixed(4),
    'Gain: '+u.gain.toFixed(4),
    'Rule mix [surprise, neighbor, resource]:',
    '  '+u.rule_mix.map(v=>v.toFixed(3)).join(' / '),
    'Contacts incoming/outgoing: '+incoming+' / '+outgoing,
    'ATP proxy: '+u.atp,
    'Structural material: '+u.material,
    'Observed cue: '+(s.observed_cue?.join(', ')??'not yet observed'),
    'Most recent label (posthoc): '+(s.observed_label===null?'—':s.observed_label),
    'Pre-label forecast: '+(s.prelabel_prediction===null?'—':s.prelabel_prediction.toFixed(4))
  ].join('\n');
  const events=s.recent_events.map(e=>(e.type==='growth'?'GROWN':'PRUNED')+
               ' '+e.source+'→'+e.target+' at '+e.tick);
  const losses=s.recent_remainders.map(e=>e.kind+' '+e.source+'→'+e.target+' at '+e.tick);
  $('#remainder').textContent=([...events,...losses].slice(-16).join('\n'))||
                               'No contact changes in this recorded interval.';
}
function show(index){
  if(!run)return;
  current=Math.min(run.frames.length-1,Math.max(0,Math.trunc(index)));
  const frame=run.frames[current];
  $('#frame').value=String(current);
  $('#progress').textContent=(current+1)+' / '+run.frames.length;
  $('#previous').disabled=current===0;
  $('#next').disabled=current===run.frames.length-1;
  $('#event').textContent=frame.state.tick===0?'Initial, weakly differentiated substrate':
    'After world encounter '+frame.state.tick+' · '+frame.state.recent_events.length+' recorded contact transitions';
  $('#event-type').textContent='Stochastic model decisions, fixed world seed; displayed frames are subsamples, not hand-selected stages. Hidden phase is revealed only after the encounter.';
  draw(frame);inspect();
}
function stop(){clearInterval(timer);timer=null;$('#play').textContent='Play';$('#play').setAttribute('aria-pressed','false');}
function choose(mode){stop();run=data.runs.find(x=>x.mode===mode)||data.runs[0];
  $('#frame').max=String(run.frames.length-1);show(current);}
function setup3D(){
  scene=new THREE.Scene();scene.background=new THREE.Color('#061723');
  camera=new THREE.PerspectiveCamera(46,1,.1,150);
  renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.65));
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  viewport.append(renderer.domElement);
  controls=new OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;controls.dampingFactor=.07;
  controls.minDistance=1.6;controls.maxDistance=35;
  camera.position.set(8,8,12);controls.target.set(0,0,0);controls.update();
  scene.add(new THREE.AmbientLight('#b4dced',1.6));
  const light=new THREE.PointLight('#91efce',45);light.position.set(4,9,7);scene.add(light);
  const fill=new THREE.PointLight('#b7a4f8',40);fill.position.set(-5,2,-4);scene.add(fill);
  modelRoot=new THREE.Group();scene.add(modelRoot);
  const resize=()=>{const rect=viewport.getBoundingClientRect();
    renderer.setSize(Math.max(1,rect.width),Math.max(1,rect.height),false);
    camera.aspect=rect.width/Math.max(1,rect.height);camera.updateProjectionMatrix();};
  if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe(viewport);
  else window.addEventListener('resize',resize);
  resize();
  const caster=new THREE.Raycaster(),pointer=new THREE.Vector2();
  renderer.domElement.addEventListener('pointerup',event=>{
    const rect=renderer.domElement.getBoundingClientRect();
    pointer.set(((event.clientX-rect.left)/rect.width)*2-1,
                -((event.clientY-rect.top)/rect.height)*2+1);
    caster.setFromCamera(pointer,camera);
    const hit=caster.intersectObjects(selectable,false)[0];
    if(hit){selected=hit.object.userData.cell;$('#cell').value=String(selected);draw(run.frames[current]);inspect();}
  });
  function animate(){controls.update();renderer.render(scene,camera);requestAnimationFrame(animate);}
  animate(); // viewer camera only; the tissue has no browser-driven update loop.
}
async function init(){
  const res=await fetch(TRACE,{cache:'no-cache'});
  if(!res.ok)throw Error('Recorded G0-AI states were not published');
  data=validate(await res.json());
  $('#hash').textContent=data.payload_sha256.slice(0,18);
  $('#raw-trace').href=TRACE;
  $('#raw-trace').setAttribute('download','g0_ai_film.json');
  $('#raw-trace').removeAttribute('target');
  if(/^[a-f0-9]{40}$/.test(data.source_commit)){
    $('#source').href=REPO+'/blob/'+data.source_commit+'/'+data.source;
    $('#source').textContent='Python state producer at '+data.source_commit.slice(0,10)+' ↗';
  }
  $('#cell').replaceChildren();
  for(let i=0;i<20;i++){const option=document.createElement('option');
    option.value=String(i);option.textContent='Local site '+String(i).padStart(2,'0');
    $('#cell').append(option);}
  $('#cell').addEventListener('change',()=>{selected=Number($('#cell').value);draw(run.frames[current]);inspect();});
  $('#mode').replaceChildren();
  for(const r of data.runs){const opt=document.createElement('option');
    opt.value=r.mode;opt.textContent=r.title;$('#mode').append(opt);}
  try{setup3D();}catch(error){console.warn('3D rendering unavailable',error);
    $('#fallback').hidden=false;}
  $('#mode').addEventListener('change',()=>choose($('#mode').value));
  $('#previous').addEventListener('click',()=>{stop();show(current-1);});
  $('#next').addEventListener('click',()=>{stop();show(current+1);});
  $('#frame').addEventListener('input',()=>{stop();show(Number($('#frame').value));});
  $('#play').addEventListener('click',()=>{
    if(timer)return stop();
    if(current===run.frames.length-1)show(0);
    $('#play').textContent='Pause';$('#play').setAttribute('aria-pressed','true');
    timer=setInterval(()=>{if(current>=run.frames.length-1)return stop();show(current+1);},750);
  });
  choose('adaptive');
}
init().catch(error=>{console.warn('G0-AI film failed',error);
  $('#event').textContent='Recorded experiment unavailable: '+error.message;
  $('#fallback').hidden=false;
  $('#fallback').querySelector('p').textContent=error.message;});
