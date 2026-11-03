// Live G0-AI research microscope.
// All tissue transitions originate in the source-pinned Python TelosSubstrate.
// JS parses deliberately narrow input tasks and projects Python states into 3D.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';

const $ = id => document.getElementById(id);
const ROOT = 'https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis';
const CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
const MATH_EXAMPLES = ['2^5 - 3*(4+1)', '(12 + 8) / 2', '7^2 - 3*3', '42 - (6*7)', '4*(3+5) - 1'];
const LANGUAGE_EXAMPLES = ['Why do models remember a question?', 'Not every pattern is a proof.', 'Can different worlds recognize one another?', 'The geometry changes, but the evidence remains.'];
const NEGATORS = new Set(['not','no','never','neither','without','cannot','none','nobody','nothing',"don't","isn't","wasn't","can't"]);
const QUESTIONS = new Set(['why','what','who','whom','where','when','how','which']);
const state = {task:'math', py:null, pyReady:false, substrate:null, seed:93101, mode:'adaptive', last:null, selected:0, busy:false, autoTimer:null, demoIndex:0, motion:!matchMedia('(prefers-reduced-motion: reduce)').matches, showSlow:true, showContacts:true, scene:null, events:[]};
function status(message,isError=false){$('runtime-status').textContent=message;$('runtime-status').classList.toggle('error',isError);}
function assertFinite(value){if(!Number.isFinite(value)||Math.abs(value)>1e9)throw Error('Value is outside the supported arithmetic range');return value;}
function parseArithmetic(input){
  if(input.length>160)throw Error('Expression too long (160 characters maximum).');
  const tokens=[];let index=0;const lex=/\s*(\d+(?:\.\d+)?|[()+\-*/^])/y;
  while(index<input.length){lex.lastIndex=index;const m=lex.exec(input);if(!m){if(input.slice(index).trim()==='')break;throw Error('Unsupported character near position '+(index+1));}tokens.push(m[1]);index=lex.lastIndex;if(tokens.length>90)throw Error('Expression is too complex');}
  if(!tokens.length)throw Error('Enter an arithmetic expression.');
  let cursor=0,depth=0;
  const peek=()=>tokens[cursor],take=()=>tokens[cursor++];
  function primary(){
    if(++depth>35)throw Error('Expression nesting limit reached.');
    let result;
    if(peek()==='('){take();result=expr();if(take()!==')')throw Error('Missing closing parenthesis');}
    else {const t=take();if(t===undefined||!/^\d+(?:\.\d+)?$/.test(t))throw Error('Expected a number.');result=Number(t);}
    depth--;return assertFinite(result);
  }
  function power(){let a=primary();if(peek()==='^'){take();a=assertFinite(a**unary());}return a;}
  function unary(){if(peek()==='+'){take();return unary();}if(peek()==='-'){take();return -unary();}return power();}
  function product(){let a=unary();while(peek()==='*'||peek()==='/'){const op=take(),b=unary();if(op==='/'&&b===0)throw Error('Division by zero');a=assertFinite(op==='*'?a*b:a/b);}return a;}
  function expr(){let a=product();while(peek()==='+'||peek()==='-'){const op=take(),b=product();a=assertFinite(op==='+'?a+b:a-b);}return a;}
  const value=expr();if(cursor!==tokens.length)throw Error('Unexpected operator or unmatched parenthesis.');
  const even=Number.isInteger(value)&&Math.abs(value%2)<1e-10;
  const cue=[value>=0?1:-1, even?1:-1, tokens.filter(t=>'+-*/^'.includes(t)).length>=2?1:-1,tokens.includes('^')?1:-1];
  return [{cues:cue,label:even,detail:'Arithmetic = '+String(value)+' · externally computed even-integer target '+String(even),source:input.slice(0,100)}];
}
function parseLanguage(input){
  const words=input.match(/[\p{L}\p{N}']+/gu)||[];
  if(!words.length)throw Error('Enter at least one word.');
  if(words.length>48)throw Error('Up to 48 tokens per encounter are supported.');
  return words.map((word,i)=>{
    const lower=word.toLocaleLowerCase('en');const vowels=(lower.match(/[aeiou]/g)||[]).length;
    const neg=NEGATORS.has(lower),question=QUESTIONS.has(lower);
    return {cues:[lower.length>=5?1:-1,vowels>=2?1:-1,neg?1:-1,question?1:-1],
      label:neg||question,detail:'Token '+(i+1)+'/'+words.length+': '+word+' · negation='+neg+' / interrogative='+question,
      source:word};
  });
}
function setTask(kind){
  stopAuto();state.task=kind;
  for(const [id,on] of [['math-tab',kind==='math'],['language-tab',kind==='language']]){$(id).classList.toggle('active',on);$(id).setAttribute('aria-pressed',String(on));}
  const math=kind==='math';$('stimulus-label').textContent=math?'Arithmetic expression':'A sentence or short passage';
  $('stimulus').placeholder=math?MATH_EXAMPLES[0]:LANGUAGE_EXAMPLES[0];
  $('stimulus').value=math?MATH_EXAMPLES[0]:LANGUAGE_EXAMPLES[0];
  $('mapping').textContent=math?'Arithmetic is evaluated by a small deterministic browser parser (not the tissue). Cues: nonnegative value, even integer, multiple operators, exponent present. External label: even integer.':'A deterministic lexical tokenizer (not the tissue) maps each word to four cues: length ≥5, ≥2 vowels, negation lexicon, question-word lexicon. External label: a negator or question word. No semantic understanding is asserted.';
}
function addEvent(line){state.events.unshift(line);state.events=state.events.slice(0,10);$('events').replaceChildren(...state.events.map(message=>{const p=document.createElement('p');p.textContent=message;return p;}));}
function updateReadout(snapshot,encounter){
  state.last=snapshot;
  const {units,contacts,stats}=snapshot;
  $('tick').textContent='t = '+String(stats.tick).padStart(4,'0');
  $('count-contacts').textContent=String(stats.contacts).padStart(2,'0');
  $('count-growth').textContent='Growth '+stats.growth+' · Pruned '+stats.pruning;
  $('mean-activity').textContent=(units.reduce((s,u)=>s+Math.abs(u.activity),0)/units.length).toFixed(3);
  $('energy').textContent=stats.energy_spent.toLocaleString();
  $('ledger').textContent='Ledger '+stats.ledger.join(' / ')+' (must be 0 / 0)';
  $('prediction').textContent=encounter ? (100*snapshot.prediction).toFixed(1)+'%' : '—';
  $('truth').textContent=encounter?'External target '+String(encounter.label):'Awaiting target';
  $('live-state').textContent='PYTHON TISSUE · '+(encounter?'STEP COMMITTED':'READY');
  $('scene-mode').textContent='LIVE G0-AI · '+stats.contacts+' CONTACTS';
  $('activity-bar').style.width=((stats.tick%40)/40*100)+'%';
  $('source-status').textContent='Python / Pyodide · '+state.mode;
  $('cue-grid').replaceChildren(...((encounter?.cues)||[null,null,null,null]).map((c,i)=>{const el=document.createElement('span');el.textContent='c'+(i+1)+'  '+(c===null?'—':c>0?'+1':'−1');return el;}));
  const select=$('site');
  if(select.options.length!==units.length){select.replaceChildren();units.forEach((u,i)=>{const opt=document.createElement('option');opt.value=String(i);opt.textContent='Site '+String(i).padStart(2,'0');select.append(opt);});}
  select.value=String(state.selected);updateSite();
  if(state.scene)state.scene.setTissue(snapshot);
  if(encounter){
    const mse=(snapshot.prediction-Number(encounter.label))**2;
    addEvent('t'+stats.tick+'  '+encounter.detail+' · p='+snapshot.prediction.toFixed(3)+' · online Brier='+mse.toFixed(4));
    if(snapshot.events.length)for(const evt of snapshot.events.slice(-3)){addEvent('t'+evt.tick+'  '+evt.type+'  '+evt.source+' → '+evt.target);}
  }
}
function updateSite(){
  const u=state.last?.units[state.selected];if(!u)return;
  $('site-state').textContent=JSON.stringify({site:state.selected,activity:u.activity,fast_trace:u.fast_trace,slow_trace:u.slow_trace,novelty:u.novelty,regulome:u.regulome,rule_mix:u.rule_mix,rate:u.rate,atp:u.atp,material:u.material},null,2);
  $('reading-title').textContent='Local site '+String(state.selected).padStart(2,'0');
  $('reading-summary').textContent='This is a synthetic local unit, not an assigned biological organ. Its present activity, fast and slow memories, and regulatory mixture are read directly from the Python source state.';
}
const BRIDGE=`
import json
from __main__ import TelosSubstrate
_esmf = None
def _esmf_start(seed, mode):
    global _esmf
    _esmf = TelosSubstrate(seed=seed, mode=mode, n=20, channels=4)
    return _esmf_snapshot(None)
def _esmf_snapshot(prediction):
    s = _esmf.s
    units = [{"xyz": list(u.xyz), "activity": u.activity,
              "fast_trace": u.fast_trace, "slow_trace": u.slow_trace,
              "novelty": u.novelty, "regulome": u.regulome,
              "rate": u.rate, "rule_mix": list(u.rule_mix),
              "atp": u.atp, "material": u.material} for u in s.units]
    contacts = [{"source": c.source, "target": c.target,
                 "weight": c.weight, "age": c.age, "fitness": c.fitness}
                for c in s.contacts.values()]
    stats = _esmf.integrity()
    stats["tick"] = s.tick
    return json.dumps({"units": units, "contacts": contacts,
                      "stats": stats, "events": s.events[-8:],
                      "prediction": prediction})
def _esmf_step(cues, label):
    global _esmf
    if _esmf is None:
        raise RuntimeError("not initialized")
    pred = _esmf.predict(tuple(cues))
    _esmf.observe(label)
    return _esmf_snapshot(pred)
`;
async function startPython(){
 try{
  status('Downloading local Python engine…');const {loadPyodide}=await import(CDN+'pyodide.mjs');
  state.py=await loadPyodide({indexURL:CDN});
  status('Loading source-pinned G0-AI model…');
  const response=await fetch('./runtime/telos_substrate_ai.py',{cache:'no-store'});if(!response.ok)throw Error('Model source not available: HTTP '+response.status);
  const source=await response.text();
  if(!source.includes('class TelosSubstrate'))throw Error('Model source signature missing');
  await state.py.runPythonAsync('exec('+JSON.stringify(source)+')');
  await state.py.runPythonAsync(BRIDGE);
  state.pyReady=true;
  await resetTissue();
  for(const id of ['run','auto','reset'])$(id).disabled=false;
  $('run').textContent='Process through live Python tissue';
  status('Python G0-AI ready · 20 sites · CPU / WebAssembly · source-pinned snapshot');
 }catch(error){console.error(error);status('Python runtime unavailable: '+error.message+' · Recorded view remains available.',true);$('run').textContent='Python unavailable';$('source-status').textContent='Unavailable';}
}
async function resetTissue(){
 if(!state.pyReady)return;
 stopAuto();state.busy=true;
 try{
  const data=JSON.parse(await state.py.runPythonAsync("_esmf_start("+state.seed+", '"+state.mode+"')"));
  state.events=[];updateReadout(data,null);addEvent('New Python substrate · 20 initially unassigned local units · seed '+state.seed);
 }catch(error){status('Reset failed: '+error.message,true);}finally{state.busy=false;}
}
async function runEncounter(){
 if(!state.pyReady||state.busy)return;
 state.busy=true;$('run').disabled=true;
 try{
  const encounters=state.task==='math'?parseArithmetic($('stimulus').value):parseLanguage($('stimulus').value);
  for(const encounter of encounters){
   if(!Array.isArray(encounter.cues)||encounter.cues.length!==4||encounter.cues.some(v=>v!==1&&v!==-1))throw Error('Invalid four-channel cue');
   const python="_esmf_step(("+encounter.cues.join(',')+"), "+(encounter.label?'True':'False')+")";
   const snapshot=JSON.parse(await state.py.runPythonAsync(python));
   if(snapshot.stats.ledger.some(v=>v!==0))throw Error('Python resource ledger failed');
   updateReadout(snapshot,encounter);
  }
  status(encounters.length+' local Python transition(s) processed · no network inference');
 }catch(error){console.error(error);status('Encounter rejected: '+error.message,true);stopAuto();}
 finally{state.busy=false;$('run').disabled=!state.pyReady;}
}
function stopAuto(){if(state.autoTimer){clearInterval(state.autoTimer);state.autoTimer=null;}$('auto').textContent='Start encounter stream';}
function toggleAuto(){
 if(state.autoTimer){stopAuto();return;}
 if(!state.pyReady)return;
 $('auto').textContent='Stop encounter stream';
 state.autoTimer=setInterval(()=>{if(state.busy)return;const examples=state.task==='math'?MATH_EXAMPLES:LANGUAGE_EXAMPLES;state.demoIndex=(state.demoIndex+1)%examples.length;$('stimulus').value=examples[state.demoIndex];void runEncounter();},1200);
}
function createTissueScene(container,onSelect){
 const scene=new THREE.Scene();scene.background=new THREE.Color('#06121c');scene.fog=new THREE.FogExp2('#06121c',.045);
 const camera=new THREE.PerspectiveCamera(47,1,.08,100);camera.position.set(4.6,3.4,7.8);
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'high-performance'});renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.23;container.append(renderer.domElement);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.055;controls.minDistance=2.5;controls.maxDistance=16;controls.enablePan=false;controls.autoRotate=false;
 scene.add(new THREE.AmbientLight('#b2d9ec',.56));
 for(const [c,x,y,z,p] of [['#8cf7d4',3,5,4,70],['#aa8dff',-4,-3,-2,55],['#f4cc98',0,3,-5,25]]){const light=new THREE.PointLight(c,p,20);light.position.set(x,y,z);scene.add(light);}
 const world=new THREE.Group();scene.add(world);
 const ringMaterial=new THREE.MeshBasicMaterial({color:'#386878',transparent:true,opacity:.12,side:THREE.DoubleSide,depthWrite:false});
 for(const [r,rx,ry] of [[2.1,.3,.6],[2.65,1.1,.1],[3.2,.9,1.6]]){const ring=new THREE.Mesh(new THREE.TorusGeometry(r,.004,2,180),ringMaterial);ring.rotation.set(rx,ry,0);world.add(ring);}
 const starGeo=new THREE.BufferGeometry(),starPositions=[];for(let i=0;i<650;i++){const t=(i*2.399963),r=4.2+Math.sqrt(i/650)*10;starPositions.push(Math.cos(t)*r,Math.sin(t*.77)*r*.8,Math.sin(t)*r);}starGeo.setAttribute('position',new THREE.Float32BufferAttribute(starPositions,3));scene.add(new THREE.Points(starGeo,new THREE.PointsMaterial({color:'#607988',size:.019,transparent:true,opacity:.45,depthWrite:false})));
 const nodes=[],halos=[],pickables=[],links=new Map(),pulses=[];const materialPos=new THREE.Color('#7eead5'),materialNeg=new THREE.Color('#ac95f0');
 const nodeGeo=new THREE.IcosahedronGeometry(.105,2);const glowGeo=new THREE.SphereGeometry(.16,18,12);
 for(let i=0;i<20;i++){
  const mat=new THREE.MeshStandardMaterial({color:'#8bdac8',emissive:'#5fe1bf',emissiveIntensity:.6,metalness:.15,roughness:.22});
  const mesh=new THREE.Mesh(nodeGeo,mat);mesh.userData.index=i;world.add(mesh);nodes.push(mesh);pickables.push(mesh);
  const halo=new THREE.Mesh(glowGeo,new THREE.MeshBasicMaterial({color:'#f1b882',transparent:true,opacity:.07,depthWrite:false,blending:THREE.AdditiveBlending}));world.add(halo);halos.push(halo);
 }
 const pulseGeo=new THREE.SphereGeometry(.022,8,6);
 for(let i=0;i<34;i++){const mesh=new THREE.Mesh(pulseGeo,new THREE.MeshBasicMaterial({color:i%2?'#a7b9ff':'#8cecd5',transparent:true,opacity:.0,depthWrite:false,blending:THREE.AdditiveBlending}));world.add(mesh);pulses.push(mesh);}
 const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();renderer.domElement.addEventListener('pointerdown',event=>{pointer.x=event.clientX;pointer.y=event.clientY;});renderer.domElement.addEventListener('pointerup',event=>{if(Math.abs(event.clientX-pointer.x)>6||Math.abs(event.clientY-pointer.y)>6)return;const rect=renderer.domElement.getBoundingClientRect();raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),camera);const hit=raycaster.intersectObjects(pickables)[0];if(hit)onSelect(hit.object.userData.index);});
 let previous=null;const colors={positive:new THREE.Color('#70f0d8'),negative:new THREE.Color('#b99dff')};
 const bloomEnabled=renderer.capabilities.isWebGL2 && !matchMedia('(prefers-reduced-motion: reduce)').matches;
 let composer=null;
 try{if(bloomEnabled){composer=new EffectComposer(renderer);composer.addPass(new RenderPass(scene,camera));composer.addPass(new UnrealBloomPass(new THREE.Vector2(800,600),.72,.55,.26));}}catch(error){console.warn('Bloom disabled:',error);composer=null;}
 const clock=new THREE.Clock();
 const resize=()=>{const width=Math.max(container.clientWidth,320),height=Math.max(container.clientHeight,320);camera.aspect=width/height;camera.updateProjectionMatrix();renderer.setSize(width,height,false);composer?.setSize(width,height);};new ResizeObserver(resize).observe(container);resize();
 function setTissue(snapshot){
  previous=snapshot;const {units,contacts}=snapshot;
  units.forEach((u,i)=>{const mesh=nodes[i],halo=halos[i];mesh.position.set(...u.xyz.map(v=>v*2.15));halo.position.copy(mesh.position);
   const t=THREE.MathUtils.clamp((u.activity+1)/2,0,1);mesh.material.color.copy(colors.negative).lerp(colors.positive,t);mesh.material.emissive.copy(mesh.material.color);
   const scale=.85+Math.abs(u.activity)*.95+(i===state.selected ? .24 : 0);mesh.scale.setScalar(scale);
   halo.scale.setScalar(.8+Math.abs(u.slow_trace)*4);halo.material.opacity=state.showSlow? .05+Math.abs(u.slow_trace)*.18 : 0;
  });
  const active=new Set();contacts.forEach(c=>{const key=c.source+'-'+c.target;active.add(key);const positive=c.weight>=0;
   let edge=links.get(key);
   if(!edge){const geo=new THREE.BufferGeometry().setFromPoints([nodes[c.source].position,nodes[c.target].position]);const mat=new THREE.LineBasicMaterial({color:positive?'#67dfcb':'#ac96ed',transparent:true,opacity:.25,depthWrite:false});edge=new THREE.Line(geo,mat);world.add(edge);links.set(key,edge);}
   else edge.geometry.setFromPoints([nodes[c.source].position,nodes[c.target].position]);
   edge.visible=state.showContacts;edge.material.color.set(positive?'#67dfcb':'#ac96ed');edge.material.opacity=.17+.65*Math.min(1,Math.abs(c.weight));
  });
  for(const [key,edge] of links){if(!active.has(key)){world.remove(edge);edge.geometry.dispose();edge.material.dispose();links.delete(key);}}
 }
 function frame(){
  requestAnimationFrame(frame);const elapsed=clock.getElapsedTime();
  if(state.motion){world.rotation.y+=.00055;}
  pulses.forEach((p,i)=>{if(!previous||!state.motion||!state.showContacts||!previous.contacts.length){p.material.opacity=0;return;}
   const c=previous.contacts[i%previous.contacts.length],a=nodes[c.source].position,b=nodes[c.target].position,t=(elapsed*(.27+((i%5)*.09))+i*.13)%1;
   p.position.lerpVectors(a,b,t);p.material.opacity=.12+.5*Math.sin(t*Math.PI);});
  controls.update();if(composer)composer.render();else renderer.render(scene,camera);
 }
 frame();
 return {setTissue,updateOptions(){if(previous)setTissue(previous);},select(index){state.selected=index;if(previous)setTissue(previous);}};
}
async function init(){
 $('math-tab').addEventListener('click',()=>setTask('math'));
 $('language-tab').addEventListener('click',()=>setTask('language'));
 $('example').addEventListener('click',()=>{const list=state.task==='math'?MATH_EXAMPLES:LANGUAGE_EXAMPLES;state.demoIndex=(state.demoIndex+1)%list.length;$('stimulus').value=list[state.demoIndex];});
 $('run').addEventListener('click',()=>void runEncounter());$('auto').addEventListener('click',toggleAuto);$('reset').addEventListener('click',()=>void resetTissue());
 $('site').addEventListener('change',()=>{state.selected=Number($('site').value);updateSite();state.scene?.select(state.selected);});
 for(const [id,key] of [['slow','showSlow'],['contacts','showContacts'],['motion','motion']])$(id).addEventListener('change',()=>{state[key]=$(id).checked;state.scene?.updateOptions();});
 $('motion').checked=state.motion;
 try{state.scene=createTissueScene($('viewport'),index=>{state.selected=index;$('site').value=String(index);updateSite();state.scene?.select(index);});$('fallback').hidden=true;}
 catch(error){console.warn('3D fallback:',error);$('fallback-message').textContent='WebGL unavailable. The live Python readout still works.';}
 await startPython();
}
init();
