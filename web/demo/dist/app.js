import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import loadMujoco from './vendor/mujoco.js';
const metadataName='metadata.json';
import {Network,cat,floats,bytes,observation,inverse} from './inference-onnx.js';
const $=id=>document.getElementById(id),titles={g1:'Unitree G1',h2:'Unitree H2',a3:'Agibot A3'};
function checkpointLabel(m){return m.mixed_checkpoints?`policy ${m.policy_iterations.toLocaleString()} / world ${m.world_iterations.toLocaleString()} (mixed)`:`checkpoint ${m.completed_iterations.toLocaleString()}`;}
const notice=(s,error=false)=>{$('notice').textContent=s;$('notice').classList.toggle('error',error);};
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0xffffff,0);$('viewport').append(renderer.domElement);
const scene=new THREE.Scene();scene.up.set(0,0,1);const camera=new THREE.PerspectiveCamera(38,1,.01,120);camera.up.set(0,0,1);camera.position.set(4,-9,4.5);const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,0,.85);controls.enableDamping=true;
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
scene.fog=new THREE.Fog(0xf5f5f7,7,17);
scene.add(new THREE.HemisphereLight(0xffffff,0x999999,2.1));
const light=new THREE.DirectionalLight(0xffffff,2.8);light.position.set(-3,-4,9);light.castShadow=true;light.shadow.mapSize.set(2048,2048);Object.assign(light.shadow.camera,{left:-5,right:5,top:5,bottom:-5});light.shadow.bias=-.0005;light.shadow.normalBias=.02;light.shadow.radius=4;scene.add(light);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.ShadowMaterial({opacity:.15}));ground.position.z=-.008;ground.receiveShadow=true;scene.add(ground);
const grid=new THREE.GridHelper(40,40,0xd2d2d7,0xd2d2d7);grid.rotation.x=Math.PI/2;grid.material.transparent=true;grid.material.opacity=.38;scene.add(grid);
function fitCamera(){const width=$('viewport').clientWidth,height=$('viewport').clientHeight;renderer.setSize(width,height);camera.aspect=width/height;const d=Math.max(5.2,3.1/camera.aspect);camera.position.set(d*.6,-d*.8,d*.46);controls.target.set(0,0,.8);camera.updateProjectionMatrix();}
new ResizeObserver(fitCamera).observe($('viewport'));
let showWorld=true,showGoal=false,showSim=false;
let mj,device,net,meta,model,physical,imaginary,target,groups=[],clip,frame=0,worldState,previous,worldPrevious,history,worldHistory,step=0,playing=true,busy=false,loading=false,ready=false,currentRobot='g1',lastTick=0,avg=0,force=[0,0,0],forceSteps=0;
const v3=new THREE.Vector3(),q4=new THREE.Quaternion();
function putPose(data,s){data.qpos.set(s.slice(0,7));meta.qpos_indices.forEach((qi,i)=>data.qpos[qi]=s[13+i]);data.qvel.set(s.slice(7,10),0);data.qvel.set(inverse(s.slice(3,7),s.slice(10,13)),3);meta.qvel_indices.forEach((qi,i)=>data.qvel[qi]=s[13+meta.layout.joints+i]);if(currentRobot==='a3')for(const name of ['head_yaw_joint','head_pitch_joint']){const i=meta.joint_names.indexOf(name);if(i>=0){data.qpos[meta.qpos_indices[i]]=0;data.qvel[meta.qvel_indices[i]]=0;}}mj.mj_forward(model,data);}
function getState(data){const result=new Float32Array(meta.layout.dim),j=meta.layout.joints;function body(id){const p=data.xpos.slice(3*id,3*id+3),q=data.xquat.slice(4*id,4*id+4),c=data.cvel.slice(6*id,6*id+6),root=meta.body_root_ids[id],com=data.subtree_com.slice(3*root,3*root+3),o=p.map((v,i)=>v-com[i]);const linear=[c[3]+c[1]*o[2]-c[2]*o[1],c[4]+c[2]*o[0]-c[0]*o[2],c[5]+c[0]*o[1]-c[1]*o[0]];return cat(p,q,linear,c.slice(0,3));}result.set(body(meta.root_body_id));meta.qpos_indices.forEach((qi,i)=>result[13+i]=data.qpos[qi]);meta.qvel_indices.forEach((qi,i)=>result[13+j+i]=data.qvel[qi]);meta.tracked_body_ids.forEach((id,i)=>result.set(body(id),13+2*j+13*i));return result;}
function stateAt(i){const d=meta.layout.dim;return clip.states.slice(Math.min(i,clip.frames-1)*d,(Math.min(i,clip.frames-1)+1)*d);}
function reset(at=0){if(!ready)return;frame=at;step=0;previous=new Float32Array(meta.layout.joints);worldPrevious=previous.slice();worldState=stateAt(frame);for(const d of [physical,imaginary,target]){mj.mj_resetData(model,d);putPose(d,worldState);}history=cat(...Array(10).fill(observation(worldState,previous,meta,stateAt(frame))));worldHistory=history.slice();forceSteps=0;physical.xfrc_applied.fill(0);updateViews();}
async function makeGeometry(base){const geometries=new Map();for(const x of meta.render.meshes){const [vert,face]=await Promise.all([floats(base+x.vertices.file,x.vertices.dtype,x.vertices.compression),bytes(base+x.faces.file,x.faces.compression).then(b=>new Uint32Array(b))]);const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(vert,3));g.setIndex(new THREE.BufferAttribute(face,1));g.computeVertexNormals();geometries.set(x.id,g);}
const visualGroup=meta.render.geoms.some(g=>g.group===1)?1:2;const included=meta.render.geoms.filter(g=>g.body>0&&g.rgba[3]>0&&g.group===visualGroup);for(const [idx,color] of [0x8c9c90,0x37a8bc,0xce8654].entries()){const root=new THREE.Group();root.userData.geometries=[];for(const g of included){let geo;switch(g.type){case 7:geo=geometries.get(g.mesh);break;case 2:geo=new THREE.SphereGeometry(g.size[0],16,12);break;case 3:geo=new THREE.CapsuleGeometry(g.size[0],2*g.size[1],6,12);geo.rotateX(Math.PI/2);break;case 5:geo=new THREE.CylinderGeometry(g.size[0],g.size[0],2*g.size[1],16);geo.rotateX(Math.PI/2);break;case 6:geo=new THREE.BoxGeometry(...g.size.map(x=>2*x));break;case 4:geo=new THREE.SphereGeometry(1,16,12);geo.scale(...g.size);break;default:continue;}if(!geo)continue;const material=new THREE.MeshStandardMaterial({color:new THREE.Color(...g.rgba.slice(0,3)),metalness:.15,roughness:.55,transparent:idx===0,opacity:idx===0?.22:1,depthWrite:idx!==0});const mesh=new THREE.Mesh(geo,material);mesh.castShadow=idx!==0;mesh.receiveShadow=true;mesh.userData.spec=g;mesh.userData.lane=idx;root.add(mesh);}scene.add(root);groups.push(root);}}
function draw(root,data){for(const mesh of root.children){const g=mesh.userData.spec,id=g.body;q4.set(data.xquat[4*id+1],data.xquat[4*id+2],data.xquat[4*id+3],data.xquat[4*id]);mesh.position.fromArray(g.pos).applyQuaternion(q4).add(v3.fromArray(data.xpos,3*id));mesh.quaternion.copy(q4).multiply(new THREE.Quaternion(g.quat[1],g.quat[2],g.quat[3],g.quat[0]));}}
function displayLayers(){$('worldToggle').setAttribute('aria-pressed',String(showWorld));$('refToggle').setAttribute('aria-pressed',String(showGoal));$('simToggle').setAttribute('aria-pressed',String(showSim));$('dragHint').hidden=!showWorld&&!showSim;if(groups.length!==3)return;groups[0].visible=showGoal;groups[1].visible=showWorld;groups[2].visible=showSim;document.body.classList.toggle('sim-on',showSim);$('refToggle').setAttribute('aria-pressed',String(showGoal));$('simToggle').setAttribute('aria-pressed',String(showSim));for(const mesh of groups[1].children){mesh.material.color.set(showSim?0x5da1e5:new THREE.Color(...mesh.userData.spec.rgba.slice(0,3)));}updateViews();}
function updateViews(){if(!ready)return;putPose(target,stateAt(frame));putPose(imaginary,worldState);const origin=showWorld?worldState:showSim?physical.qpos:stateAt(frame);[target,imaginary,physical].forEach((d,i)=>{groups[i].position.set(-origin[0],-origin[1],0);draw(groups[i],d);});$('depth').textContent=`STEP ${step.toLocaleString()} SINCE THE LAST RESET`;const root=meta.root_body_id;const hintData=showWorld?imaginary:physical;const hintGroup=showWorld?groups[1]:groups[2];const hint=new THREE.Vector3().fromArray(hintData.xpos,3*root).add(hintGroup.position);hint.z=0;hint.project(camera);$('dragHint').style.left=`${Math.max(20,Math.min(80,(hint.x+1)*50))}%`;$('dragHint').style.top=`${Math.max(35,Math.min(78,(1-hint.y)*50+3))}%`;}
function syncPlay(){$('play').textContent=playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',playing?'Pause':'Play');}
async function tick(){if(!ready||loading||busy||!playing)return;busy=true;const start=performance.now();try{if(frame>=clip.playable_frames-1){reset();return;}const goals=clip.states.slice(frame*meta.layout.dim,(frame+11)*meta.layout.dim);let action=$('policy').checked?await net.action(history,goals):new Float32Array(meta.layout.joints);const noise=+$('noise').value;action=action.map(v=>Math.max(-20,Math.min(20,v+noise*Math.sqrt(-2*Math.log(Math.max(1e-8,Math.random())))*Math.cos(2*Math.PI*Math.random()))));if(step%32===0){worldState=getState(physical);worldHistory=history.slice();worldPrevious=previous.slice();}if(forceSteps>0){const mass=model.body_subtreemass[meta.root_body_id];const dv=force.map(f=>f*meta.control_dt/Math.max(mass,1));for(const offset of [0,...meta.tracked_body_ids.map((_,i)=>13+2*meta.layout.joints+13*i)])for(let j=0;j<3;j++)worldState[offset+7+j]+=dv[j];worldHistory.set(observation(worldState,worldPrevious,meta,stateAt(frame)),worldHistory.length-meta.layout.obs_dim);}let wa=$('policy').checked?await net.action(worldHistory,goals):new Float32Array(meta.layout.joints);wa=wa.map(v=>Math.max(-20,Math.min(20,v+noise*Math.sqrt(-2*Math.log(Math.max(1e-8,Math.random())))*Math.cos(2*Math.PI*Math.random()))));worldState=await net.world(worldState,wa,meta.context,meta);for(let i=0;i<meta.layout.joints;i++)physical.ctrl[i]=meta.default_pose[i]+meta.action_scale[i]*action[i];physical.xfrc_applied.fill(0);if(forceSteps>0){physical.xfrc_applied.set(force,6*meta.root_body_id);forceSteps--;}for(let i=0;i<meta.substeps;i++)mj.mj_step(model,physical);mj.mj_forward(model,physical);const s=getState(physical);if([...s,...worldState].some(x=>!Number.isFinite(x))){playing=false;syncPlay();notice('Non-finite state. Paused — reset to restart.',true);return;}history=cat(history.slice(meta.layout.obs_dim),observation(s,action,meta,stateAt(frame+1)));worldHistory=cat(worldHistory.slice(meta.layout.obs_dim),observation(worldState,wa,meta,stateAt(frame+1)));previous=action;worldPrevious=wa;frame++;step++;updateViews();avg=avg*.9+(performance.now()-start)*.1;}catch(e){playing=false;syncPlay();notice(e.message,true);console.error(e);}finally{busy=false;}}
function animate(now){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera);if(now-lastTick>20){lastTick=now;tick();}}requestAnimationFrame(animate);
const motionName=item=>item.id.replace('_subject',' · subject ');
async function loadMotion(item){notice('Loading motion · '+motionName(item));const states=await floats(`assets/${currentRobot}/${item.state.file}`,item.state.dtype,item.state.compression);if(states.length!==item.frames*meta.layout.dim)throw Error('Motion data size mismatch');clip={...item,states};$('motionLabel').textContent=motionName(item);notice('');document.querySelectorAll('#motions button').forEach(b=>b.classList.toggle('selected',b.dataset.motion===item.id));reset();}
function listMotions(){const search=$('filter').value.toLowerCase();$('motions').replaceChildren();for(const motion of meta.motions.filter(m=>(m.id+' '+m.label).toLowerCase().includes(search))){const b=document.createElement('button');b.textContent=motionName(motion);b.title=`${(motion.playable_frames/motion.fps).toFixed(1)} s · ${motion.playable_frames.toLocaleString()} frames`;b.dataset.motion=motion.id;b.classList.toggle('selected',clip?.id===motion.id);b.onclick=async()=>{if(loading)return;loading=true;while(busy)await new Promise(r=>setTimeout(r,8));try{await loadMotion(motion);closePanels();}catch(e){notice(e.message,true);}finally{loading=false;}};$('motions').append(b);}}
async function chooseRobot(robot){if(loading)return;loading=true;while(busy)await new Promise(r=>setTimeout(r,8));ready=false;currentRobot=robot;$('motions').replaceChildren();$('checkpoint').textContent='';$('motionLabel').textContent='Motions';$('device').disabled=true;notice('Loading '+titles[robot]+'…');try{await net?.dispose();for(const root of groups){scene.remove(root);root.children.forEach(x=>{x.material.dispose();x.geometry.dispose();});}groups=[];for(const d of [physical,imaginary,target])d?.delete();model?.delete();const base=`assets/${robot}/`;const resp=await fetch(base+metadataName,{cache:'no-store'});if(!resp.ok)throw Error('Robot assets are unavailable');meta=await resp.json();mj??=await loadMujoco();const dir='/'+robot;mj.FS.mkdirTree(dir);for(const file of [meta.model.xml,...meta.model.files]){const packed=meta.model.compressed_files?.[file];const contents=new Uint8Array(await bytes(base+(packed||file),packed?'gzip':undefined));const path=dir+'/'+file;mj.FS.mkdirTree(path.slice(0,path.lastIndexOf('/')));mj.FS.writeFile(path,contents);}model=mj.MjModel.mj_loadXML(dir+'/'+meta.model.xml);physical=new mj.MjData(model);imaginary=new mj.MjData(model);target=new mj.MjData(model);await makeGeometry(base);net=new Network(device);await net.load(meta,base,s=>notice('Loading network · '+s));const fixture=await fetch(base+meta.inference_fixture).then(r=>r.json());const a=await net.action(new Float32Array(fixture.history),new Float32Array(fixture.goals)),w=await net.world(new Float32Array(fixture.state),a,meta.context,meta);const err=Math.max(...a.map((v,i)=>Math.abs(v-fixture.action[i])),...w.map((v,i)=>Math.abs(v-fixture.world_next[i])));if(err>.005)throw Error('Inference validation failed: '+err);$('checkpoint').textContent=`LaFAN · UMR · ${meta.motions.length} motions · train40 / test37 · ${titles[robot]} · ${checkpointLabel(meta)} · ${meta.checkpoint_stage==='numerical-validation'?'numerical validation':'early training'}`;$('checkpoint').title=`Fixture max error ${err.toExponential(2)}; MuJoCo ${meta.mujoco_version}`;$('friction').value=1;paintSlider($('friction'));$('frictionValue').value='50%';if(!meta.motions?.length)throw Error('Motion data is not included. Add compatible licensed motions with scripts.export_web_motions; see the browser demo guide.');ready=true;await loadMotion(meta.motions[0]);listMotions();displayLayers();playing=true;syncPlay();notice('');}catch(e){console.error(e);notice(e.message,true);}finally{loading=false;$('device').disabled=false;}}
function closePanels(){$('motionPanel').hidden=true;$('motionsBtn').setAttribute('aria-expanded','false');$('ctx').classList.remove('open');$('ctxBtn').setAttribute('aria-expanded','false');}
function paintSlider(el){el.style.setProperty('--p',((+el.value- +el.min)/(+el.max- +el.min)*100)+'%');}
$('play').onclick=()=>{playing=!playing;syncPlay();};
$('reset').onclick=async()=>{if(loading)return;loading=true;while(busy)await new Promise(r=>setTimeout(r,8));reset();notice('');loading=false;};
$('filter').oninput=()=>{if(meta)listMotions();};
$('noise').oninput=()=>{$('noiseValue').value=(+$('noise').value).toFixed(2);paintSlider($('noise'));};
$('friction').oninput=()=>{const f=+$('friction').value;$('frictionValue').value=Math.round((f-.3)/1.4*100)+'%';paintSlider($('friction'));if(model){for(let i=0;i<model.ngeom;i++)model.geom_friction[3*i]=f;meta.context[0]=f;}};
$('device').onchange=()=>chooseRobot($('device').value);
$('worldToggle').onclick=()=>{showWorld=!showWorld;displayLayers();};
$('refToggle').onclick=()=>{showGoal=!showGoal;displayLayers();};
$('simToggle').onclick=()=>{showSim=!showSim;displayLayers();};
$('motionsBtn').onclick=()=>{const open=$('motionPanel').hidden;closePanels();$('motionPanel').hidden=!open;$('motionsBtn').setAttribute('aria-expanded',String(open));if(open)$('filter').focus();};
$('ctxBtn').onclick=()=>{const open=!$('ctx').classList.contains('open');closePanels();$('ctx').classList.toggle('open',open);$('ctxBtn').setAttribute('aria-expanded',String(open));};
for(const el of [$('noise'),$('friction')])paintSlider(el);
document.addEventListener('keydown',e=>{if(e.target.matches('input,select,button')&&e.code!=='Escape')return;const actions={Space:'play',KeyR:'reset',KeyW:'worldToggle',KeyG:'refToggle',KeyS:'simToggle'};if(actions[e.code]){e.preventDefault();$(actions[e.code]).click();}else if(e.code==='KeyP'){$('policy').checked=!$('policy').checked;}else if(e.code==='Escape')closePanels();});
const ray=new THREE.Raycaster();let drag=null;
// Capture before OrbitControls so grabbing a robot does not orbit the camera.
renderer.domElement.addEventListener('pointerdown',e=>{closePanels();if(!ready||loading)return;const rect=renderer.domElement.getBoundingClientRect();ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);if(ray.intersectObjects(groups.filter(g=>g.visible).flatMap(g=>g.children)).length){drag=[e.clientX,e.clientY];controls.enabled=false;renderer.domElement.setPointerCapture(e.pointerId);$('dragHint').style.opacity='0';}},true);
renderer.domElement.addEventListener('pointermove',e=>{if(!drag)return;const dx=(e.clientX-drag[0])*3,dy=(e.clientY-drag[1])*3;const right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1);force=[Math.max(-200,Math.min(200,dx*right.x-dy*up.x)),Math.max(-200,Math.min(200,dx*right.y-dy*up.y)),0];forceSteps=2;});
function releaseDrag(){drag=null;controls.enabled=true;forceSteps=0;}
renderer.domElement.addEventListener('pointerup',releaseDrag);renderer.domElement.addEventListener('pointercancel',releaseDrag);renderer.domElement.addEventListener('lostpointercapture',releaseDrag);
try{
 if(!window.isSecureContext)throw Error('WebGPU requires trusted HTTPS. Install the provided CA certificate, then reopen the HTTPS URL. CPU fallback is disabled.');
 const adapter=await navigator.gpu?.requestAdapter();
 if(!adapter)throw Error('WebGPU is unavailable. Use a WebGPU-capable browser with hardware acceleration enabled. CPU fallback is disabled.');
 device=await adapter.requestDevice();
 device.lost.then(info=>{ready=false;playing=false;syncPlay();notice('WebGPU device lost: '+info.message+'. Reload to reconnect. CPU fallback is disabled.',true);});
 document.documentElement.dataset.inference='onnxruntime-webgpu-f32';
 await chooseRobot('g1');
}catch(e){notice(e.message,true);$('device').disabled=true;$('play').disabled=true;$('reset').disabled=true;console.error(e);}


// Check lightweight metadata; only download weights when the published version changes.
let checkingCheckpoint=false;
async function checkCheckpoint(){
 if(checkingCheckpoint||loading||!ready||document.hidden)return;
 checkingCheckpoint=true;
 const robot=currentRobot,oldMeta=meta;
 let candidate=null,locked=false,wasPlaying=playing;
 try{
  const base=`assets/${robot}/`;
  const response=await fetch(base+metadataName,{cache:'no-store'});
  if(!response.ok)throw Error('Checkpoint metadata unavailable');
  const next=await response.json();
  if(currentRobot!==robot||meta!==oldMeta||loading)return;
  if(next.inference_fixture===oldMeta.inference_fixture)return;
  if(JSON.stringify(next.joint_names)!==JSON.stringify(oldMeta.joint_names)||JSON.stringify(next.body_names)!==JSON.stringify(oldMeta.body_names))throw Error('Checkpoint robot schema mismatch');
  // A physics migration needs a new MuJoCo model, not only new network weights.
  if(next.model.sha256!==oldMeta.model.sha256){await chooseRobot(robot);return;}
  loading=true;locked=true;wasPlaying=playing;$('device').disabled=true;
  while(busy)await new Promise(r=>setTimeout(r,8));
  notice('Updating checkpoint · '+next.completed_iterations.toLocaleString());
  candidate=new Network(device);
  await candidate.load(next,base);
  const fixtureResponse=await fetch(base+next.inference_fixture);
  if(!fixtureResponse.ok)throw Error('Checkpoint validation fixture unavailable');
  const f=await fixtureResponse.json();
  const a=await candidate.action(new Float32Array(f.history),new Float32Array(f.goals));
  const w=await candidate.world(new Float32Array(f.state),a,next.context,next);
  const err=Math.max(...a.map((v,i)=>Math.abs(v-f.action[i])),...w.map((v,i)=>Math.abs(v-f.world_next[i])));
  if(!Number.isFinite(err)||err>.005)throw Error('Checkpoint validation failed: '+err);
  next.context[0]=+$('friction').value;
  const old=net;net=candidate;candidate=null;meta=next;await old.dispose();
  reset();playing=wasPlaying;syncPlay();
  $('checkpoint').textContent=`LaFAN · UMR · ${meta.motions.length} motions · train40 / test37 · ${titles[robot]} · ${checkpointLabel(meta)} · early training`;
  $('checkpoint').title=`Fixture max error ${err.toExponential(2)}; auto update enabled`;
  notice('');
 }catch(e){console.error('Checkpoint update: keeping current model',e);if(locked)notice('Checkpoint update failed; keeping current model. Retrying automatically.',true);}
 finally{await candidate?.dispose();if(locked){loading=false;$('device').disabled=false;}checkingCheckpoint=false;}
}
setInterval(checkCheckpoint,60000);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)checkCheckpoint();});
