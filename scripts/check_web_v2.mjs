// Offline arithmetic/packing parity; production browser requires WebGPU.
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import {Network,observation,goalFeatures,fkState} from '../web/demo/dist/inference-onnx.js';
const root=path.resolve(process.argv[2]);
globalThis.fetch=async url=>new Response(await fs.readFile(new URL(url)));
const m=JSON.parse(await fs.readFile(process.argv[3]||path.join(root,'metadata.json'))),f=JSON.parse(await fs.readFile(path.join(root,m.inference_fixture)));
const ort=await import('../web/demo/node_modules/onnxruntime-node/dist/index.js');
const n=new Network(null,{runtime:ort,offlineValidation:true});await n.load(m,pathToFileURL(root+'/').href);
const a=await n.action(new Float32Array(f.history),new Float32Array(f.goals));
const w=await n.world(new Float32Array(f.state),a,m.context,m);
const err=(x,y)=>Math.max(...x.map((v,i)=>Math.abs(v-y[i])));
const results={action_error:err(a,f.action),world_error:err(w,f.world_next)};
if(f.goal_features)results.goal_features_error=err(goalFeatures(f.goals,m),f.goal_features);
if(f.observation)results.observation_error=err(observation(f.state,new Float32Array(m.layout.joints),m,f.state),f.observation);
if(Object.values(results).some(v=>!Number.isFinite(v)||v>.005))throw Error(JSON.stringify(results));
let h=new Float32Array(f.history),s=new Float32Array(f.state),maxRollout=0;
for(const row of f.rollout||[]){
 const a=await n.action(h,new Float32Array(f.goals));s=await n.world(s,a,m.context,m);
 maxRollout=Math.max(maxRollout,err(s,row.state),err(a,row.action));
 h=Float32Array.from([...h.slice(m.layout.obs_dim),...observation(s,a,m,new Float32Array(f.goals.slice(m.layout.dim,2*m.layout.dim)))]);
}
results.rollout32_max_abs=maxRollout;
if(!Number.isFinite(maxRollout)||maxRollout>.01)throw Error(JSON.stringify(results));
await n.dispose();console.log(JSON.stringify(results));
