// Dataset-free ONNX CPU fixture for browser runtime/packing smoke tests.
// This is not an independent PyTorch reference or a locomotion-quality test.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Network, observation, goalFeatures, fkState, cat} from '../web/demo/dist/inference-onnx.js';
import * as ort from '../web/demo/node_modules/onnxruntime-node/dist/index.js';
const root=path.resolve(process.argv[2]);
globalThis.fetch=async url=>new Response(await fs.readFile(new URL(url)));
const m=JSON.parse(await fs.readFile(path.join(root,'metadata.json')));
const n=new Network(null,{runtime:ort,offlineValidation:true});
await n.load(m,pathToFileURL(root+'/').href);
const state=fkState(Float32Array.from([0,0,1,1,0,0,0,0,0,0,0,0,0]),m.default_pose,new Float32Array(m.layout.joints),m.kinematics);
const obs=observation(state,new Float32Array(m.layout.joints),m,state);
const history=cat(...Array.from({length:m.history_length},()=>obs));
const goals=cat(...Array.from({length:11},()=>state));
const action=await n.action(history,goals);
const world=await n.world(state,action,m.context,m);
const fixture={reference:'Synthetic rest pose; ONNX Runtime CPU smoke test',state:[...state],history:[...history],goals:[...goals],action:[...action],world_next:[...world],observation:[...obs],goal_features:[...goalFeatures(goals,m)],rollout:[]};
let s=state,h=history;
for(let i=0;i<32;i++){
 const a=await n.action(h,goals);s=await n.world(s,a,m.context,m);
 if([...s,...a].some(v=>!Number.isFinite(v)))throw Error('Non-finite synthetic rollout');
 fixture.rollout.push({action:[...a],state:[...s]});
 h=cat(h.slice(m.layout.obs_dim),observation(s,a,m,state));
}
await n.dispose();
await fs.writeFile(path.join(root,m.inference_fixture),JSON.stringify(fixture));
console.log(JSON.stringify({robot:m.robot,steps:32,fixture:m.inference_fixture}));
