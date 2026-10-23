// Validate the released policy using ONNX Runtime Node; browser GPU testing is separate.
import fs from 'node:fs/promises';
import path from 'node:path';
import * as ort from '../web/demo/node_modules/onnxruntime-node/dist/index.js';
const root=path.resolve(process.argv[2]);
const meta=JSON.parse(await fs.readFile(path.join(root,'metadata.json')));
const fixtures=JSON.parse(await fs.readFile(path.join(root,'inference_fixture.json')));
const session=await ort.InferenceSession.create(path.join(root,'policy.onnx'),{executionProviders:['cpu'],intraOpNumThreads:2});
let maxAbs=0;
for(const f of fixtures){
 const result=await session.run({history:new ort.Tensor('float32',Float32Array.from(f.history.flat(2)),meta.inputs.history),goal_features:new ort.Tensor('float32',Float32Array.from(f.goal_features.flat()),meta.inputs.goal_features)});
 for(let i=0;i<result.action.data.length;i++){
  const error=Math.abs(result.action.data[i]-f.action[0][i]);
  if(!Number.isFinite(error)||error>2e-4+2e-4*Math.abs(f.action[0][i]))throw Error('Policy parity failed');
  maxAbs=Math.max(maxAbs,error);
 }
}
await session.release();
console.log(JSON.stringify({robot:meta.robot,checks:fixtures.length,node_onnx_max_abs:maxAbs}));
