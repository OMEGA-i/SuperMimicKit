import {mkdir,copyFile} from 'node:fs/promises';
await mkdir('dist/vendor',{recursive:true});
for(const [from,to] of [['@mujoco/mujoco/mujoco.js','mujoco.js'],['@mujoco/mujoco/mujoco.wasm','mujoco.wasm'],['three/build/three.core.js','three.core.js'],['three/build/three.module.js','three.module.js'],['three/examples/jsm/controls/OrbitControls.js','OrbitControls.js'],['three/LICENSE','THREE-LICENSE.txt']])await copyFile('node_modules/'+from,'dist/vendor/'+to);

for(const name of ['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm'])await copyFile('node_modules/onnxruntime-web/dist/'+name,'dist/vendor/'+name);
await copyFile('node_modules/onnxruntime-web/README.md','dist/vendor/ONNX-RUNTIME-README.md');
