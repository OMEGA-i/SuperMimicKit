export const cat=(...xs)=>Float32Array.from(xs.flatMap(x=>Array.from(x)));
export function normalizeState(s,m){for(const off of [3,...Array.from({length:m.layout.bodies},(_,i)=>13+2*m.layout.joints+13*i+3)]){const n=Math.hypot(...s.slice(off,off+4))||1;for(let j=0;j<4;j++)s[off+j]/=n;}return s;}
export function inverse(q,v){const n=Math.hypot(...q)||1,w=q[0]/n,x=-q[1]/n,y=-q[2]/n,z=-q[3]/n;const t=[2*(y*v[2]-z*v[1]),2*(z*v[0]-x*v[2]),2*(x*v[1]-y*v[0])];return [v[0]+w*t[0]+y*t[2]-z*t[1],v[1]+w*t[1]+z*t[0]-x*t[2],v[2]+w*t[2]+x*t[1]-y*t[0]];}
const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),scale=(a,x)=>a.map(v=>v*x);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const qc=q=>[q[0],-q[1],-q[2],-q[3]],qr=(q,v)=>inverse(qc(q),v);
const qm=(a,b)=>[a[0]*b[0]-a.slice(1).reduce((n,v,i)=>n+v*b[i+1],0),...add(add(scale(b.slice(1),a[0]),scale(a.slice(1),b[0])),cross(a.slice(1),b.slice(1)))];
const qexp=v=>{const n=Math.hypot(...v),k=n<1e-8?.5:Math.sin(n/2)/n;return [Math.cos(n/2),...scale(v,k)];};
const yaw=q=>{const [w,x,y,z]=q,t=Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))/2;return [Math.cos(t),0,0,Math.sin(t)];};
const rot6=q=>[...qr(q,[1,0,0]),...qr(q,[0,1,0])];
const norm=(x,n)=>Float32Array.from(x,(v,i)=>(v-n.mean[i%n.mean.length])/n.std[i%n.std.length]);
export function observation(s,a,m,g){const j=m.layout.joints;if(!g)throw Error('Current reference required');return cat(s.slice(13,13+j).map((v,i)=>v-m.default_pose[i]),s.slice(13+j,13+2*j),inverse(s.slice(3,7),s.slice(10,13)),inverse(s.slice(3,7),[0,0,-1]),a,rot6(qm(qc(s.slice(3,7)),g.slice(3,7))));}
export function goalFeatures(seq,m){const d=m.layout.dim,j=m.layout.joints,now=seq.slice(0,d),h=qc(yaw(now.slice(3,7))),out=[];for(let t=1;t<=10;t++){const f=seq.slice(t*d,(t+1)*d);out.push(...qr(h,sub(f.slice(0,3),now.slice(0,3))),...rot6(qm(h,f.slice(3,7))),...inverse(f.slice(3,7),f.slice(7,10)),...inverse(f.slice(3,7),f.slice(10,13)),...f.slice(13,13+j),...scale(f.slice(13+j,13+2*j),.1));}return Float32Array.from(out);}
function invariant(s,m){const j=m.layout.joints,r=s.slice(3,7);return cat([s[2]],rot6(qm(qc(yaw(r)),r)),inverse(r,s.slice(7,10)),inverse(r,s.slice(10,13)),s.slice(13,13+2*j));}
function contextFeatures(c){return cat([(c[0]-.75)/.45],c.slice(1,-3).map(v=>v/.01),c.slice(-3).map((v,i)=>v/[.025,.05,.05][i]));}
export function fkState(root,q,qd,k){const P=[],Q=[],V=[],W=[];for(let i=0;i<k.parents.length;i++){const par=k.parents[i],j=k.joints[i];let p,r,v,w;if(par<0){p=Array.from(root.slice(0,3));r=Array.from(root.slice(3,7));v=Array.from(root.slice(7,10));w=Array.from(root.slice(10,13));}else{const r0=qm(Q[par],k.rotation[i]),p0=add(P[par],qr(Q[par],k.offset[i])),v0=add(V[par],cross(W[par],sub(p0,P[par])));if(j>=0){r=qm(r0,qexp(scale(k.axis[i],q[j]-k.reference[i])));const a0=qr(r0,k.anchor[i]),a1=qr(r,k.anchor[i]);p=sub(add(p0,a0),a1);w=add(W[par],scale(qr(r0,k.axis[i]),qd[j]));v=sub(add(v0,cross(W[par],a0)),cross(w,a1));}else{p=p0;r=r0;v=v0;w=W[par];}}P.push(p);Q.push(r);V.push(v);W.push(w);}return cat(root,q,qd,...k.tracked.map(i=>cat(P[i],Q[i],V[i],W[i])));}
export async function bytes(url,compression){const r=await fetch(url);if(!r.ok)throw Error(`${r.status}: ${url}`);return compression==='gzip'?new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer():r.arrayBuffer();}
export async function floats(url,dtype='float32',compression){const buffer=await bytes(url,compression);if(dtype==='float16'){const bits=new Uint16Array(buffer),out=new Float32Array(bits.length);for(let i=0;i<bits.length;i++){const h=bits[i],sign=(h&32768)?-1:1,e=(h>>10)&31,f=h&1023;out[i]=sign*(e===0?f*2**-24:e===31?(f?NaN:Infinity):(1+f/1024)*2**(e-15));}return out;}return new Float32Array(buffer);}
let configuredRuntime=null;
export class Network{
 constructor(device,{runtime=null,offlineValidation=false}={}){this.device=device;this.runtime=runtime;this.offlineValidation=offlineValidation;this.sessions={};}
 async load(meta,base,progress){
  this.meta=meta;
  if(meta.inference_backend!=='onnxruntime-webgpu-f32'||!meta.onnx)throw Error('ONNX checkpoint required');
  if(!this.device&&!this.offlineValidation)throw Error('WebGPU required; CPU fallback is disabled');
  this.ort=this.runtime||await import('./vendor/ort.webgpu.min.mjs');
  if(!this.offlineValidation&&configuredRuntime!==this.ort){
   this.ort.env.wasm.numThreads=1;
   this.ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
   this.ort.env.webgpu.device=this.device;
   configuredRuntime=this.ort;
  }
  try{
   for(const name of ['policy','dynamics']){
    progress?.(name+' · ONNX FP32');
    const spec=meta.onnx[name],data=await bytes(base+spec.file);
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),x=>x.toString(16).padStart(2,'0')).join('');
    if(hash!==spec.sha256)throw Error('ONNX checksum mismatch: '+name);
    this.sessions[name]=await this.ort.InferenceSession.create(new Uint8Array(data),{
     executionProviders:[this.offlineValidation?'cpu':'webgpu'],
     graphOptimizationLevel:'all',
     ...(this.offlineValidation?{intraOpNumThreads:2}:{extra:{session:{disable_cpu_ep_fallback:'1'}}})
    });
   }
  }catch(e){await this.dispose();throw e;}
 }
 async run(name,values){
  const spec=this.meta.onnx[name],feeds={};
  for(const [key,dims] of Object.entries(spec.inputs))feeds[key]=new this.ort.Tensor('float32',Float32Array.from(values[key]),dims);
  let result;
  try{result=await this.sessions[name].run(feeds);return Float32Array.from(result[spec.output].data);}
  finally{Object.values(feeds).forEach(t=>t.dispose());if(result)Object.values(result).forEach(t=>t.dispose());}
 }
 async action(history,goals){return this.run('policy',{history,goal_features:goalFeatures(goals,this.meta)});}
 async world(s,a,c,m){
  const j=m.layout.joints,d=await this.run('dynamics',{features:cat(invariant(s,m),a,contextFeatures(c))}),r=s.slice(3,7),rq=qm(r,qexp(d.slice(3,6))),rn=Math.hypot(...rq);
  const root=cat(add(s.slice(0,3),qr(yaw(r),d.slice(0,3))),scale(rq,1/rn),add(s.slice(7,10),qr(r,d.slice(6,9))),add(s.slice(10,13),qr(r,d.slice(9,12))));
  return fkState(root,add(s.slice(13,13+j),d.slice(12,12+j)),add(s.slice(13+j,13+2*j),d.slice(12+j)),m.kinematics);
 }
 async dispose(){const sessions=Object.values(this.sessions);this.sessions={};await Promise.all(sessions.map(s=>s.release()));}
}
