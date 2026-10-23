import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const code=fs.readFileSync('web/demo/dist/app.js','utf8').split('// Check lightweight metadata;')[1];
function fixture({same=false,fail=false,changedModel=false}={}){
 const old={model:{sha256:'old-model'},inference_fixture:'old.json',joint_names:['j'],body_names:['b'],context:[1]};
 const next={...old,model:{sha256:changedModel?'new-model':'old-model'},inference_fixture:same?'old.json':'new.json',completed_iterations:2000,motions:[{id:'test'}],context:[1]};
 const counters={created:0,oldDisposed:0,newDisposed:0,reset:0,reload:0};
 const elements={device:{disabled:false},friction:{value:'0.7'},checkpoint:{}};
 const ctx={console:{error(){}},document:{hidden:false,addEventListener(){}},setInterval(){},setTimeout,Float32Array,
 ready:true,loading:false,busy:false,playing:false,currentRobot:'g1',meta:old,device:{},titles:{g1:'G1'},
 chooseRobot:async()=>{counters.reload++;ctx.meta=next;},
 net:{dispose(){counters.oldDisposed++;}},$:id=>elements[id],notice(){},syncPlay(){},reset(){counters.reset++;ctx.playing=true;},
 fetch:async url=>({ok:true,json:async()=>url.endsWith('metadata.json')?next:{history:[0],goals:[0],state:[0],action:[0],world_next:[0]}}),
 Network:class{constructor(){counters.created++;}async load(){}async action(){return new Float32Array([fail?NaN:0]);}async world(){return new Float32Array([0]);}dispose(){counters.newDisposed++;}}
 };
 vm.createContext(ctx);vm.runInContext('// Check lightweight metadata;'+code,ctx);
 return {ctx,counters,old,elements};
}
{
 const {ctx,counters}=fixture({same:true});await ctx.checkCheckpoint();assert.equal(counters.created,0);
}
{
 const {ctx,counters,elements}=fixture();await ctx.checkCheckpoint();assert.equal(ctx.meta.completed_iterations,2000);assert.equal(counters.oldDisposed,1);assert.equal(counters.reset,1);assert.equal(ctx.playing,false);assert.equal(ctx.meta.context[0],0.7);assert.equal(ctx.loading,false);assert.equal(elements.device.disabled,false);
}
{
 const {ctx,counters,old}=fixture({fail:true});await ctx.checkCheckpoint();assert.equal(ctx.meta,old);assert.equal(counters.oldDisposed,0);assert.equal(counters.newDisposed,1);assert.equal(ctx.loading,false);
}
{const {ctx,counters}=fixture({changedModel:true});await ctx.checkCheckpoint();assert.equal(counters.reload,1);assert.equal(counters.created,0);assert.equal(ctx.meta.model.sha256,'new-model');}
console.log('4 browser checkpoint lifecycle tests passed');
