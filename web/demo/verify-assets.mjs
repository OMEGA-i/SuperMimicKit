import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const lock=JSON.parse(await readFile('assets.lock.json','utf8'));
for(const [file,expected] of Object.entries(lock.files)){
 const bytes=await readFile(file),hash=createHash('sha256').update(bytes).digest('hex');
 if(hash!==expected)throw Error('Asset differs from source lock: '+file);
}
console.log('Verified '+Object.keys(lock.files).length+' generated assets against source lock.');
