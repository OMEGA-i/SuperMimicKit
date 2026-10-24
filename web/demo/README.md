# SuperMimicKit browser demo

Device selection supports G1, H2 and A3. Policy and world-model inference use
ONNX Runtime Web / WebGPU. Physics uses MuJoCo WASM on the CPU; rendering uses
Three.js. HTTPS (with a trusted certificate) or localhost is required for WebGPU.

```bash
cd web/demo
npm ci
npm run prepare:vendor
```

From the repository root, inspect the export commands:

```bash
python -m scripts.export_web_demo --help
python -m scripts.export_web_onnx --help
python -m scripts.export_web_motions --help
python -m scripts.serve_web_demo --help
```

Download the three robot ZIPs from the [preview release](https://github.com/OMEGA-i/SuperMimicKit/releases/tag/demo-preview-20261024)
and extract their `g1/`, `h2/`, `a3/` directories into `web/demo/dist/assets/`.
They include FP32 ONNX networks, matching robot assets and a dataset-free synthetic
fixture. **They do not include motion data.** The page displays a setup message
until compatible motion references are supplied.

With your own appropriately licensed LaFAN test37 retargeted motions, run from the
repository root (replace `YOUR_MOTIONS` with your local dataset root):

```bash
for robot in g1 h2 a3; do
  .venv/bin/python -m scripts.export_web_motions \
    --motions "YOUR_MOTIONS/$robot" --assets "web/demo/dist/assets/$robot" \
    --expected-count 37
done
python3 -m http.server 8767 --bind 127.0.0.1 --directory web/demo/dist
```

Open `http://localhost:8767`. The exporter checks 50 Hz, joint/body names, dimensions
and finite values; retarget using the package's matching model. Prepare source data
according to the [data guide](../../docs/getting_started.md) and its licenses. Do not
redistribute adapted motions unless your permissions allow it.

G1 uses policy 60k + world 55.5k; H2 uses paired 30k; A3 uses paired 100k. The UI shows
both iterations for mixed pairs. Actions are clamped to `[-20,20]` after noise, as
in training; A3 uses its exported equality constraints without adding duplicates.
The world visualization resynchronizes to physics every 32 steps.

For an offline smoke test after installing web dependencies:

```bash
node scripts/check_web_v2.mjs web/demo/dist/assets/g1
```

The released synthetic fixture checks runtime consistency, not motion quality or
independent PyTorch parity. `validate-onnx.html` runs the 32-step fixtures in WebGPU.

The refresh tools publish versioned checkpoints and atomically switch metadata
only after validation. Existing browser sessions check metadata for newer
versions. Keep old weight files available while sessions may still use them.
Generate a fresh asset lock with `scripts/write_web_asset_lock.py` for your own
exports before running `npm run verify:assets`.
