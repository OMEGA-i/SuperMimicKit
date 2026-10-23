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

Export compatible robot assets, ONNX networks, normalization metadata and
motions into `web/demo/dist/assets/<robot>`. No weights or motion data are
included. The demo requires these generated assets before it can run; it is
not a hosted pretrained demo. Keep model, normalizer and motion schemas paired.

The refresh tools publish versioned checkpoints and atomically switch metadata
only after validation. Existing browser sessions check metadata for newer
versions. Keep old weight files available while sessions may still use them.
Generate a fresh asset lock with `scripts/write_web_asset_lock.py` for your own
exports before running `npm run verify:assets`.
