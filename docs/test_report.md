# Test Report — ComfyUI-YEAHUD

Environment: ComfyUI 0.24.0 venv (Python 3.12.8, torch 2.7.0.dev, numpy 1.26.4,
Pillow 11.0.0, MPS). `pytest` is not installed in this venv, so each file is also a
self-runner (`python tests/test_X.py`, exit 1 on failure). All tests are
ComfyUI-/network-/model-free (synthetic numpy/torch fixtures).

## Unit + integration suite — **573 passed, 0 failed** (18 files)

| File | Passed | Covers |
|---|---:|---|
| test_utils.py | 39 | seed-stable hashing (4 PYTHONHASHSEED subprocesses), rng, LRU |
| test_geometry.py | 50 | xyxy math, padding, EMA smoothing, IoU, label_rect on-frame fuzz, asset_rect |
| test_palette.py | 46 | hex parse, default fallback, deterministic per-track color, contrast |
| test_alpha.py | 30 | inverse-alpha recovery, luminance fallback, trim, recolor, resize |
| test_captions.py | 56 | source modes, template tokens (no-raise), tolerant JSON |
| test_text_render.py | 33 | font cache + fallback, measure, truncate/ellipsis, strip |
| test_procedural_hud.py | 35 | corners/outline/solid/crosshair/registration, bounds clip |
| test_bbox_adapter.py | 44 | §25.1 — dict/list/list[list], xywh/xyxy, normalized, malformed, broadcast |
| test_mask_adapter.py | 35 | mask shapes, connected components, native bbox out |
| test_track_adapter.py | 31 | SAM3_TRACK_DATA bit-unpack, object-index identity, occlusion skip |
| test_tracker.py | 58 | §25.3 — passthrough + greedy, occlusion, order-change, determinism |
| test_assets.py | 24 | eligible slots, recolor+resize cache, luminance/none fallbacks |
| test_compositor.py | 47 | §25.4 — shape/range/batch, passthrough, no-mutate, clip, fill, assets |
| test_label_placement.py | 8 | in-bounds, boundary flip, temporal stability, 50-label, collision |
| test_annotations.py | 18 | schema parse/validate/normalize, bbox coercion, merge, consensus, **sensitive filter** |
| test_annotation_render.py | 10 | display labels, child face/footwear, frame ball, multi-group, density, fallback |
| test_integration_renderer.py | 9 | **§25.5** — real BOUNDING_BOX + SAM3_TRACK_DATA + MASK end-to-end |
| test_cloud_safety.py | OK | no eval/exec/subprocess/network/pip in shipped source |

## Integration (most critical — brief §25.5)
`run_hud_renderer` driven with the **exact native runtime structures** (verified from
`comfy_extras/nodes_sam3.py`):
- `BOUNDING_BOX` = `list[list[{x,y,width,height,score}]]` — ✅ renders, 2 dets/frame.
- `SAM3_TRACK_DATA` = `{packed_masks uint8 [N,N_obj,Hm,Wm//8], n_frames, scores, orig_size}` —
  ✅ primary path, object index = stable identity, **preferred over bboxes**.
- `MASK` `[B,H,W]` — ✅ fallback via connected components.
- No-detection frames pass through unchanged; output `[B,H,W,3]` float32 0..1; batch
  length preserved; same seed reproducible; frame-count mismatch reported, not fatal.

## Node registration (production simulation)
Loaded the package the way ComfyUI does (`importlib.spec_from_file_location` under a
mangled name; only the dir `__init__` inserts on `sys.path`):
- **V3** (`comfy_api` present): `comfy_entrypoint` → **8 nodes** register, all
  `define_schema()` build (renderer = 70 inputs / 3 outputs).
- **V1** (`comfy_api` absent): `NODE_CLASS_MAPPINGS` → **8 nodes**, all `INPUT_TYPES()` build.

## Sample output
`python examples/make_sample.py` → `examples/sample_output.mp4` (60×640×360 @ 24fps),
`docs/sample_before.png`, `docs/sample_after.png`. tracking_json: 4 tracks, source
`track_data`, 4 annotated, caption_mode `detailed`, category colors, child face/footwear
+ ball frame annotation, collision-aware placement. Visual: corner brackets, crosshairs,
registration marks, truncated detailed primary labels, micro-tag children.

## Dense / performance
`test_dense_50_labels_completes`: 50 annotated tracks on 720p (2 frames) renders to a
valid `[B,720,1280,3]` tensor without error. `test_label_density_low_hides_more_than_high`
confirms density gating. Caches (font / recolored-asset / resized-asset 8px-bucket /
per-frame plans) and a single CPU↔PIL hop per frame keep dense frames usable; progress +
interruption supported.

## Workflow validation
`python examples/build_workflows.py` builds + validates 5 importable graphs — every node
type is registered and every link connects type-matching sockets (validated at build
time and re-checked). No third-party / missing nodes.

## REAL SAM3 integration — EXECUTED & PASSED
Ran ComfyUI's native `SAM3_VideoTrack` headless (reference ComfyUI 0.24.0 + venv, MPS) on
a real 24-frame clip using `sam3.1_multiplex_fp16.safetensors` (Comfy-Org/sam3.1, 1.6 GB):
- Real `SAM3_TRACK_DATA`: `packed_masks [24, 6, 1008, 126] uint8`, `n_frames=24`, `n_obj=6`,
  `scores=[0.95,0.93,0.90,0.86,0.85,0.63]`, `orig_size=(376,896)` — **matches
  `runtime_findings.md` §3.5 exactly** (the adapter was built against this contract).
- 6 persistent objects → renderer (asset_pack + Tektur + track_random) → 144 detections,
  stable IDs/colors across first/mid/last (`docs/real_sam3_*.png`), original audio + 24 fps
  preserved → `examples/real_sam3_output.mp4`. Harness: `examples/run_real_sam3.py`.
- Perf: ~7 s/frame on MPS (Apple); far faster on a CUDA GPU / in Comfy Desktop.

## Known limitations
- **Comfy Cloud presence is not self-serve** — see `cloud_compatibility.md` (custom-node
  upload is disabled; inclusion is curated by Comfy Org). Cloud's running ComfyUI version
  and whether core SAM3 is enabled there are **not publicly documented** — verify in-session
  with the Inspect nodes.
- Detailed labels require an annotation source (manual/VLM); the renderer never invents them.
- `two_line` label length mode currently renders single-line (no multi-line strip yet).
