# Architecture — ComfyUI-YEAHUD

## Three independent layers
```
Tracking  (native SAM3)      → stable identity + masks/boxes over time
   │  SAM3_TRACK_DATA (preferred) | BOUNDING_BOX + MASK (fallback)
   ▼
Annotation (YEA_ANNOTATIONS) → short/detailed labels, actions, child regions
   │  manual JSON | VLM workflow | secondary detections | safe generic fallback
   ▼
Rendering (this pack)         → PNG/procedural frames, crosshairs, label strips, text
```
The renderer **never invents** descriptive text — it only renders what the annotation
layer supplies (or a safe generic label). Caption generation (if any) lives **outside**
the renderer in a composable workflow, so the core stays network/model-free (Cloud-safe).

## Package layout
```
__init__.py            V3 comfy_entrypoint (8 nodes); V1 NODE_CLASS_MAPPINGS fallback; sys.path bootstrap
nodes/                 thin schema + wiring (no business logic):
  hud_renderer.py        run_hud_renderer() + YEAHUDRenderer (V3)
  inspect_track_data.py inspect_bbox.py mask_to_bbox.py
  annotation_json.py annotation_builder.py track_crop_extractor.py detailed_label_prompt.py
  v1_compat.py           V1 mirror, reuses the same run_* functions
yea_hud/            schema-agnostic business logic (torch/numpy/PIL only):
  types.py               Detection, Track, TrackStyle, RenderItem, DEFAULTS, enums
  utils.py               stable_hash / rng_for / unit_float / LRUCache
  geometry.py            xyxy math, padding, EMA smoothing, IoU, importance, label_rect, asset_rect
  bbox_adapter.py        native BOUNDING_BOX → Detection (3 shapes, defensive)
  mask_adapter.py        MASK normalize + connected components → Detection / native bboxes
  track_adapter.py       SAM3_TRACK_DATA → Detection (object index = persistent id)
  tracker.py             passthrough (native id) OR greedy IoU; lifecycle; deterministic style
  annotations.py         YEA_ANNOTATIONS schema/validate/normalize/merge/consensus + sensitive filter
  label_placement.py     candidate positions + collision cost + temporal stability
  alpha.py               Load Image inverse-alpha recovery, trim, recolor, resize
  palette.py             palette parse, deterministic per-track color, contrast
  text_render.py         font cache, measure, truncate, label strip
  procedural_hud.py      corners / outline / solid / crosshair / registration
  compositor.py          per-frame single-RGBA-overlay draw, alpha-composite, stack
web/js/yea_hud.js   optional UX (rendering works headless without it)
```

## Identity & determinism
- **Passthrough:** when a detection carries a `source_track_id` (SAM3 object index,
  namespaced `person:0`), `Track.track_id` == that id → identity survives occlusion for free.
- **Greedy:** otherwise IoU+centroid+area matching with monotonic, never-reused ids.
- A track's visual identity (color, frame/target asset, primitive style, show/hide,
  glitch personality) is derived **once** from `stable_hash(seed, track_id)` and cached —
  the same subject never re-rolls per frame. Per-frame micro-variation (flicker, jitter)
  uses `stable_hash(seed, track_id, frame_index)`. Same seed ⇒ identical output.

## Per-frame render pipeline (compositor §17 order)
base frame → solid blocks → frame assets / procedural frames → crosshair →
label strips → text → registration marks. One CPU/numpy hop + one PIL RGBA overlay per
frame; input tensor never mutated; output `[B,H,W,3]` float32 0..1, batch length preserved.

## Label placement
The renderer measures each label, builds `PlacementRequest`s (anchor, size, priority,
primary?), and `LabelPlacer` assigns each a rect minimizing a cost over candidate
positions (boundary overflow, overlap, covering a primary subject, connector length,
**temporal jump** vs the track's last position). Density / `max_labels_per_frame` /
collision mode decide what stays visible; the compositor draws at the resolved rect.

## Schema compatibility
V3 is primary (matches native SAM3's own API). The loader prefers `NODE_CLASS_MAPPINGS`
over `comfy_entrypoint`, so `__init__` defines exactly one: V3 entrypoint when
`comfy_api` imports, else the V1 mappings. Business logic is shared, so there is no
duplicated implementation.
