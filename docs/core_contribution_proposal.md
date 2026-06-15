# Proposal (optional) — "SAM3 Track Overlay" as a ComfyUI core node

> Phase-4 evaluation (brief §28 / cloud strategy). A **non-branded, minimal** subset
> of YonderHUD proposed for `comfy_extras`, so it ships with Cloud's base ComfyUI
> (the only route that sidesteps the undocumented custom-node curation gate).

## Rationale
ComfyUI core already has `SAM3_TrackPreview`, which draws colored masks + numeric
ids onto frames as a *preview* (temp MP4, `is_output_node`, no tensor output). A
natural complement is an overlay node that returns an **IMAGE batch** with designed
HUD graphics driven by the same `SAM3_TRACK_DATA` — usable mid-graph, feeding Create
Video. It fills a real gap (detection→presentation) with **zero new dependencies**,
matching SAM3's "extra-dependency-free, single-GPU" design.

## Proposed node — `SAM3_TrackOverlay`
- **Category:** `image/detection` (next to the other SAM3 nodes).
- **Inputs:** `track_data` (SAM3_TRACK_DATA), `images` (IMAGE, optional — falls back to
  black like `SAM3_TrackPreview`), `style` (COMBO: brackets/outline/solid/target/labels),
  `line_width` (INT), `opacity` (FLOAT), `seed` (INT, deterministic per-track color),
  `show_labels` (BOOLEAN).
- **Output:** `images` (IMAGE).
- **Impl:** torch-only. Reuse `comfy.ldm.sam3.tracker.unpack_masks`; per-object index =
  stable id; deterministic palette via `hash(seed, obj_index)`; box from mask extent;
  brackets/crosshair/label drawn with the existing GPU glyph approach already in
  `SAM3_TrackPreview._draw_number_gpu`. No Pillow needed for the core-minimal version.

## What is intentionally LEFT OUT of the core version
User PNG frame/target assets, alpha recovery, recolor pipeline, the IoU fallback
tracker, BOUNDING_BOX/MASK adapters, caption JSON/template system, asset packs.
These stay in the branded **ComfyUI-YonderHUD** custom pack (richer, Pillow-based).

## Path
1. Open a discussion/RFC referencing PR #13408 (SAM3) and `SAM3_TrackPreview`.
2. Offer `SAM3_TrackOverlay` as a focused PR (torch-only, V3 schema, tests).
3. Keep YonderHUD as the full-featured superset on the Registry.

> This is a proposal only — not implemented in this repo. The branded renderer is the
> shipped deliverable; the core node is a future contribution route to guaranteed Cloud presence.
