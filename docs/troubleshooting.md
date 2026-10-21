# Troubleshooting — ComfyUI-HUDKit

### "HUDKit Renderer requires either native SAM3_TRACK_DATA, BOUNDING_BOX, or MASK"
No detection source is connected. Wire **SAM3 Video Track** → `track_data` (preferred),
or **SAM3 Detect** → `bboxes` / `masks`. At least one is required.

### A socket won't connect / labels land on the wrong subject
Don't trust socket color. Run **HUDKit Inspect SAM3 Track Data** or **HUDKit Inspect
Bounding Boxes** on the upstream output and read `debug_json` — it prints the real
Python type, nesting, dict keys, per-frame counts and coordinate format. This pack
targets the **native** types (`SAM3_TRACK_DATA`, `BOUNDING_BOX = list[list[{x,y,width,
height}]]`, `MASK`). A community/third-party SAM3 pack may emit different types — verify
before wiring (especially on Comfy Cloud; see `cloud_compatibility.md`).

### My white PNG renders as a solid black box
Connect the Load Image **MASK** output too (not just IMAGE). ComfyUI's Load Image emits
`MASK = 1 − alpha`; the renderer recovers true alpha from it. With no mask it falls back
to image luminance. If the asset has no real alpha channel, that's expected — give it a
transparent background.

### Labels overlap / leave the frame in crowded scenes
Lower `label_density`, set `max_labels_per_frame`, or use
`label_collision_mode = reposition_and_shorten` / `hide_low_priority`. Primary subjects
keep their labels; low-priority secondary labels are shortened or hidden first. Long
detailed labels auto-truncate with `…`; use `caption_mode = short/medium` or
`label_length_mode = adaptive` for denser frames.

### Track identity flickers / a person changes color mid-video
Prefer `SAM3_TRACK_DATA` (native persistent identity) over per-frame `BOUNDING_BOX`.
On the bbox path, raise `track_memory_frames` and/or lower `track_iou_threshold`.
Increase `smoothing` to reduce box jitter (too high lags fast motion).

### Detailed descriptions are generic ("PERSON") instead of "SHIRTLESS MAN…"
That text must be supplied — the renderer never invents it. Provide it via **YEA
Annotation JSON** (manual) or a VLM workflow (**Track Crop Extractor → your VLM →
Annotation Builder**). Without annotations you get safe generic group labels by design.

### Frame count mismatch
If SAM3 track data covers fewer frames than the image batch, the extra frames render
**without** detections (never auto-repeat) and `tracking_json.frame_mismatch` reports it.
Make sure the same frames feed both SAM3 and the renderer.

### Invalid annotation JSON
By default the renderer logs a `validation_report`, falls back to generic labels, and
still renders the whole video. Set `strict=True` on **HUDKit Annotation JSON** to fail
fast instead. The report cites the parse position and a minimal valid example.

### Slow on long videos
Use a VLM **once per track** (Track Crop Extractor) — never caption every frame. Asset
recolor/resize, fonts and text measurements are cached. A progress bar shows per-frame
work; the run is interruptible.

### Node doesn't appear after install
Check the ComfyUI log for `ComfyUI-HUDKit: registered N V3 nodes`. If it logged the
**V1** line instead, your ComfyUI predates `comfy_api` — the V1 fallback still registers
all 8 nodes. If nothing logged, confirm the folder is `custom_nodes/ComfyUI-HUDKit`
and `numpy`/`Pillow` are installed in ComfyUI's Python.
