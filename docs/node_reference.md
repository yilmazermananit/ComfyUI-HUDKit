# Node Reference — ComfyUI-HUDKit

8 nodes. Categories: `HUDKit` (renderer + inspectors + mask→bbox) and
`HUDKit/Annotation` (the semantic-label layer). All register via the V3 schema
(`comfy_entrypoint`); a V1 `NODE_CLASS_MAPPINGS` fallback mirrors them on older ComfyUI.

## HUDKit Renderer  (`YEAHUDRenderer`)
Main compositor.

**Key inputs** (required): `frames` IMAGE; `seed` INT; `style_mode` COMBO
(`mixed/assets_only/procedural_only/corners/outline/solid/target_only/labels_only`);
`max_objects` INT; `display_probability` FLOAT; `padding_x/y` INT; `smoothing` FLOAT;
`track_iou_threshold` FLOAT; `track_memory_frames` INT; `frame_opacity`/`fill_opacity`/
`label_opacity` FLOAT; `line_width` INT; `target_scale`/`label_scale` FLOAT.

**Tracking sources** (optional, priority top→bottom): `track_data` SAM3_TRACK_DATA
(preferred) · `secondary_track_data_1..3` SAM3_TRACK_DATA (+ `primary_group_label`,
`secondary_group_label_1..3` STRING for namespacing/colors) · `bboxes` BOUNDING_BOX ·
`masks` MASK.

**Annotations** (optional): `annotations` YEA_ANNOTATIONS (preferred) ·
`annotations_json` STRING · `caption_mode` COMBO (`auto/short/medium/detailed/action/
technical/mixed`) · `label_length_mode` (`single_line/two_line/truncate/adaptive`) ·
`label_collision_mode` (`none/hide_low_priority/reposition/reposition_and_shorten/
dense_editorial`) · `label_density` FLOAT · `max_labels_per_frame` INT ·
`show_child_annotations`/`show_action_label`/`show_context_label`/`show_confidence`/
`show_track_id` BOOL · `annotation_priority_threshold` INT · `color_mode`
(`track_random/category_based/source_based/fixed_palette/annotation_override`).

**Assets** (optional): `frame_1..6_image` + `frame_1..6_mask`, `target_1..2_image` +
`target_1..2_mask` (Load Image IMAGE+MASK pairs), or a single `asset_pack` from the
**HUDKit Asset Pack** node. White-on-transparent PNGs; recolored to the track color.
Frame designs are typically a top label-bar + thin corner brackets; supply several at
different proportions (a portrait one, a square one, a wide one) and the renderer does
**aspect-aware selection** — per box it picks the design whose *content* aspect (w/h)
is closest (log-symmetric) to the box, so a tall box gets the tall design and a wide box
the wide one, with no stretch deformation. The choice is computed on the smoothed box so
it is stable across frames. Solid/filled frame assets (alpha coverage > 0.30) are dropped
by the Asset Pack by default so the box interior stays transparent.

**Captions/style** (optional):
`caption_source`, `fixed_label`, `captions_json`, `caption_template`, `palette_json`,
`small_object_mode`, `label_text_color_mode`, `asset_fit_mode`, `trim_transparent_assets`,
`padding_mode`, `flicker_probability`, `position_jitter_px`, `min_box_area_ratio` ·
`target_probability` FLOAT (chance a track shows a centre reticle; `0` = none — the frame
design already brackets the box) · `fill_probability` FLOAT · `detail_density` FLOAT
(procedural machine-vision registration marks; default `0` since user designs carry their
own marks — raise it for the asset-less procedural look) · `asset_line_boost` INT
(stroke-thickening in px for frame/target PNGs, default `2` — a thin-line design shrunk
onto a small box renders faint, so its strokes are dilated by this amount in output
pixels; `0` = strokes as-is, raise for bolder brackets).

**Outputs:** `images` IMAGE · `debug_preview` IMAGE (first annotated frame) ·
`tracking_json` STRING.

## HUDKit Inspect SAM3 Track Data  (`YEAInspectSAM3TrackData`)
In: `track_data` SAM3_TRACK_DATA. Out: `debug_json` STRING, `track_data` (pass-through).
Logs packed_masks shape/dtype, n_frames, scores, orig_size, per-frame visibility.

## HUDKit Inspect Bounding Boxes  (`YEAInspectBoundingBoxes`)
In: `bboxes` BOUNDING_BOX (opt), `masks` MASK (opt). Out: `debug_json` STRING, `bboxes`
(pass-through). Summarizes type/nesting/keys/per-frame counts/coord format (bounded).

## HUDKit Mask to Native BBoxes  (`YEAMaskToNativeBBoxes`)
In: `masks` MASK, `threshold` FLOAT, `min_area` INT, `merge_distance` INT.
Out: `bboxes` BOUNDING_BOX (native `list[list[{x,y,width,height,score}]]`), `debug_json` STRING.

## HUDKit Annotation JSON  (`YEAAnnotationJSON`)
In: `annotations_json` STRING (multiline), `strict` BOOL. Out: `annotations`
YEA_ANNOTATIONS, `normalized_json` STRING, `validation_report` STRING.

## HUDKit Annotation Builder  (`YEAAnnotationBuilder`)
In (all opt): `track_data` SAM3_TRACK_DATA, `crop_manifest_json`, `captions_json`
(VLM list OR by-track dict), `manual_annotations_json`, `secondary_detections_json`,
`default_category`, `primary_group_label`, `label_template`. Out: `annotations`,
`annotations_json`, `validation_report`. Reduces per-crop VLM responses to one
consensus label per track.

## HUDKit Track Crop Extractor  (`YEATrackCropExtractor`)
In: `frames` IMAGE, `track_data` SAM3_TRACK_DATA, `samples_per_track` INT,
`crop_padding` FLOAT, `minimum_track_area` FLOAT, `selection_mode` COMBO
(`best/largest/sharpest/most_centered/multi_sample`). Out: `crops` IMAGE,
`crop_manifest_json` STRING, `contact_sheet` IMAGE, `track_indices_json` STRING.

## HUDKit Detailed Label Prompt  (`YEADetailedLabelPrompt`)
In: `detail_level` COMBO, `include_clothing/action/objects/scene/visible_details` BOOL,
`maximum_words` INT, `uppercase` BOOL. Out: `system_prompt` STRING, `user_prompt`
STRING, `json_schema` STRING. **No network** — emits text only.

---

## YEA_ANNOTATIONS schema (v1)
```json
{
  "version": 1,
  "video": {"width": 1920, "height": 1080, "fps": 24.0, "frame_count": 360},
  "tracks": [{
    "track_id": "person:0", "source": "sam3_primary", "category": "person",
    "short_label": "PERSON", "medium_label": "SHIRTLESS MAN",
    "detailed_label": "SHIRTLESS MAN WITH TATTOOS WEARING BEIGE SWEATPANTS",
    "action_label": "KICKING SOCCER BALL", "context_label": "ON OUTDOOR COURT",
    "display_label": "SHIRTLESS MAN WITH TATTOOS KICKING SOCCER BALL",
    "confidence": 0.94,
    "attributes": {"clothing": ["beige sweatpants"], "visible_details": ["tattoos"],
                   "action": ["kicking"], "interacting_objects": ["soccer ball"]},
    "style_overrides": {"priority": 100, "preferred_color": null, "preferred_asset": null,
                        "preferred_label_mode": "detailed"},
    "children": [{
      "annotation_id": "person:0:face", "category": "human_face", "short_label": "HUMAN FACE",
      "region_source": "secondary_detector", "region_mode": "bbox",
      "bbox_by_frame": {"0": [297, 112, 345, 175]}
    }]
  }],
  "frame_annotations": {"0": [{"annotation_id": "ball:0", "category": "soccer_ball",
    "short_label": "SOCCER BALL", "bbox": [218, 385, 252, 420], "confidence": 0.89, "priority": 70}]}
}
```
- Track ids are **namespaced** (`person:0`, `face:0`) — globally unique across groups.
- bbox in any of `xyxy/xywh × pixel/normalized` (use `bbox_format`); normalized internally to abs-pixel xyxy.
- Children & frame_annotations move independently when per-frame regions are supplied.

## tracking_json (renderer output)
```json
{"frames": 60, "detections": 240, "tracks": 4, "source": "track_data",
 "style_mode": "mixed", "seed": 77, "image_size": [640, 360], "annotated_tracks": 4,
 "caption_mode": "detailed", "n_obj": 4, "groups": 1}
```
Plus `frame_mismatch` when detection-frame count ≠ image-batch count.

## Universal detection JSON (secondary_detections_json / Annotation Builder)
A list of `{frame_index, track_id?, parent_track_id?, bbox, bbox_format?, label?,
short_label?, score?, category?, source?}`. Normalized to abs-pixel xyxy; zero-area
boxes rejected; malformed entries reported, never fatal.
