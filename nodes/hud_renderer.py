"""YEA HUD Renderer — main node.

Consumes native SAM3 tracking output (preferred: SAM3_TRACK_DATA; fallback:
BOUNDING_BOX and/or MASK), tracks stable identities, attaches semantic annotations
(short/medium/detailed labels + child regions, supplied via YEA_ANNOTATIONS /
JSON — NEVER invented here), and composites designed HUD graphics over the frames.
Outputs an IMAGE batch ready for Create Video / Save Video.

Three layers stay separate: tracking (SAM3) -> annotation (labels) -> rendering.
Business logic lives in ``yea_hud/*``; this file is thin wiring + V3 schema and
is reused verbatim by the V1 shim.
"""

from __future__ import annotations

import dataclasses
import json
import logging

logger = logging.getLogger("YEAHUD.renderer")

DISPLAY_NAME = "YEA HUD Renderer"
NODE_ID = "YEAHUDRenderer"
CATEGORY = "YEA/HUD"
TRACK_DATA_TYPE = "SAM3_TRACK_DATA"
ANNOTATIONS_TYPE = "YEA_ANNOTATIONS"

DETECTION_SOURCE_ERROR = (
    "YEA HUD Renderer requires either native SAM3_TRACK_DATA, native BOUNDING_BOX, "
    "or MASK input. Connect SAM3 Video Track (track_data), SAM3 Detect (bboxes/masks), "
    "or a MASK source. Use 'YEA Inspect SAM3 Track Data' / 'YEA Inspect Bounding "
    "Boxes' to verify the real structure."
)

_SMALL_OBJECT_RATIO = 0.01

# Category -> color (color_mode 'category_based'); brief default_category_mapping.
DEFAULT_CATEGORY_COLORS = {
    "person": "#FF2A1A", "human_face": "#225CFF", "face": "#225CFF",
    "footwear": "#8DFF28", "object": "#FFB0DF", "action": "#FF2A1A",
    "soccer_ball": "#FFB0DF", "ball": "#FFB0DF", "vehicle": "#FFFFFF",
}


# ---------------------------------------------------------------------------
# Detection sourcing (multi-group aware) — priority TRACK_DATA > BBOX > MASK
# ---------------------------------------------------------------------------

def _category_of(source_track_id, default_category):
    """Derive a category from a namespaced id ('person:0' -> 'person')."""
    if isinstance(source_track_id, str) and ":" in source_track_id:
        return source_track_id.split(":", 1)[0]
    return default_category


def _build_detections(groups, bboxes, masks, B, W, H, default_category):
    """Build per-frame detections + (id->category) map + source label.

    ``groups`` is a list of (track_data, category, group_label) for SAM3_TRACK_DATA
    sources (primary + optional secondaries). Each object index is namespaced
    ('person:0', 'face:0', ...) so identities are globally unique across groups and
    flow through the tracker passthrough path (Track.track_id == that namespaced id).
    """
    from yea_hud import track_adapter, bbox_adapter, mask_adapter
    from yea_hud.annotations import namespace_id

    meta = {}
    category_by_id = {}

    if groups:
        combined = [[] for _ in range(B)]
        total_obj = 0
        for (td, category, _label) in groups:
            dets, n_obj = track_adapter.adapt_track_data(td)
            total_obj += n_obj
            for f, frame_dets in enumerate(dets):
                if f >= B:
                    break
                for d in frame_dets:
                    nid = namespace_id(category, d.source_track_id)
                    d.source_track_id = nid
                    d.label = d.label or _label
                    category_by_id[nid] = category
                    combined[f].append(d)
        meta["n_obj"] = total_obj
        meta["groups"] = len(groups)
        return combined, "track_data", meta, category_by_id

    if bboxes is not None:
        dets = bbox_adapter.adapt_bboxes(bboxes, B, W, H)
        for fd in dets:
            for d in fd:
                category_by_id[d.source_track_id if d.source_track_id is not None else d.source_index] = default_category
        return dets, "bboxes", meta, category_by_id

    if masks is not None:
        mask_list = mask_adapter.normalize_mask_batch(masks, B)
        dets = []
        for f, m in enumerate(mask_list):
            dets.append(mask_adapter.mask_to_detections(
                m, f, threshold=0.5, min_area=max(1, int(W * H * 0.0001)), merge_distance=0))
        return dets, "masks", meta, category_by_id

    raise ValueError(DETECTION_SOURCE_ERROR)


def _align_to_batch(dets, B, meta):
    n = len(dets)
    if n < B:
        meta["frame_mismatch"] = {"detections_frames": n, "image_frames": B, "policy": "missing frames empty"}
        dets = list(dets) + [[] for _ in range(B - n)]
    elif n > B:
        meta["frame_mismatch"] = {"detections_frames": n, "image_frames": B, "policy": "extra detection frames ignored"}
        dets = list(dets[:B])
    return dets


def _small_object_style(style, mode):
    if mode == "hide":
        return None
    if mode == "target_only":
        return dataclasses.replace(style, primitive_style="target_only", use_fill=False,
                                   use_target=True, use_caption=False, frame_asset_id=-1)
    if mode == "label_only":
        return dataclasses.replace(style, primitive_style="target_only", use_fill=False,
                                   use_target=False, use_caption=True, frame_asset_id=-1)
    if mode == "tiny_corners":
        return dataclasses.replace(style, primitive_style="corners", use_fill=False,
                                   use_caption=False, frame_asset_id=-1)
    return style


def _category_color(category, palette, seed):
    from yea_hud.palette import parse_hex
    from yea_hud.utils import stable_hash
    hexc = DEFAULT_CATEGORY_COLORS.get(category)
    rgb = parse_hex(hexc) if hexc else None
    if rgb is not None:
        return rgb
    return palette[stable_hash(category) % len(palette)]


def _label_for(ann_track, caption_mode, length_mode, area_ratio, *, show_action, show_context,
               show_confidence, show_track_id, show_category, fixed_fallback, track_id):
    """Resolve the display text for a track from its annotation (or a safe fallback)."""
    from yea_hud.annotations import resolve_display_label
    mode = caption_mode
    if length_mode == "adaptive" and caption_mode in ("auto", "mixed"):
        mode = "detailed" if area_ratio >= 0.06 else ("medium" if area_ratio >= 0.012 else "short")
    if ann_track is not None:
        base = resolve_display_label(ann_track, mode)
        extras = []
        if show_action and ann_track.action_label and ann_track.action_label.upper() not in base.upper():
            extras.append(ann_track.action_label)
        if show_context and ann_track.context_label:
            extras.append(ann_track.context_label)
        if extras:
            base = base + " " + " ".join(extras)
        if show_confidence and ann_track.confidence is not None:
            base = f"{base} {int(ann_track.confidence * 100)}%"
        return base or fixed_fallback
    return fixed_fallback


def run_hud_renderer(frames, *, seed, style_mode, max_objects, display_probability,
                     padding_x, padding_y, smoothing, track_iou_threshold, track_memory_frames,
                     frame_opacity, fill_opacity, label_opacity, line_width, target_scale,
                     label_scale=1.0, track_data=None, bboxes=None, masks=None,
                     secondary_track_data=None, group_labels=None, default_category="person",
                     primary_group_label="PERSON",
                     frame_images=None, frame_masks=None, target_images=None, target_masks=None,
                     captions_json="", fixed_label="PERSON", palette_json="",
                     caption_source="fixed", caption_template="{track:02d}. {label}",
                     small_object_mode="target_only", label_text_color_mode="auto",
                     trim_transparent_assets=True, asset_fit_mode="stretch",
                     padding_mode="pixels", flicker_probability=0.04, style_flash_probability=0.02,
                     position_jitter_px=0, opacity_jitter=0.05,
                     min_box_area_ratio=0.0004, large_box_area_ratio=0.10,
                     max_cache_items=512,
                     annotations=None, annotations_json="", caption_mode="auto",
                     label_density=1.0, max_labels_per_frame=0, max_labels_per_track=4,
                     label_collision_mode="reposition_and_shorten", label_length_mode="adaptive",
                     show_confidence=False, show_track_id=False, show_category=False,
                     show_child_annotations=True, show_action_label=True, show_context_label=False,
                     secondary_annotation_scale=0.6, annotation_priority_threshold=0,
                     color_mode="track_random", font="", asset_pack=None,
                     target_probability=0.0, fill_probability=0.25, detail_density=0.0,
                     asset_line_boost=2,
                     progress=None, interrupt=None):
    """Render HUD overlays. Returns (images, debug_preview, tracking_json_str)."""
    import torch  # noqa: F401
    from yea_hud import palette as palette_mod
    from yea_hud import captions as captions_mod
    from yea_hud import geometry, text_render
    from yea_hud.tracker import Tracker, micro_variation
    from yea_hud.assets import AssetBank
    from yea_hud.compositor import render_frames
    from yea_hud.types import RenderItem
    from yea_hud.label_placement import LabelPlacer, PlacementRequest
    from yea_hud.annotations import parse_annotations_json, AnnotationSet, namespace_id

    B = int(frames.shape[0]); H = int(frames.shape[1]); W = int(frames.shape[2])
    frame_area = float(max(1, W * H))
    palette = palette_mod.resolve_palette(palette_json)

    # --- annotations: object passthrough or JSON; never invented here ---
    if isinstance(annotations, AnnotationSet):
        aset = annotations
    else:
        aset, _ = parse_annotations_json(annotations_json, W=W, H=H, frame_count=B)
    ann_by_id = aset.tracks_by_id()

    # --- detection groups (multi-category) ---
    groups = []
    if track_data is not None:
        groups.append((track_data, default_category, primary_group_label))
    for i, td in enumerate(secondary_track_data or []):
        if td is not None:
            gl = (group_labels[i] if group_labels and i < len(group_labels) else default_category)
            cat = gl.lower().replace(" ", "_") if gl else f"group{i+1}"
            groups.append((td, cat, gl))

    dets, source, meta, category_by_id = _build_detections(
        groups, bboxes, masks, B, W, H, default_category)
    dets = _align_to_batch(dets, B, meta)

    # Assets come from a single asset_pack (preferred, collapses 16 sockets) or the
    # individual image/mask lists (used by tests / make_sample).
    if asset_pack:
        frame_pairs = asset_pack.get("frame_pairs", []) or []
        target_pairs = asset_pack.get("target_pairs", []) or []
    else:
        frame_pairs = list(zip(frame_images or [], frame_masks or []))
        target_pairs = list(zip(target_images or [], target_masks or []))
    assets = AssetBank(frame_pairs, target_pairs, trim=trim_transparent_assets,
                       fit_frame=asset_fit_mode, fit_target="contain", max_cache_items=max_cache_items,
                       line_boost=asset_line_boost)

    tracker = Tracker(seed=int(seed), iou_threshold=track_iou_threshold, max_centroid_ratio=0.35,
                      memory_frames=int(track_memory_frames), smoothing=smoothing, palette=palette,
                      n_frame_assets=assets.n_frame_assets, n_target_assets=assets.n_target_assets,
                      style_mode=style_mode, display_probability=display_probability, W=W, H=H,
                      target_probability=target_probability, fill_probability=fill_probability)

    captions = captions_mod.parse_captions_json(captions_json)
    placer = LabelPlacer(W, H, collision_mode=label_collision_mode)
    # frame cap derived from density unless an explicit cap is set.
    eff_max_labels = int(max_labels_per_frame) if max_labels_per_frame else max(1, int(round(40 * float(label_density))))

    def _strip_size(box, text, is_micro):
        bh = max(0.0, box[3] - box[1])
        factor = 0.10 if is_micro else 0.12
        fs = int(max(8 if is_micro else 10, round(bh * factor * max(0.1, label_scale))))
        font = text_render.get_font(fs)
        tw, th = text_render.measure(text, font)
        return int(tw + 14), int(th + 8)

    plans = []
    total_dets = 0
    seen_tracks = set()
    for f in range(B):
        if interrupt is not None:
            interrupt()
        pairs = tracker.update(f, dets[f])
        total_dets += len(dets[f])
        ranked = sorted(pairs, key=lambda p: geometry.importance(p[0].smoothed_box, W, H), reverse=True)
        if max_objects and max_objects > 0:
            ranked = ranked[:int(max_objects)]

        items = []
        order = 0
        place_reqs = []          # (item_index, PlacementRequest)
        for track, det in ranked:
            seen_tracks.add(track.track_id)
            if not track.show:
                continue
            box = track.smoothed_box
            area_ratio = geometry.area(box) / frame_area
            if area_ratio < float(min_box_area_ratio):
                continue
            style = track.style
            is_small = area_ratio < _SMALL_OBJECT_RATIO
            if is_small:
                style = _small_object_style(style, small_object_mode)
                if style is None:
                    continue

            key = str(track.track_id)
            category = category_by_id.get(track.track_id, _category_of(track.track_id, default_category))
            ann_track = ann_by_id.get(key)
            priority = (ann_track.style_overrides.priority if ann_track else
                        (100 if category == default_category else 60))
            if annotation_priority_threshold and priority < int(annotation_priority_threshold):
                continue

            order += 1
            placement = geometry.pad_box(box, padding_x, padding_y, W, H, mode=padding_mode)

            # label text: annotation display label, else caption source, else group label
            if ann_track is not None or annotations is not None or annotations_json:
                raw = _label_for(ann_track, caption_mode, label_length_mode, area_ratio,
                                 show_action=show_action_label, show_context=show_context_label,
                                 show_confidence=show_confidence, show_track_id=show_track_id,
                                 show_category=show_category, fixed_fallback=(det.label or fixed_label),
                                 track_id=key)
            else:
                raw = captions_mod.resolve_caption(
                    source=caption_source, det=det, track_id=track.track_id, order_index=order,
                    fixed_label=fixed_label, captions=captions, template=caption_template, frame_index=f)
            # A label is shown whenever a caption is intended and resolvable. Small
            # objects respect their small_object_mode caption decision; normal tracks
            # always show their label (hiding is driven by density / collision /
            # priority — NOT by the track's random caption gate, which would otherwise
            # silently drop annotation labels).
            label_text = ""
            caption_allowed = caption_source != "none" and (style.use_caption if is_small else True)
            if caption_allowed and raw:
                prefix = f"{order:02d}. " if show_track_id or caption_source == "fixed" else ""
                label_text = (prefix + raw).upper()
                if not style.use_caption:
                    style = dataclasses.replace(style, use_caption=True)

            # color
            if color_mode == "category_based":
                rgb = _category_color(category, palette, int(seed))
            else:
                _, rgb = palette_mod.color_for_track(palette, int(seed), track.track_id)
            if ann_track and ann_track.style_overrides.preferred_color:
                from yea_hud.palette import parse_hex
                pc = parse_hex(ann_track.style_overrides.preferred_color)
                if pc:
                    rgb = pc

            mv = micro_variation(int(seed), track.track_id, f, flicker_p=flicker_probability,
                                 style_flash_p=style_flash_probability, jitter_px=int(position_jitter_px),
                                 opacity_jitter=opacity_jitter)
            item = RenderItem(track_id=track.track_id, order_index=order, box=placement, color_rgb=rgb,
                              style=style, label_text=label_text, score=det.score,
                              opacity_mul=mv["opacity_mul"], line_dropout=mv["line_dropout"],
                              target_flash=mv["target_flash"], fill_flash=mv["fill_flash"], jitter=mv["jitter"],
                              priority=priority, is_primary=(category == default_category),
                              label_style="solid_bar")
            if label_text:
                tw, th = _strip_size(placement, label_text, False)
                place_reqs.append((len(items), PlacementRequest(
                    key=key, anchor=placement, tw=tw, th=th, priority=priority, is_primary=item.is_primary)))
            items.append(item)

            # --- child annotations (face / footwear / held object) ---
            if show_child_annotations and ann_track is not None:
                for child in ann_track.children:
                    cbox = child.bbox_for(f)
                    if not cbox:
                        continue
                    cbox = geometry.clamp_box(tuple(cbox), W, H)
                    if geometry.is_degenerate(cbox, 2.0):
                        continue
                    ccat = child.category
                    ccolor = _category_color(ccat, palette, int(seed))
                    ctext = (child.detailed_label or child.short_label or ccat).upper()
                    child_style = dataclasses.replace(
                        style, primitive_style="corners", use_fill=False, use_target=False,
                        use_caption=True, frame_asset_id=-1, target_asset_id=-1)
                    citem = RenderItem(track_id=f"{key}:{ccat}", order_index=order, box=cbox,
                                       color_rgb=ccolor, style=child_style, label_text=ctext,
                                       score=child.confidence, opacity_mul=mv["opacity_mul"],
                                       jitter=(0, 0), priority=max(1, priority - 40), is_child=True,
                                       label_style="micro_tag", label_color_rgb=ccolor)
                    ctw, cth = _strip_size(cbox, ctext, True)
                    place_reqs.append((len(items), PlacementRequest(
                        key=f"{key}:{ccat}", anchor=cbox, tw=ctw, th=cth,
                        priority=max(1, priority - 40), is_primary=False)))
                    items.append(citem)

        # --- frame-level annotations (e.g. ball) for this frame ---
        for fa in aset.frame_list(f):
            if not fa.bbox:
                continue
            fbox = geometry.clamp_box(tuple(fa.bbox), W, H)
            if geometry.is_degenerate(fbox, 2.0):
                continue
            fcolor = _category_color(fa.category, palette, int(seed))
            ftext = (fa.detailed_label or fa.short_label or fa.category).upper()
            from yea_hud.types import TrackStyle
            fstyle = TrackStyle(primitive_style="corners", use_caption=True, use_target=False)
            fitem = RenderItem(track_id=fa.annotation_id, order_index=order + 1, box=fbox,
                               color_rgb=fcolor, label_text=ftext, style=fstyle, score=fa.confidence,
                               priority=int(fa.priority), is_child=True, label_style="micro_tag",
                               label_color_rgb=fcolor)
            ftw, fth = _strip_size(fbox, ftext, True)
            place_reqs.append((len(items), PlacementRequest(
                key=str(fa.annotation_id), anchor=fbox, tw=ftw, th=fth, priority=int(fa.priority),
                is_primary=False)))
            items.append(fitem)

        # --- resolve label placement (collision + temporal stability) ---
        if place_reqs:
            placed = placer.place([pr for _, pr in place_reqs])
            # density / per-frame cap: keep the highest-priority labels visible
            ranked_lbl = sorted(range(len(place_reqs)),
                                key=lambda k: -place_reqs[k][1].priority)
            keep = set(ranked_lbl[:eff_max_labels])
            for rank_i, ((item_idx, _pr), pl) in enumerate(zip(place_reqs, placed)):
                vis = pl.visible and (rank_i in keep)
                items[item_idx].label_rect = pl.rect
                items[item_idx].label_visible = vis
        plans.append(items)
        if progress is not None:
            try:
                progress.update(1)
            except Exception:
                pass

    params = {
        "frame_opacity": frame_opacity, "fill_opacity": fill_opacity, "label_opacity": label_opacity,
        "line_width": int(line_width), "target_scale": target_scale, "label_scale": label_scale,
        "label_text_color_mode": label_text_color_mode, "detail_density": detail_density,
    }
    # Resolve the label font: an absolute path, or a filename in yea_hud/fonts/.
    font_path = None
    if font:
        import os as _os
        from yea_hud import text_render as _tr
        if _os.path.isabs(font) and _os.path.exists(font):
            font_path = font
        else:
            _cand = _os.path.join(_os.path.dirname(_os.path.abspath(_tr.__file__)), "fonts", font)
            font_path = _cand if _os.path.exists(_cand) else None
    images = render_frames(frames, plans, assets=assets, palette=palette, params=params,
                           font_path=font_path, progress=None, interrupt=interrupt)

    preview_idx = next((i for i, p in enumerate(plans) if p), 0)
    debug_preview = images[preview_idx:preview_idx + 1]

    tracking = {"frames": B, "detections": int(total_dets), "tracks": len(seen_tracks),
                "source": source, "style_mode": style_mode, "seed": int(seed),
                "image_size": [W, H], "annotated_tracks": len(ann_by_id),
                "caption_mode": caption_mode}
    tracking.update(meta)
    tracking_json = json.dumps(tracking, ensure_ascii=False, default=str)
    return images, debug_preview, tracking_json


# --- V3 node (primary) ------------------------------------------------------
try:
    from comfy_api.latest import io
    from yea_hud.types import (
        STYLE_MODES, SMALL_OBJECT_MODES, CAPTION_SOURCES, LABEL_TEXT_COLOR_MODES,
        ASSET_FIT_MODES, PADDING_MODES, DEFAULTS,
    )
    from yea_hud.annotations import CAPTION_MODES
    from yea_hud.label_placement import COLLISION_MODES

    SAM3TrackData = io.Custom(TRACK_DATA_TYPE)
    YEAAnnotations = io.Custom(ANNOTATIONS_TYPE)
    YEAAssetPack = io.Custom("YEA_HUD_ASSET_PACK")
    LABEL_LENGTH_MODES = ["single_line", "two_line", "truncate", "adaptive"]
    COLOR_MODES = ["track_random", "category_based", "source_based", "fixed_palette", "annotation_override"]

    def _asset_inputs():
        ins = []
        for i in range(1, 7):
            ins.append(io.Image.Input(f"frame_{i}_image", optional=True,
                                      tooltip=f"White-on-transparent HUD frame PNG #{i} (IMAGE)."))
            ins.append(io.Mask.Input(f"frame_{i}_mask", optional=True,
                                     tooltip=f"MASK output of Load Image for frame PNG #{i}."))
        for i in range(1, 3):
            ins.append(io.Image.Input(f"target_{i}_image", optional=True))
            ins.append(io.Mask.Input(f"target_{i}_mask", optional=True))
        return ins

    class YEAHUDRenderer(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID, display_name=DISPLAY_NAME, category=CATEGORY,
                description=(
                    "Render designed, densely-labelled HUD overlays from native SAM3 tracking. "
                    "Prefers SAM3_TRACK_DATA (stable native identities); falls back to BOUNDING_BOX "
                    "/ MASK. Attaches stable semantic labels from YEA_ANNOTATIONS / JSON (never "
                    "invented here), renders primary + child annotations with collision-aware label "
                    "placement, recolors transparent PNG assets, and outputs an IMAGE batch."
                ),
                search_aliases=["hud", "sam3", "overlay", "tracking", "caption", "annotation", "label"],
                inputs=[
                    io.Image.Input("frames", tooltip="Decoded video frames [B,H,W,3]."),
                    io.Int.Input("seed", default=DEFAULTS["seed"], min=0, max=0xFFFFFFFFFFFF, control_after_generate=True),
                    io.Combo.Input("style_mode", options=list(STYLE_MODES), default="mixed"),
                    io.Int.Input("max_objects", default=DEFAULTS["max_objects"], min=0, max=256),
                    io.Float.Input("display_probability", default=DEFAULTS["display_probability"], min=0.0, max=1.0, step=0.01),
                    io.Int.Input("padding_x", default=DEFAULTS["padding_x"], min=-256, max=512),
                    io.Int.Input("padding_y", default=DEFAULTS["padding_y"], min=-256, max=512),
                    io.Float.Input("smoothing", default=DEFAULTS["smoothing"], min=0.0, max=0.98, step=0.01),
                    io.Float.Input("track_iou_threshold", default=DEFAULTS["track_iou_threshold"], min=0.0, max=1.0, step=0.01),
                    io.Int.Input("track_memory_frames", default=DEFAULTS["track_memory_frames"], min=0, max=120),
                    io.Float.Input("frame_opacity", default=DEFAULTS["frame_opacity"], min=0.0, max=1.0, step=0.01),
                    io.Float.Input("fill_opacity", default=DEFAULTS["fill_opacity"], min=0.0, max=1.0, step=0.01),
                    io.Float.Input("label_opacity", default=DEFAULTS["label_opacity"], min=0.0, max=1.0, step=0.01),
                    io.Int.Input("line_width", default=DEFAULTS["line_width"], min=1, max=32),
                    io.Float.Input("target_scale", default=DEFAULTS["target_scale"], min=0.01, max=1.0, step=0.01),
                    io.Float.Input("label_scale", default=DEFAULTS["label_scale"], min=0.25, max=4.0, step=0.05),
                    # --- tracking sources (priority track_data > bboxes > masks) ---
                    SAM3TrackData.Input("track_data", optional=True, tooltip="PREFERRED primary SAM3_TRACK_DATA."),
                    SAM3TrackData.Input("secondary_track_data_1", optional=True, tooltip="Optional 2nd SAM3 group (e.g. face)."),
                    io.String.Input("primary_group_label", default="PERSON", optional=True),
                    io.String.Input("secondary_group_label_1", default="", optional=True),
                    io.BoundingBox.Input("bboxes", optional=True, force_input=True, tooltip="Fallback BOUNDING_BOX."),
                    io.Mask.Input("masks", optional=True, tooltip="Fallback / supplementary MASK."),
                    # --- annotations (semantic labels) ---
                    YEAAnnotations.Input("annotations", optional=True, tooltip="PREFERRED. From Annotation Builder / JSON."),
                    io.String.Input("annotations_json", default="", multiline=True, optional=True,
                                    tooltip="Inline YEA_ANNOTATIONS JSON (used if no annotations object)."),
                    io.Combo.Input("caption_mode", options=list(CAPTION_MODES), default="auto", optional=True),
                    io.Combo.Input("label_length_mode", options=LABEL_LENGTH_MODES, default="adaptive", optional=True),
                    io.Combo.Input("label_collision_mode", options=list(COLLISION_MODES), default="reposition_and_shorten", optional=True),
                    io.Float.Input("label_density", default=1.0, min=0.0, max=1.0, step=0.05, optional=True),
                    io.Int.Input("max_labels_per_frame", default=0, min=0, max=200, optional=True, tooltip="0 = derive from label_density."),
                    io.Boolean.Input("show_child_annotations", default=True, optional=True),
                    io.Boolean.Input("show_action_label", default=True, optional=True),
                    io.Boolean.Input("show_context_label", default=False, optional=True),
                    io.Boolean.Input("show_confidence", default=False, optional=True),
                    io.Boolean.Input("show_track_id", default=False, optional=True),
                    io.Float.Input("secondary_annotation_scale", default=0.6, min=0.2, max=1.0, step=0.05, optional=True),
                    io.Int.Input("annotation_priority_threshold", default=0, min=0, max=200, optional=True),
                    io.Combo.Input("color_mode", options=COLOR_MODES, default="track_random", optional=True),
                    io.String.Input("font", default="", optional=True,
                                    tooltip="Label font: a filename in yea_hud/fonts/ or an absolute .ttf/.otf path. Empty = bundled/system default."),
                    YEAAssetPack.Input("asset_pack", optional=True,
                                       tooltip="From 'YEA HUD Asset Pack' — your 6 frame + 2 target PNGs in one input."),
                    # --- captions (fallback when no annotations) / palette ---
                    io.Combo.Input("caption_source", options=list(CAPTION_SOURCES), default="fixed", optional=True),
                    io.String.Input("fixed_label", default="PERSON", optional=True),
                    io.String.Input("captions_json", default="", multiline=True, optional=True),
                    io.String.Input("caption_template", default="{track:02d}. {label}", optional=True),
                    io.String.Input("palette_json", default="", optional=True),
                    io.Combo.Input("small_object_mode", options=list(SMALL_OBJECT_MODES), default="target_only", optional=True),
                    io.Combo.Input("label_text_color_mode", options=list(LABEL_TEXT_COLOR_MODES), default="auto", optional=True),
                    io.Combo.Input("asset_fit_mode", options=list(ASSET_FIT_MODES), default="stretch", optional=True),
                    io.Boolean.Input("trim_transparent_assets", default=True, optional=True),
                    io.Combo.Input("padding_mode", options=list(PADDING_MODES), default="pixels", optional=True),
                    io.Float.Input("flicker_probability", default=DEFAULTS["flicker_probability"], min=0.0, max=1.0, step=0.01, optional=True),
                    io.Int.Input("position_jitter_px", default=DEFAULTS["position_jitter_px"], min=0, max=64, optional=True),
                    io.Float.Input("min_box_area_ratio", default=DEFAULTS["min_box_area_ratio"], min=0.0, max=1.0, step=0.0001, optional=True),
                ],
                outputs=[
                    io.Image.Output("images", display_name="images"),
                    io.Image.Output("debug_preview", display_name="debug_preview"),
                    io.String.Output("tracking_json", display_name="tracking_json"),
                ],
            )

        @classmethod
        def execute(cls, frames, seed, style_mode, max_objects, display_probability, padding_x,
                    padding_y, smoothing, track_iou_threshold, track_memory_frames, frame_opacity,
                    fill_opacity, label_opacity, line_width, target_scale, label_scale=1.0,
                    track_data=None, secondary_track_data_1=None, primary_group_label="PERSON",
                    secondary_group_label_1="",
                    bboxes=None, masks=None, annotations=None, annotations_json="", caption_mode="auto",
                    label_length_mode="adaptive", label_collision_mode="reposition_and_shorten",
                    label_density=1.0, max_labels_per_frame=0, show_child_annotations=True,
                    show_action_label=True, show_context_label=False, show_confidence=False,
                    show_track_id=False, secondary_annotation_scale=0.6, annotation_priority_threshold=0,
                    color_mode="track_random", caption_source="fixed", fixed_label="PERSON",
                    captions_json="", caption_template="{track:02d}. {label}", palette_json="",
                    small_object_mode="target_only", label_text_color_mode="auto", asset_fit_mode="stretch",
                    trim_transparent_assets=True, padding_mode="pixels", flicker_probability=0.04,
                    position_jitter_px=0, min_box_area_ratio=0.0004, font="", asset_pack=None) -> io.NodeOutput:
            import comfy.utils
            import comfy.model_management as mm
            B = int(frames.shape[0])
            pbar = comfy.utils.ProgressBar(B)
            secondary_track_data = [secondary_track_data_1]
            group_labels = [secondary_group_label_1]
            images, debug_preview, tracking_json = run_hud_renderer(
                frames, seed=seed, style_mode=style_mode, max_objects=max_objects,
                display_probability=display_probability, padding_x=padding_x, padding_y=padding_y,
                smoothing=smoothing, track_iou_threshold=track_iou_threshold,
                track_memory_frames=track_memory_frames, frame_opacity=frame_opacity,
                fill_opacity=fill_opacity, label_opacity=label_opacity, line_width=line_width,
                target_scale=target_scale, label_scale=label_scale, track_data=track_data,
                secondary_track_data=secondary_track_data, group_labels=group_labels,
                primary_group_label=primary_group_label, bboxes=bboxes, masks=masks,
                asset_pack=asset_pack,
                captions_json=captions_json, fixed_label=fixed_label, palette_json=palette_json,
                caption_source=caption_source, caption_template=caption_template,
                small_object_mode=small_object_mode, label_text_color_mode=label_text_color_mode,
                trim_transparent_assets=trim_transparent_assets, asset_fit_mode=asset_fit_mode,
                padding_mode=padding_mode, flicker_probability=flicker_probability,
                position_jitter_px=position_jitter_px, min_box_area_ratio=min_box_area_ratio,
                annotations=annotations, annotations_json=annotations_json, caption_mode=caption_mode,
                label_density=label_density, max_labels_per_frame=max_labels_per_frame,
                label_collision_mode=label_collision_mode, label_length_mode=label_length_mode,
                show_confidence=show_confidence, show_track_id=show_track_id,
                show_child_annotations=show_child_annotations, show_action_label=show_action_label,
                show_context_label=show_context_label, secondary_annotation_scale=secondary_annotation_scale,
                annotation_priority_threshold=annotation_priority_threshold, color_mode=color_mode,
                font=font, default_category="person", progress=pbar,
                interrupt=mm.throw_exception_if_processing_interrupted)
            return io.NodeOutput(images, debug_preview, tracking_json)

    NODE = YEAHUDRenderer
except ImportError:  # pragma: no cover
    NODE = None
