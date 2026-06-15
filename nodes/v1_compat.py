"""V1 compatibility shim — used ONLY when ``comfy_api`` (V3 schema) is unavailable.

ComfyUI's loader prefers ``NODE_CLASS_MAPPINGS`` over ``comfy_entrypoint``, so the
top-level ``__init__`` imports these mappings ONLY in the V1 branch. No rendering
logic lives here — every node calls the shared ``run_*`` functions, so there is a
single source of truth (brief §20). All 8 nodes are mirrored.
"""

from __future__ import annotations

from yonder_hud.types import (
    STYLE_MODES, SMALL_OBJECT_MODES, CAPTION_SOURCES, LABEL_TEXT_COLOR_MODES,
    ASSET_FIT_MODES, PADDING_MODES, DEFAULTS,
)
from yonder_hud.annotations import CAPTION_MODES
from yonder_hud.label_placement import COLLISION_MODES
from .hud_renderer import run_hud_renderer, DISPLAY_NAME as R_NAME, NODE_ID as R_ID, CATEGORY
from .inspect_bbox import run_inspect_bbox, DISPLAY_NAME as IB_NAME, NODE_ID as IB_ID
from .inspect_track_data import run_inspect_track_data, DISPLAY_NAME as IT_NAME, NODE_ID as IT_ID
from .mask_to_bbox import run_mask_to_bbox, DISPLAY_NAME as M_NAME, NODE_ID as M_ID
from .annotation_json import run_annotation_json, DISPLAY_NAME as AJ_NAME, NODE_ID as AJ_ID
from .annotation_builder import run_annotation_builder, DISPLAY_NAME as AB_NAME, NODE_ID as AB_ID
from .track_crop_extractor import run_track_crop_extractor, DISPLAY_NAME as CE_NAME, NODE_ID as CE_ID
from .detailed_label_prompt import run_detailed_label_prompt, DISPLAY_NAME as LP_NAME, NODE_ID as LP_ID

ANN_CAT = "Yonder/HUD/Annotation"
_F = lambda v: ("FLOAT", {"default": v, "min": 0.0, "max": 1.0, "step": 0.01})
LABEL_LENGTH_MODES = ["single_line", "two_line", "truncate", "adaptive"]
COLOR_MODES = ["track_random", "category_based", "source_based", "fixed_palette", "annotation_override"]


class YonderHUDRendererV1:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images", "debug_preview", "tracking_json")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "frames": ("IMAGE",),
            "seed": ("INT", {"default": DEFAULTS["seed"], "min": 0, "max": 0xFFFFFFFFFFFF}),
            "style_mode": (list(STYLE_MODES), {"default": "mixed"}),
            "max_objects": ("INT", {"default": DEFAULTS["max_objects"], "min": 0, "max": 256}),
            "display_probability": _F(DEFAULTS["display_probability"]),
            "padding_x": ("INT", {"default": DEFAULTS["padding_x"], "min": -256, "max": 512}),
            "padding_y": ("INT", {"default": DEFAULTS["padding_y"], "min": -256, "max": 512}),
            "smoothing": ("FLOAT", {"default": DEFAULTS["smoothing"], "min": 0.0, "max": 0.98, "step": 0.01}),
            "track_iou_threshold": _F(DEFAULTS["track_iou_threshold"]),
            "track_memory_frames": ("INT", {"default": DEFAULTS["track_memory_frames"], "min": 0, "max": 120}),
            "frame_opacity": _F(DEFAULTS["frame_opacity"]),
            "fill_opacity": _F(DEFAULTS["fill_opacity"]),
            "label_opacity": _F(DEFAULTS["label_opacity"]),
            "line_width": ("INT", {"default": DEFAULTS["line_width"], "min": 1, "max": 32}),
            "target_scale": ("FLOAT", {"default": DEFAULTS["target_scale"], "min": 0.01, "max": 1.0, "step": 0.01}),
            "label_scale": ("FLOAT", {"default": DEFAULTS["label_scale"], "min": 0.25, "max": 4.0, "step": 0.05}),
        }
        optional = {
            "track_data": ("SAM3_TRACK_DATA",),
            "secondary_track_data_1": ("SAM3_TRACK_DATA",),
            "secondary_track_data_2": ("SAM3_TRACK_DATA",),
            "secondary_track_data_3": ("SAM3_TRACK_DATA",),
            "primary_group_label": ("STRING", {"default": "PERSON"}),
            "secondary_group_label_1": ("STRING", {"default": ""}),
            "secondary_group_label_2": ("STRING", {"default": ""}),
            "secondary_group_label_3": ("STRING", {"default": ""}),
            "bboxes": ("BOUNDING_BOX",),
            "masks": ("MASK",),
            "annotations": ("YONDER_ANNOTATIONS",),
            "annotations_json": ("STRING", {"default": "", "multiline": True}),
            "caption_mode": (list(CAPTION_MODES), {"default": "auto"}),
            "label_length_mode": (LABEL_LENGTH_MODES, {"default": "adaptive"}),
            "label_collision_mode": (list(COLLISION_MODES), {"default": "reposition_and_shorten"}),
            "label_density": _F(1.0),
            "max_labels_per_frame": ("INT", {"default": 0, "min": 0, "max": 200}),
            "show_child_annotations": ("BOOLEAN", {"default": True}),
            "show_action_label": ("BOOLEAN", {"default": True}),
            "show_context_label": ("BOOLEAN", {"default": False}),
            "show_confidence": ("BOOLEAN", {"default": False}),
            "show_track_id": ("BOOLEAN", {"default": False}),
            "annotation_priority_threshold": ("INT", {"default": 0, "min": 0, "max": 200}),
            "color_mode": (COLOR_MODES, {"default": "track_random"}),
            "caption_source": (list(CAPTION_SOURCES), {"default": "fixed"}),
            "fixed_label": ("STRING", {"default": "PERSON"}),
            "captions_json": ("STRING", {"default": "", "multiline": True}),
            "caption_template": ("STRING", {"default": "{track:02d}. {label}"}),
            "palette_json": ("STRING", {"default": ""}),
            "small_object_mode": (list(SMALL_OBJECT_MODES), {"default": "target_only"}),
            "label_text_color_mode": (list(LABEL_TEXT_COLOR_MODES), {"default": "auto"}),
            "asset_fit_mode": (list(ASSET_FIT_MODES), {"default": "stretch"}),
            "trim_transparent_assets": ("BOOLEAN", {"default": True}),
            "padding_mode": (list(PADDING_MODES), {"default": "pixels"}),
            "flicker_probability": _F(DEFAULTS["flicker_probability"]),
            "position_jitter_px": ("INT", {"default": 0, "min": 0, "max": 64}),
            "min_box_area_ratio": ("FLOAT", {"default": DEFAULTS["min_box_area_ratio"], "min": 0.0, "max": 1.0, "step": 0.0001}),
        }
        for i in range(1, 7):
            optional[f"frame_{i}_image"] = ("IMAGE",)
            optional[f"frame_{i}_mask"] = ("MASK",)
        for i in range(1, 3):
            optional[f"target_{i}_image"] = ("IMAGE",)
            optional[f"target_{i}_mask"] = ("MASK",)
        return {"required": required, "optional": optional}

    def run(self, frames, seed, style_mode, max_objects, display_probability, padding_x, padding_y,
            smoothing, track_iou_threshold, track_memory_frames, frame_opacity, fill_opacity,
            label_opacity, line_width, target_scale, label_scale=1.0, track_data=None,
            secondary_track_data_1=None, secondary_track_data_2=None, secondary_track_data_3=None,
            primary_group_label="PERSON", secondary_group_label_1="", secondary_group_label_2="",
            secondary_group_label_3="", bboxes=None, masks=None, annotations=None, annotations_json="",
            caption_mode="auto", label_length_mode="adaptive", label_collision_mode="reposition_and_shorten",
            label_density=1.0, max_labels_per_frame=0, show_child_annotations=True, show_action_label=True,
            show_context_label=False, show_confidence=False, show_track_id=False,
            annotation_priority_threshold=0, color_mode="track_random", caption_source="fixed",
            fixed_label="PERSON", captions_json="", caption_template="{track:02d}. {label}", palette_json="",
            small_object_mode="target_only", label_text_color_mode="auto", asset_fit_mode="stretch",
            trim_transparent_assets=True, padding_mode="pixels", flicker_probability=0.04,
            position_jitter_px=0, min_box_area_ratio=0.0004, **assets):
        import comfy.utils
        import comfy.model_management as mm
        pbar = comfy.utils.ProgressBar(int(frames.shape[0]))
        return run_hud_renderer(
            frames, seed=seed, style_mode=style_mode, max_objects=max_objects,
            display_probability=display_probability, padding_x=padding_x, padding_y=padding_y,
            smoothing=smoothing, track_iou_threshold=track_iou_threshold,
            track_memory_frames=track_memory_frames, frame_opacity=frame_opacity,
            fill_opacity=fill_opacity, label_opacity=label_opacity, line_width=line_width,
            target_scale=target_scale, label_scale=label_scale, track_data=track_data,
            secondary_track_data=[secondary_track_data_1, secondary_track_data_2, secondary_track_data_3],
            group_labels=[secondary_group_label_1, secondary_group_label_2, secondary_group_label_3],
            primary_group_label=primary_group_label, bboxes=bboxes, masks=masks,
            frame_images=[assets.get(f"frame_{i}_image") for i in range(1, 7)],
            frame_masks=[assets.get(f"frame_{i}_mask") for i in range(1, 7)],
            target_images=[assets.get(f"target_{i}_image") for i in range(1, 3)],
            target_masks=[assets.get(f"target_{i}_mask") for i in range(1, 3)],
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
            show_context_label=show_context_label, annotation_priority_threshold=annotation_priority_threshold,
            color_mode=color_mode, progress=pbar,
            interrupt=mm.throw_exception_if_processing_interrupted)


class YonderInspectBoundingBoxesV1:
    CATEGORY = CATEGORY; FUNCTION = "run"
    RETURN_TYPES = ("STRING", "BOUNDING_BOX"); RETURN_NAMES = ("debug_json", "bboxes"); OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"bboxes": ("BOUNDING_BOX",), "masks": ("MASK",)}}

    def run(self, bboxes=None, masks=None):
        return (run_inspect_bbox(bboxes, masks), bboxes)


class YonderInspectSAM3TrackDataV1:
    CATEGORY = CATEGORY; FUNCTION = "run"
    RETURN_TYPES = ("STRING", "SAM3_TRACK_DATA"); RETURN_NAMES = ("debug_json", "track_data"); OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"track_data": ("SAM3_TRACK_DATA",)}}

    def run(self, track_data=None):
        return (run_inspect_track_data(track_data), track_data)


class YonderMaskToNativeBBoxesV1:
    CATEGORY = CATEGORY; FUNCTION = "run"
    RETURN_TYPES = ("BOUNDING_BOX", "STRING"); RETURN_NAMES = ("bboxes", "debug_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "masks": ("MASK",),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "min_area": ("INT", {"default": 64, "min": 0, "max": 1_000_000}),
            "merge_distance": ("INT", {"default": 0, "min": 0, "max": 4096}),
        }}

    def run(self, masks, threshold=0.5, min_area=64, merge_distance=0):
        return run_mask_to_bbox(masks, threshold, min_area, merge_distance)


class YonderAnnotationJSONV1:
    CATEGORY = ANN_CAT; FUNCTION = "run"
    RETURN_TYPES = ("YONDER_ANNOTATIONS", "STRING", "STRING")
    RETURN_NAMES = ("annotations", "normalized_json", "validation_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"annotations_json": ("STRING", {"default": "", "multiline": True})},
                "optional": {"strict": ("BOOLEAN", {"default": False})}}

    def run(self, annotations_json, strict=False):
        return run_annotation_json(annotations_json, strict=strict)


class YonderAnnotationBuilderV1:
    CATEGORY = ANN_CAT; FUNCTION = "run"
    RETURN_TYPES = ("YONDER_ANNOTATIONS", "STRING", "STRING")
    RETURN_NAMES = ("annotations", "annotations_json", "validation_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "track_data": ("SAM3_TRACK_DATA",),
            "crop_manifest_json": ("STRING", {"default": "", "multiline": True}),
            "captions_json": ("STRING", {"default": "", "multiline": True}),
            "manual_annotations_json": ("STRING", {"default": "", "multiline": True}),
            "secondary_detections_json": ("STRING", {"default": "", "multiline": True}),
            "default_category": ("STRING", {"default": "person"}),
            "primary_group_label": ("STRING", {"default": "PERSON"}),
            "label_template": ("STRING", {"default": ""}),
        }}

    def run(self, track_data=None, crop_manifest_json="", captions_json="", manual_annotations_json="",
            secondary_detections_json="", default_category="person", primary_group_label="PERSON",
            label_template=""):
        return run_annotation_builder(track_data, crop_manifest_json, captions_json,
                                      manual_annotations_json, secondary_detections_json,
                                      default_category, label_template, primary_group_label)


class YonderTrackCropExtractorV1:
    CATEGORY = ANN_CAT; FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("crops", "crop_manifest_json", "contact_sheet", "track_indices_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "track_data": ("SAM3_TRACK_DATA",),
            "samples_per_track": ("INT", {"default": 3, "min": 1, "max": 12}),
            "crop_padding": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "minimum_track_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0005}),
            "selection_mode": (["best", "largest", "sharpest", "most_centered", "multi_sample"], {"default": "best"}),
        }}

    def run(self, frames, track_data, samples_per_track=3, crop_padding=0.15, minimum_track_area=0.0,
            selection_mode="best"):
        return run_track_crop_extractor(frames, track_data, samples_per_track, crop_padding,
                                        minimum_track_area, selection_mode)


class YonderDetailedLabelPromptV1:
    CATEGORY = ANN_CAT; FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt", "json_schema")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "detail_level": (["short", "medium", "detailed"], {"default": "detailed"}),
            "include_clothing": ("BOOLEAN", {"default": True}),
            "include_action": ("BOOLEAN", {"default": True}),
            "include_objects": ("BOOLEAN", {"default": True}),
            "include_scene": ("BOOLEAN", {"default": True}),
            "include_visible_details": ("BOOLEAN", {"default": True}),
            "maximum_words": ("INT", {"default": 12, "min": 2, "max": 40}),
            "uppercase": ("BOOLEAN", {"default": True}),
        }}

    def run(self, detail_level="detailed", include_clothing=True, include_action=True,
            include_objects=True, include_scene=True, include_visible_details=True,
            maximum_words=12, uppercase=True):
        return run_detailed_label_prompt(detail_level, include_clothing, include_action,
                                         include_objects, include_scene, include_visible_details,
                                         maximum_words, uppercase)


NODE_CLASS_MAPPINGS = {
    R_ID: YonderHUDRendererV1,
    IT_ID: YonderInspectSAM3TrackDataV1,
    IB_ID: YonderInspectBoundingBoxesV1,
    M_ID: YonderMaskToNativeBBoxesV1,
    AJ_ID: YonderAnnotationJSONV1,
    AB_ID: YonderAnnotationBuilderV1,
    CE_ID: YonderTrackCropExtractorV1,
    LP_ID: YonderDetailedLabelPromptV1,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    R_ID: R_NAME, IT_ID: IT_NAME, IB_ID: IB_NAME, M_ID: M_NAME,
    AJ_ID: AJ_NAME, AB_ID: AB_NAME, CE_ID: CE_NAME, LP_ID: LP_NAME,
}
