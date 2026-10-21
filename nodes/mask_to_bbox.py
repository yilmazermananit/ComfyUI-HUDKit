"""HUDKit Mask to Native BBoxes — utility node.

Converts a native MASK batch into a native BOUNDING_BOX value (the same
list[list[dict{x,y,width,height,score}]] shape SAM3 emits) using connected
components. Lets the renderer's bbox path be exercised when only masks are
available, with NO third-party bbox type (no BBOX / SEGS / Impact / KJNodes).
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("YEAHUD.mask_to_bbox")

DISPLAY_NAME = "HUDKit Mask to Native BBoxes"
NODE_ID = "YEAMaskToNativeBBoxes"
CATEGORY = "HUDKit"


def run_mask_to_bbox(masks, threshold: float = 0.5, min_area: int = 64,
                     merge_distance: int = 0):
    """Return (native_bboxes, debug_json_str)."""
    from yea_hud.mask_adapter import masks_to_native_bboxes

    bboxes = masks_to_native_bboxes(masks, threshold=threshold, min_area=min_area,
                                    merge_distance=merge_distance)
    counts = [len(f) for f in bboxes] if isinstance(bboxes, list) else []
    summary = {
        "frames": len(counts),
        "detections_per_frame": counts[:50],
        "total_detections": int(sum(counts)),
        "params": {"threshold": threshold, "min_area": min_area, "merge_distance": merge_distance},
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    logger.info("HUDKit Mask to Native BBoxes: %s", text)
    return bboxes, text


try:
    from comfy_api.latest import io

    class YEAMaskToNativeBBoxes(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Convert a MASK batch to a native BOUNDING_BOX value via connected "
                    "components (no third-party bbox types). Fallback for the renderer when "
                    "no native bboxes / track data are available."
                ),
                search_aliases=["mask", "bbox", "bounding box", "connected components"],
                inputs=[
                    io.Mask.Input("masks", tooltip="MASK batch [B,H,W] (or [H,W])."),
                    io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01),
                    io.Int.Input("min_area", default=64, min=0, max=1_000_000,
                                 tooltip="Drop components smaller than this many pixels."),
                    io.Int.Input("merge_distance", default=0, min=0, max=4096,
                                 tooltip="Merge component boxes whose gap is within this many pixels."),
                ],
                outputs=[
                    io.BoundingBox.Output("bboxes", display_name="bboxes"),
                    io.String.Output("debug_json", display_name="debug_json"),
                ],
            )

        @classmethod
        def execute(cls, masks, threshold=0.5, min_area=64, merge_distance=0) -> io.NodeOutput:
            bboxes, text = run_mask_to_bbox(masks, threshold, min_area, merge_distance)
            return io.NodeOutput(bboxes, text)

    NODE = YEAMaskToNativeBBoxes
except ImportError:  # pragma: no cover
    NODE = None
