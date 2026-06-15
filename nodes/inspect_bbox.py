"""YEA Inspect Bounding Boxes — debug node.

Logs and returns a safe structural summary of a native BOUNDING_BOX value and
passes the bboxes through unchanged. Use when a SAM3 / detection node's bbox
output does not behave as expected, to see the REAL runtime structure instead of
guessing. Production-safe: bounded output (first frames / first detections only).
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("YEAHUD.inspect_bbox")

DISPLAY_NAME = "YEA Inspect Bounding Boxes"
NODE_ID = "YEAInspectBoundingBoxes"
CATEGORY = "YEA/HUD"


def run_inspect_bbox(bboxes=None, masks=None, max_frames: int = 3, max_dets: int = 10) -> str:
    """Return a compact JSON string describing the bbox runtime structure.

    Pure / schema-agnostic so both the V3 node and the V1 shim can call it.
    """
    from yea_hud.bbox_adapter import summarize_bboxes

    summary = summarize_bboxes(bboxes, max_frames=max_frames, max_dets=max_dets)

    if masks is not None:
        try:
            import torch  # noqa: F401
            shape = tuple(getattr(masks, "shape", ()))
            summary["masks"] = {
                "python_type": type(masks).__name__,
                "shape": list(shape),
                "dtype": str(getattr(masks, "dtype", None)),
                "ndim": len(shape),
            }
        except Exception as exc:  # pragma: no cover - defensive
            summary["masks"] = {"error": repr(exc)}

    text = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    logger.info("YEA Inspect Bounding Boxes:\n%s", text)
    return text


# --- V3 node (primary) ------------------------------------------------------
try:
    from comfy_api.latest import io, ui

    class YEAInspectBoundingBoxes(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Inspect the real runtime Python structure of a native BOUNDING_BOX "
                    "value (type, nesting, dict keys, per-frame detection counts, coordinate "
                    "format) and pass it through unchanged."
                ),
                search_aliases=["bbox", "debug", "inspect", "sam3"],
                inputs=[
                    io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                         tooltip="Native BOUNDING_BOX (e.g. from SAM3 Detect)."),
                    io.Mask.Input("masks", optional=True, tooltip="Optional MASK to summarize alongside."),
                ],
                outputs=[
                    io.String.Output("debug_json", display_name="debug_json"),
                    io.BoundingBox.Output("bboxes", display_name="bboxes"),
                ],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, bboxes=None, masks=None) -> io.NodeOutput:
            text = run_inspect_bbox(bboxes, masks)
            return io.NodeOutput(text, bboxes, ui=ui.PreviewText(text))

    NODE = YEAInspectBoundingBoxes
except ImportError:  # pragma: no cover - V1 environments
    NODE = None
