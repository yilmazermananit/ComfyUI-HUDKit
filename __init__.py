"""ComfyUI-YEAHUD — node registration entry point.

Prefers the V3 schema (``comfy_api`` + ``comfy_entrypoint``), exactly like the
native SAM3 nodes. If ``comfy_api`` is unavailable (older ComfyUI), falls back to
V1 ``NODE_CLASS_MAPPINGS``. The loader prefers NODE_CLASS_MAPPINGS over
comfy_entrypoint, so we define ONLY ONE of them depending on what's available.
Business logic lives in ``yea_hud``; this file does registration only.

ComfyUI loads custom-node packages via ``importlib.spec_from_file_location`` under a
mangled module name and does NOT add the package directory to ``sys.path``. Our
node modules use absolute imports (``from yea_hud import ...``), so we insert the
package directory onto ``sys.path`` first — the standard custom-node pattern.
"""

import logging
import os
import sys

logger = logging.getLogger("YEAHUD")

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

# Optional frontend (UX only — rendering works without it).
WEB_DIRECTORY = "./web/js"

__all__ = ["WEB_DIRECTORY"]

try:
    from comfy_api.latest import ComfyExtension  # noqa: F401
    _HAS_V3 = True
except Exception:  # pragma: no cover - exercised on old ComfyUI
    _HAS_V3 = False

if _HAS_V3:
    from nodes.hud_renderer import NODE as _Renderer
    from nodes.inspect_track_data import NODE as _InspectTrack
    from nodes.inspect_bbox import NODE as _InspectBBox
    from nodes.mask_to_bbox import NODE as _MaskToBBox
    from nodes.annotation_json import NODE as _AnnotationJSON
    from nodes.annotation_builder import NODE as _AnnotationBuilder
    from nodes.track_crop_extractor import NODE as _CropExtractor
    from nodes.detailed_label_prompt import NODE as _LabelPrompt

    _NODES = [n for n in (_Renderer, _InspectTrack, _InspectBBox, _MaskToBBox,
                          _AnnotationJSON, _AnnotationBuilder, _CropExtractor, _LabelPrompt)
              if n is not None]

    class YEAHUDExtension(ComfyExtension):
        async def get_node_list(self):
            return _NODES

    async def comfy_entrypoint() -> "YEAHUDExtension":
        return YEAHUDExtension()

    __all__ += ["comfy_entrypoint"]
    logger.info("ComfyUI-YEAHUD: registered %d V3 nodes (category YEA/HUD).", len(_NODES))
else:
    from nodes.v1_compat import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401

    __all__ += ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
    logger.info("ComfyUI-YEAHUD: comfy_api V3 unavailable; registered V1 NODE_CLASS_MAPPINGS.")
