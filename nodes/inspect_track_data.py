"""YEA Inspect SAM3 Track Data — debug node.

Logs and returns a safe structural summary of a native SAM3_TRACK_DATA value
(the dict produced by SAM3 Video Track: packed_masks / n_frames / scores /
orig_size) and passes it through unchanged. This is the node to run first when
wiring a SAM3 video-tracking workflow, to confirm the REAL runtime structure of
the track data on the target ComfyUI / Comfy Cloud build before relying on it.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("YEAHUD.inspect_track_data")

DISPLAY_NAME = "YEA Inspect SAM3 Track Data"
NODE_ID = "YEAInspectSAM3TrackData"
CATEGORY = "YEA/HUD"

# Native custom type string verified from comfy_extras/nodes_sam3.py:
#   SAM3TrackData = io.Custom("SAM3_TRACK_DATA")
TRACK_DATA_TYPE = "SAM3_TRACK_DATA"


def run_inspect_track_data(track_data=None) -> str:
    """Return a compact JSON string describing the SAM3_TRACK_DATA structure."""
    from yea_hud.track_adapter import summarize_track_data

    summary = summarize_track_data(track_data)
    text = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    logger.info("YEA Inspect SAM3 Track Data:\n%s", text)
    return text


try:
    from comfy_api.latest import io, ui

    SAM3TrackData = io.Custom(TRACK_DATA_TYPE)

    class YEAInspectSAM3TrackData(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Inspect the real runtime structure of native SAM3_TRACK_DATA "
                    "(packed_masks shape/dtype, n_frames, per-object scores, orig_size, "
                    "per-frame object visibility) and pass it through unchanged."
                ),
                search_aliases=["sam3", "track", "video", "debug", "inspect"],
                inputs=[
                    SAM3TrackData.Input("track_data", tooltip="Native SAM3_TRACK_DATA from SAM3 Video Track."),
                ],
                outputs=[
                    io.String.Output("debug_json", display_name="debug_json"),
                    SAM3TrackData.Output("track_data", display_name="track_data"),
                ],
                is_output_node=True,
            )

        @classmethod
        def execute(cls, track_data=None) -> io.NodeOutput:
            text = run_inspect_track_data(track_data)
            return io.NodeOutput(text, track_data, ui=ui.PreviewText(text))

    NODE = YEAInspectSAM3TrackData
except ImportError:  # pragma: no cover
    NODE = None
