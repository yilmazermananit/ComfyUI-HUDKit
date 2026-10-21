"""HUDKit Annotation JSON — paste / edit / validate annotation JSON manually.

Turns a hand-written or pasted YEA_ANNOTATIONS JSON string into a validated
``AnnotationSet`` object (+ normalized JSON + a validation report). No network,
no model — pure parse/validate/normalize.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("YEAHUD.annotation_json")

DISPLAY_NAME = "HUDKit Annotation JSON"
NODE_ID = "YEAAnnotationJSON"
CATEGORY = "HUDKit/Annotation"
ANNOTATIONS_TYPE = "YEA_ANNOTATIONS"

_EXAMPLE = (
    '{\n'
    '  "version": 1,\n'
    '  "tracks": [\n'
    '    {"track_id": "person:0", "category": "person", "short_label": "PERSON",\n'
    '     "detailed_label": "SHIRTLESS MAN WITH TATTOOS WEARING BEIGE SWEATPANTS",\n'
    '     "action_label": "KICKING SOCCER BALL",\n'
    '     "display_label": "SHIRTLESS MAN WITH TATTOOS KICKING SOCCER BALL"}\n'
    '  ]\n'
    '}'
)


def run_annotation_json(annotations_json: str, width: int = 0, height: int = 0,
                        frame_count: int = 0, strict: bool = False):
    """Return (AnnotationSet, normalized_json_str, validation_report_str)."""
    from yea_hud.annotations import parse_annotations_json

    aset, report = parse_annotations_json(annotations_json, W=width, H=height,
                                          frame_count=frame_count, strict=strict)
    n_children = sum(len(t.children) for t in aset.tracks)
    n_frame_annos = sum(len(v) for v in aset.frame_annotations.values())
    lines = [
        f"tracks: {len(aset.tracks)}",
        f"child annotations: {n_children}",
        f"frame annotations: {n_frame_annos}",
        f"issues: {len(report)}",
    ]
    lines += [f"  - {r}" for r in report[:30]]
    if not aset.tracks and not aset.frame_annotations:
        lines.append("NOTE: empty annotation set. Example:")
        lines.append(_EXAMPLE)
    return aset, aset.to_json(indent=2), "\n".join(lines)


try:
    from comfy_api.latest import io

    YEAAnnotations = io.Custom(ANNOTATIONS_TYPE)

    class YEAAnnotationJSON(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Paste / edit / validate YEA_ANNOTATIONS JSON. Outputs a validated "
                    "annotation object, normalized JSON, and a validation report. The renderer "
                    "uses these stable labels; it never invents descriptions itself."
                ),
                search_aliases=["annotation", "label", "json", "caption"],
                inputs=[
                    io.String.Input("annotations_json", multiline=True, default=_EXAMPLE,
                                    tooltip="YEA_ANNOTATIONS JSON (tracks/children/frame_annotations)."),
                    io.Boolean.Input("strict", default=False, optional=True,
                                     tooltip="If true, invalid JSON raises instead of falling back to empty."),
                ],
                outputs=[
                    YEAAnnotations.Output("annotations", display_name="annotations"),
                    io.String.Output("normalized_json", display_name="normalized_json"),
                    io.String.Output("validation_report", display_name="validation_report"),
                ],
            )

        @classmethod
        def execute(cls, annotations_json, strict=False) -> io.NodeOutput:
            aset, norm, report = run_annotation_json(annotations_json, strict=strict)
            return io.NodeOutput(aset, norm, report)

    NODE = YEAAnnotationJSON
except ImportError:  # pragma: no cover
    NODE = None
