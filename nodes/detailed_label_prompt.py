"""HUDKit Detailed Label Prompt — generate a strict VLM prompt (NO network call).

Produces a system prompt, a user prompt and a JSON schema string that any connected
vision-language / partner node can consume to describe a track crop. This node never
makes a network request, downloads a model, or hardcodes a specific VLM — it only
emits text. The strict prompt enforces the project's safety rules (visually
observable only; no identity / names / ethnicity / religion / nationality / medical).
"""

from __future__ import annotations

import json

DISPLAY_NAME = "HUDKit Detailed Label Prompt"
NODE_ID = "YEADetailedLabelPrompt"
CATEGORY = "HUDKit/Annotation"

DETAIL_LEVELS = ("short", "medium", "detailed")

_SAFETY = [
    "Return VALID JSON ONLY — no prose, no markdown, no code fences.",
    "Describe ONLY visually observable information in the image.",
    "Do NOT identify the real person. Do NOT guess names.",
    "Do NOT infer ethnicity, race, religion, nationality, medical condition, age, or any "
    "other sensitive/protected attribute.",
    "Prefer clothing, accessories, pose, action, interacting objects and visible scene context.",
    "Use concise UPPERCASE English suitable for a technical HUD. Avoid poetic language.",
    "Avoid uncertain claims. If uncertain, use a simpler label.",
]


def run_detailed_label_prompt(detail_level="detailed", include_clothing=True, include_action=True,
                              include_objects=True, include_scene=True, include_visible_details=True,
                              maximum_words=12, uppercase=True):
    """Return (system_prompt, user_prompt, json_schema_str)."""
    detail_level = detail_level if detail_level in DETAIL_LEVELS else "detailed"
    asks = []
    if include_clothing:
        asks.append("visible clothing and accessories")
    if include_visible_details:
        asks.append("visible physical details (e.g. tattoos, shirtless, hair length)")
    if include_action:
        asks.append("the action or pose")
    if include_objects:
        asks.append("objects the subject is interacting with")
    if include_scene:
        asks.append("brief visible scene context")
    focus = ", ".join(asks) if asks else "the most salient visible attribute"

    system_prompt = (
        "You are a HUD annotation engine for a video-tracking overlay. For each image crop "
        "of a single tracked subject, output a strict JSON object describing what is VISIBLE.\n"
        + "\n".join(f"- {r}" for r in _SAFETY)
        + f"\n- Limit detailed_label to at most {int(maximum_words)} words.\n"
        + ("- Output all label text in UPPERCASE.\n" if uppercase else "")
        + f"- Target detail level: {detail_level}."
    )

    user_prompt = (
        "Describe the tracked subject in this crop as JSON matching the provided schema. "
        f"Focus on: {focus}. "
        "short_label = the object category in 1-2 words (e.g. PERSON, HUMAN FACE, FOOTWEAR). "
        "detailed_label = a compact descriptive phrase of the visible attributes. "
        "action_label = the visible action if any. "
        "Fill attributes arrays only with terms you can SEE. Use empty arrays/strings when unsure."
    )

    schema = {
        "type": "object",
        "properties": {
            "track_id": {"type": "string"},
            "category": {"type": "string"},
            "short_label": {"type": "string"},
            "detailed_label": {"type": "string", "maxWords": int(maximum_words)},
            "action_label": {"type": "string"},
            "context_label": {"type": "string"},
            "attributes": {
                "type": "object",
                "properties": {
                    "clothing": {"type": "array", "items": {"type": "string"}},
                    "visible_details": {"type": "array", "items": {"type": "string"}},
                    "action": {"type": "array", "items": {"type": "string"}},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "scene": {"type": "array", "items": {"type": "string"}},
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["track_id", "category", "short_label", "detailed_label"],
    }
    return system_prompt, user_prompt, json.dumps(schema, indent=2)


try:
    from comfy_api.latest import io

    class YEADetailedLabelPrompt(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Generate a strict, safety-constrained JSON prompt for any connected VLM / "
                    "partner node. Makes NO network call and embeds no specific model — it only "
                    "emits system/user prompts + a JSON schema for describing track crops."
                ),
                search_aliases=["vlm", "prompt", "caption", "label", "annotation"],
                inputs=[
                    io.Combo.Input("detail_level", options=list(DETAIL_LEVELS), default="detailed"),
                    io.Boolean.Input("include_clothing", default=True),
                    io.Boolean.Input("include_action", default=True),
                    io.Boolean.Input("include_objects", default=True),
                    io.Boolean.Input("include_scene", default=True),
                    io.Boolean.Input("include_visible_details", default=True),
                    io.Int.Input("maximum_words", default=12, min=2, max=40),
                    io.Boolean.Input("uppercase", default=True),
                ],
                outputs=[
                    io.String.Output("system_prompt", display_name="system_prompt"),
                    io.String.Output("user_prompt", display_name="user_prompt"),
                    io.String.Output("json_schema", display_name="json_schema"),
                ],
            )

        @classmethod
        def execute(cls, detail_level="detailed", include_clothing=True, include_action=True,
                    include_objects=True, include_scene=True, include_visible_details=True,
                    maximum_words=12, uppercase=True) -> io.NodeOutput:
            sp, up, schema = run_detailed_label_prompt(
                detail_level, include_clothing, include_action, include_objects, include_scene,
                include_visible_details, maximum_words, uppercase)
            return io.NodeOutput(sp, up, schema)

    NODE = YEADetailedLabelPrompt
except ImportError:  # pragma: no cover
    NODE = None
