"""Schema-agnostic core types for ComfyUI-YEAHUD.

This module is the *contract* every other module in ``yea_hud`` implements
against. It depends only on the Python standard library (no torch, no PIL, no
ComfyUI) so it can be imported in any context and unit-tested in isolation.

Internal coordinate standard everywhere downstream of the adapters is ALWAYS:
    absolute pixel coordinates, x1, y1, x2, y2   (top-left .. bottom-right)

See docs/runtime_findings.md §3 for why (SAM3 emits absolute-pixel xywh dicts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Palette (brief §14)
# ---------------------------------------------------------------------------

#: Default HUD palette: red, pink, blue, lime.
DEFAULT_PALETTE: tuple[str, ...] = ("#FF2A1A", "#FFB0DF", "#225CFF", "#8DFF28")

#: Optional black/white micro-detail colors.
BW_DETAILS: tuple[str, ...] = ("#FFFFFF", "#0A0A0A")


# ---------------------------------------------------------------------------
# Enumerated string options (kept as plain str tuples so they map 1:1 onto
# ComfyUI COMBO widget option lists without an Enum import dance).
# ---------------------------------------------------------------------------

STYLE_MODES: tuple[str, ...] = (
    "mixed",          # default — full target visual language
    "assets_only",    # only user PNG frame/target assets
    "procedural_only",
    "corners",
    "outline",
    "solid",
    "target_only",
    "labels_only",
)

SMALL_OBJECT_MODES: tuple[str, ...] = (
    "hide",
    "target_only",
    "label_only",
    "tiny_corners",
)

CAPTION_SOURCES: tuple[str, ...] = (
    "fixed",
    "bbox_label",
    "json_by_frame",
    "json_by_track",
    "template",
    "none",
)

ASSET_FIT_MODES: tuple[str, ...] = ("stretch", "contain", "cover")

LABEL_TEXT_COLOR_MODES: tuple[str, ...] = ("auto", "white", "black")

PADDING_MODES: tuple[str, ...] = ("pixels", "percent")

#: The procedural "primitive" styles a track can be assigned in mixed mode.
PRIMITIVE_STYLES: tuple[str, ...] = (
    "corners",
    "outline",
    "solid",
    "target_only",
    "frame_asset",   # use a user frame PNG
)


# ---------------------------------------------------------------------------
# Detection — normalized output of bbox_adapter / mask_adapter (brief §4.3)
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single detection in a single frame, in absolute-pixel xyxy.

    Produced by :mod:`yea_hud.bbox_adapter` and :mod:`yea_hud.mask_adapter`.
    All consumers (tracker, compositor) read xyxy. ``x2 >= x1`` and ``y2 >= y1``
    are guaranteed by the adapters (they swap/clamp/drop degenerate boxes).
    """

    frame_index: int
    source_index: int          # index within the source frame's detection list
    x1: float
    y1: float
    x2: float
    y2: float
    score: Optional[float] = None
    label: Optional[str] = None
    source_track_id: Optional[Union[int, str]] = None
    mask_index: Optional[int] = None   # index into a flattened/individual mask batch

    # -- geometry conveniences (pure, no deps) ------------------------------
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)


# ---------------------------------------------------------------------------
# Track — internal tracking state (brief §10.1)
# ---------------------------------------------------------------------------

Box = tuple[float, float, float, float]  # xyxy


@dataclass
class TrackStyle:
    """Deterministic visual identity of a track, derived once from
    ``hash(seed, track_id)`` (brief §10.5 / §11.1). Never re-rolled per frame.
    """

    color_id: int = 0                 # index into the resolved palette
    primitive_style: str = "corners"  # one of PRIMITIVE_STYLES
    frame_asset_id: int = -1          # index into eligible frame assets, -1 = none
    target_asset_id: int = -1         # index into eligible target assets, -1 = none
    show: bool = True                 # display_probability gate
    use_fill: bool = False            # translucent solid block
    use_target: bool = False          # crosshair / target point
    use_caption: bool = True
    caption_side: str = "top"         # top|bottom — preferred label placement
    glitch_phase: float = 0.0         # 0..1 per-track glitch personality offset


@dataclass
class Track:
    track_id: int
    last_box: Box
    smoothed_box: Box
    last_seen_frame: int
    age: int = 0
    missing_frames: int = 0
    label: str = ""
    style: TrackStyle = field(default_factory=TrackStyle)

    # convenience mirrors of style fields the brief lists directly on Track;
    # kept as read-through properties so there is a single source of truth.
    @property
    def style_id(self) -> str:
        return self.style.primitive_style

    @property
    def color_id(self) -> int:
        return self.style.color_id

    @property
    def frame_asset_id(self) -> int:
        return self.style.frame_asset_id

    @property
    def target_asset_id(self) -> int:
        return self.style.target_asset_id

    @property
    def show(self) -> bool:
        return self.style.show


# ---------------------------------------------------------------------------
# RenderItem — what the compositor draws for one detection on one frame.
# Produced by the renderer node after tracking + style + placement resolution.
# ---------------------------------------------------------------------------

@dataclass
class RenderItem:
    track_id: int
    order_index: int               # 1-based display order within the frame
    box: Box                       # placement box (smoothed + padded + clamped), xyxy
    color_rgb: tuple[float, float, float]   # 0..1
    style: TrackStyle
    label_text: str                # already formatted + uppercased + truncated upstream? (compositor truncates to fit)
    score: Optional[float] = None
    # per-frame micro variation (brief §11.2), resolved from hash(seed,tid,frame)
    opacity_mul: float = 1.0
    line_dropout: bool = False
    target_flash: bool = False
    fill_flash: bool = False
    jitter: tuple[int, int] = (0, 0)
    # --- semantic annotation layer (brief: annotation/label hierarchy) ---
    priority: int = 50                 # visual/placement priority (higher first)
    is_primary: bool = False           # primary subject (vs secondary/child region)
    is_child: bool = False             # a child annotation (face/footwear/object)
    label_style: str = "solid_bar"     # solid_bar|transparent_bar|minimal_text|technical_stack|micro_tag
    label_rect: Optional[Box] = None   # pre-resolved label placement (xyxy); compositor uses it if set
    label_visible: bool = True         # collision engine may hide the text (keep the frame)
    label_color_rgb: Optional[tuple] = None  # override strip bg (e.g. category color for children)


# ---------------------------------------------------------------------------
# Default node parameters (brief §23). Single source of truth for both the
# V3 schema defaults and tests.
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, object] = {
    "seed": 77,
    "style_mode": "mixed",
    "max_objects": 24,
    "display_probability": 0.85,
    "padding_x": 8,
    "padding_y": 8,
    "smoothing": 0.55,
    "track_iou_threshold": 0.25,
    "track_memory_frames": 6,
    "frame_opacity": 1.00,
    "fill_opacity": 0.30,
    "label_opacity": 0.92,
    "line_width": 3,
    "target_scale": 0.18,
    "label_scale": 1.00,
    "flicker_probability": 0.04,
    "style_flash_probability": 0.02,
    "position_jitter_px": 0,
    "opacity_jitter": 0.05,
    "min_box_area_ratio": 0.0004,
    "large_box_area_ratio": 0.10,
    "small_object_mode": "target_only",
    "max_centroid_distance_ratio": 0.35,
    "trim_transparent_assets": True,
    "asset_fit_mode": "stretch",
    "asset_aspect_lock": False,
    "asset_inner_padding": 0.0,
    "label_text_color_mode": "auto",
    "caption_source": "fixed",
    "fixed_label": "PERSON",
    "max_cache_items": 512,
}


__all__ = [
    "DEFAULT_PALETTE", "BW_DETAILS",
    "STYLE_MODES", "SMALL_OBJECT_MODES", "CAPTION_SOURCES", "ASSET_FIT_MODES",
    "LABEL_TEXT_COLOR_MODES", "PADDING_MODES", "PRIMITIVE_STYLES",
    "Detection", "Track", "TrackStyle", "RenderItem", "Box", "DEFAULTS",
]
