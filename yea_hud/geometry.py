"""geometry.py — box math, padding, smoothing, placement.

Schema-agnostic core geometry for ComfyUI-HUDKit. Pure, deterministic compute
only: no ComfyUI imports, no network, no subprocess, no file I/O.

Coordinate standard EVERYWHERE in this module: absolute-pixel **xyxy**
``(x1, y1, x2, y2)`` with top-left origin, unless a function name says otherwise
(``xywh_*``). A "Box" is therefore ``(x1, y1, x2, y2)`` as floats.

All functions accept ints or floats and return floats (so downstream EMA /
sub-pixel placement stays smooth); callers round only at the rasterization
boundary.

See ``docs/INTERFACES.md`` ("## geometry.py") for the binding signatures and
``docs/runtime_findings.md`` §3 for why xyxy/absolute-pixel is the standard.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# ``Box`` is re-exported from types for the canonical alias; the local literal
# keeps the type identical (tuple[float, float, float, float]).
from .types import Box

__all__ = [
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "normalize_box",
    "clamp_box",
    "is_degenerate",
    "pad_box",
    "iou",
    "centroid",
    "centroid_dist",
    "area",
    "smoothing_to_alpha",
    "ema_box",
    "adaptive_alpha",
    "importance",
    "label_rect",
    "target_center",
    "asset_rect",
]


# ---------------------------------------------------------------------------
# Small internal helpers (not part of the public contract)
# ---------------------------------------------------------------------------

def _f(v) -> float:
    """Coerce a scalar to a finite float.

    Accepts python ints/floats and 0-d numpy / torch scalars (anything that
    implements ``float()``). NaN/Inf are mapped to ``0.0`` so a single bad
    coordinate can never poison downstream math. ``None`` becomes ``0.0``.
    """
    if v is None:
        return 0.0
    try:
        out = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return out


def _clampf(v: float, lo: float, hi: float) -> float:
    """Clamp ``v`` into ``[lo, hi]`` (lo/hi may be supplied in any order)."""
    if hi < lo:
        lo, hi = hi, lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def xywh_to_xyxy(x, y, w, h) -> Box:
    """Convert a top-left ``xywh`` box to absolute-pixel ``xyxy``.

    ``x, y`` is the top-left corner; ``w, h`` are width/height. Negative
    width/height are tolerated (the resulting box may have ``x2 < x1``); call
    :func:`normalize_box` afterwards if you need a canonical box. Returns floats.
    """
    x = _f(x)
    y = _f(y)
    w = _f(w)
    h = _f(h)
    return (x, y, x + w, y + h)


def xyxy_to_xywh(x1, y1, x2, y2) -> Tuple[float, float, float, float]:
    """Convert an ``xyxy`` box to top-left ``xywh``.

    Returns ``(x, y, w, h)`` where ``x, y`` is the top-left corner and ``w, h``
    is the (signed) extent ``x2 - x1`` / ``y2 - y1``. Normalize first if you
    need non-negative width/height.
    """
    x1 = _f(x1)
    y1 = _f(y1)
    x2 = _f(x2)
    y2 = _f(y2)
    return (x1, y1, x2 - x1, y2 - y1)


def normalize_box(x1, y1, x2, y2) -> Box:
    """Return a canonical ``xyxy`` box with ``x1 <= x2`` and ``y1 <= y2``.

    Swaps the X pair and/or Y pair when they arrive reversed (``x2 < x1`` or
    ``y2 < y1``). This is the single guarantee every downstream consumer relies
    on (``Detection`` promises ``x2 >= x1``, ``y2 >= y1``).
    """
    x1 = _f(x1)
    y1 = _f(y1)
    x2 = _f(x2)
    y2 = _f(y2)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def clamp_box(box: Box, W: int, H: int) -> Box:
    """Clip an ``xyxy`` box to the image bounds ``[0, W] x [0, H]``.

    The box is normalized first (so a reversed input still clips correctly),
    then each coordinate is clamped: X into ``[0, W]``, Y into ``[0, H]``.
    ``W``/``H`` are treated as the image width/height in pixels; a box may
    therefore touch the right/bottom edge exactly at ``W``/``H``.
    """
    x1, y1, x2, y2 = normalize_box(*box)
    W = _f(W)
    H = _f(H)
    # Defensive: zero / negative dimensions collapse the box to the origin.
    if W <= 0.0 or H <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    x1 = _clampf(x1, 0.0, W)
    x2 = _clampf(x2, 0.0, W)
    y1 = _clampf(y1, 0.0, H)
    y2 = _clampf(y2, 0.0, H)
    return (x1, y1, x2, y2)


def is_degenerate(box: Box, min_size: float = 1.0) -> bool:
    """True if the box is too small to draw: width < ``min_size`` or height < ``min_size``.

    Uses the *normalized* extent (absolute width/height) so a reversed box is
    judged by its true size. A non-finite or ``None`` box is degenerate.
    """
    if box is None:
        return True
    x1, y1, x2, y2 = normalize_box(*box)
    w = x2 - x1
    h = y2 - y1
    min_size = _f(min_size)
    return (w < min_size) or (h < min_size)


def pad_box(box: Box, px: float, py: float, W: int, H: int, mode: str = "pixels") -> Box:
    """Pad an ``xyxy`` box outward, then clamp to the image bounds.

    Parameters
    ----------
    box : Box
        Input box (normalized internally).
    px, py : float
        Horizontal / vertical padding. In ``mode='pixels'`` these are added to
        EACH side (so the box grows by ``2*px`` in width, ``2*py`` in height).
        In ``mode='percent'`` they are fractions of the box's own width/height
        (e.g. ``px=0.1`` adds 10 % of the current box width to each side).
    W, H : int
        Image dimensions used for the final clamp.
    mode : str
        ``'pixels'`` (default) or ``'percent'``. Any unknown value is treated
        as ``'pixels'`` (defensive).

    Returns
    -------
    Box
        Padded + clamped box. Negative padding (inset) is allowed; if it would
        invert the box the result is clamped to a valid (possibly degenerate)
        box rather than a reversed one.
    """
    x1, y1, x2, y2 = normalize_box(*box)
    px = _f(px)
    py = _f(py)
    if mode == "percent":
        w = x2 - x1
        h = y2 - y1
        dx = px * w
        dy = py * h
    else:  # 'pixels' and any unknown mode
        dx = px
        dy = py
    x1 -= dx
    x2 += dx
    y1 -= dy
    y2 += dy
    # Negative padding can over-shrink and invert; re-normalize before clamp so
    # we never emit a reversed box.
    return clamp_box(normalize_box(x1, y1, x2, y2), W, H)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def area(box: Box) -> float:
    """Absolute-pixel area of an ``xyxy`` box (``>= 0``).

    Uses normalized extents, so a reversed box still reports a non-negative
    area. A degenerate (zero-extent) box has area ``0.0``.
    """
    x1, y1, x2, y2 = normalize_box(*box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two ``xyxy`` boxes, in ``[0, 1]``.

    Both boxes are normalized first. Returns ``0.0`` when there is no overlap
    or when the union has zero area (two degenerate / coincident points).
    """
    ax1, ay1, ax2, ay2 = normalize_box(*a)
    bx1, by1, bx2, by2 = normalize_box(*b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return _clampf(inter / union, 0.0, 1.0)


def centroid(box: Box) -> Tuple[float, float]:
    """Center point ``(cx, cy)`` of an ``xyxy`` box.

    Computed from the raw (un-normalized) corners — the midpoint is identical
    whether or not the pair is swapped, so no normalization is required.
    """
    x1, y1, x2, y2 = (_f(v) for v in box)
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def centroid_dist(a: Box, b: Box) -> float:
    """Euclidean distance between the centroids of two ``xyxy`` boxes."""
    ax, ay = centroid(a)
    bx, by = centroid(b)
    return math.hypot(ax - bx, ay - by)


# ---------------------------------------------------------------------------
# Temporal smoothing (EMA)
# ---------------------------------------------------------------------------

def smoothing_to_alpha(smoothing: float) -> float:
    """Map a UI ``smoothing`` slider in ``[0, 1]`` to an EMA ``alpha``.

    ``alpha`` is the weight of the *current* observation in
    ``alpha*current + (1-alpha)*previous``:

    * ``smoothing = 0.0`` -> ``alpha = 1.0`` (raw, no smoothing / no lag)
    * ``smoothing = 0.9`` -> ``alpha ~= 0.1`` (very smooth / laggy)

    The mapping is the monotonic ``alpha = clamp(1.0 - smoothing, 0.02, 1.0)``.
    The ``0.02`` floor guarantees tracks still eventually converge even at the
    maximum smoothing setting.
    """
    return _clampf(1.0 - _f(smoothing), 0.02, 1.0)


def ema_box(current: Box, previous: Optional[Box], alpha: float) -> Box:
    """Exponential-moving-average blend of two ``xyxy`` boxes, per coordinate.

    With no history (``previous is None``) the current box is returned
    unchanged (the EMA is seeded on first sight). Otherwise each coordinate is
    ``alpha*current + (1-alpha)*previous``. ``alpha`` is clamped to ``[0, 1]``
    defensively. The result is NOT re-normalized (EMA of two valid boxes stays
    valid; callers normalize/clamp downstream as needed).
    """
    cx1, cy1, cx2, cy2 = (_f(v) for v in current)
    if previous is None:
        return (cx1, cy1, cx2, cy2)
    a = _clampf(_f(alpha), 0.0, 1.0)
    px1, py1, px2, py2 = (_f(v) for v in previous)
    inv = 1.0 - a
    return (
        a * cx1 + inv * px1,
        a * cy1 + inv * py1,
        a * cx2 + inv * px2,
        a * cy2 + inv * py2,
    )


def adaptive_alpha(base_alpha: float, current: Box, previous: Optional[Box], frame_diag: float) -> float:
    """Motion-adaptive EMA ``alpha``: faster motion -> less smoothing.

    The idea (brief §10/§11): a slow / static subject should be smoothed hard
    (use ``base_alpha``), while a fast-moving subject should snap toward its new
    position (alpha -> 1) so the overlay does not lag behind. Motion is the
    centroid displacement normalized by the frame diagonal::

        motion = centroid_dist(current, previous) / frame_diag        # >= 0
        alpha  = base_alpha + (1 - base_alpha) * min(1.0, motion)

    so the result interpolates from ``base_alpha`` (no motion) up to ``1.0``
    (motion >= one frame-diagonal). With no history, or a non-positive
    ``frame_diag``, the un-adapted ``base_alpha`` (clamped to ``[0, 1]``) is
    returned. The result is monotonically non-decreasing in motion.
    """
    base = _clampf(_f(base_alpha), 0.0, 1.0)
    if previous is None:
        return base
    diag = _f(frame_diag)
    if diag <= 0.0:
        return base
    motion = centroid_dist(current, previous) / diag
    if motion < 0.0:
        motion = 0.0
    elif motion > 1.0:
        motion = 1.0
    return base + (1.0 - base) * motion


# ---------------------------------------------------------------------------
# Importance / saliency
# ---------------------------------------------------------------------------

def importance(box: Box, W: int, H: int, center_bias: float = 0.0) -> float:
    """Saliency score of a box: relative area, optionally biased toward center.

    Base score is ``box_area / frame_area`` (in ``[0, 1]``). When
    ``center_bias > 0`` the score is multiplied by a center weight::

        weight = 1 + center_bias * (1 - dist_to_center_norm)

    where ``dist_to_center_norm`` is the box-centroid's distance from the frame
    center divided by the half-diagonal (so it is ``0`` dead-center and ``~1``
    at the corners). A box centered in frame gets the full
    ``(1 + center_bias)`` boost; a corner box gets ~no boost. ``center_bias``
    is clamped to ``>= 0``. Returns ``0.0`` for a zero-sized frame.
    """
    W = _f(W)
    H = _f(H)
    frame_area = W * H
    if frame_area <= 0.0:
        return 0.0
    base = area(box) / frame_area
    bias = _f(center_bias)
    if bias <= 0.0:
        return base
    cx, cy = centroid(box)
    fcx, fcy = W * 0.5, H * 0.5
    half_diag = 0.5 * math.hypot(W, H)
    if half_diag <= 0.0:
        return base
    dist_norm = _clampf(math.hypot(cx - fcx, cy - fcy) / half_diag, 0.0, 1.0)
    weight = 1.0 + bias * (1.0 - dist_norm)
    return base * weight


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def target_center(box: Box) -> Tuple[float, float]:
    """Crosshair / target anchor point for a box — always its centroid."""
    return centroid(box)


def label_rect(box: Box, text_w: int, text_h: int, W: int, H: int, side: str) -> Tuple[Box, str]:
    """Compute a label-strip rect that stays fully inside the image.

    The strip has the requested text size ``(text_w, text_h)`` and is placed
    immediately *outside* the box on the preferred ``side`` (``'top'`` puts the
    strip above the box, ``'bottom'`` below it). It is left-aligned to the box's
    left edge by default.

    Overflow handling (brief §13.5 — labels must never clip off-frame):

    * **Vertical flip:** if the strip would run off the top of the frame it
      flips to the bottom, and vice-versa. If it does not fit on either side it
      is clamped just inside the nearer edge (preferred side wins ties).
    * **Horizontal align:** if the left-aligned strip overflows the right edge
      it is right-aligned to the box's right edge; if it still overflows (strip
      wider than the frame or the box hugging the left edge) it is clamped to
      ``[0, W - text_w]`` so the whole strip stays on-frame.

    Parameters
    ----------
    box : Box
        Anchor box in ``xyxy``.
    text_w, text_h : int
        Strip size in pixels (already measured + truncated upstream).
    W, H : int
        Image bounds.
    side : str
        Preferred side, ``'top'`` or ``'bottom'`` (anything else -> ``'top'``).

    Returns
    -------
    (rect_xyxy, resolved_side)
        ``rect_xyxy`` is the final on-frame strip box; ``resolved_side`` is the
        side actually used (``'top'`` or ``'bottom'``) after any flip.
    """
    x1, y1, x2, y2 = normalize_box(*box)
    tw = max(0.0, _f(text_w))
    th = max(0.0, _f(text_h))
    Wf = _f(W)
    Hf = _f(H)

    pref = "top" if side != "bottom" else "bottom"

    # --- vertical placement (with flip) ---------------------------------
    # Candidate top-edge Y for each side (strip sits just outside the box).
    top_y = y1 - th          # strip directly above the box
    bottom_y = y2            # strip directly below the box

    def _fits(ry: float) -> bool:
        return ry >= 0.0 and (ry + th) <= Hf

    if pref == "top":
        if _fits(top_y):
            ry, resolved = top_y, "top"
        elif _fits(bottom_y):
            ry, resolved = bottom_y, "bottom"
        else:
            # Neither side fits the strip fully (very tall strip / tiny frame
            # / box spanning the whole height): keep preferred side, clamp in.
            ry, resolved = top_y, "top"
    else:  # pref == 'bottom'
        if _fits(bottom_y):
            ry, resolved = bottom_y, "bottom"
        elif _fits(top_y):
            ry, resolved = top_y, "top"
        else:
            ry, resolved = bottom_y, "bottom"

    # Final vertical clamp so the strip is always on-frame (guards the
    # "fits nowhere" branch and any float drift).
    if Hf > 0.0:
        ry = _clampf(ry, 0.0, max(0.0, Hf - th))
    else:
        ry = 0.0

    # --- horizontal placement (left-align, flip to right, clamp) --------
    rx = x1  # default: align strip's left edge to the box's left edge
    if rx + tw > Wf:
        # would overflow right -> right-align to the box's right edge
        rx = x2 - tw
    if Wf > 0.0:
        rx = _clampf(rx, 0.0, max(0.0, Wf - tw))
    else:
        rx = 0.0

    rect = (rx, ry, rx + tw, ry + th)
    return rect, resolved


def asset_rect(box: Box, fit: str, inner_padding: float, aspect: Optional[Tuple[int, int]]) -> Box:
    """Placement rect for an asset inside a box, given a fit mode + padding.

    The box is first shrunk by ``inner_padding`` (a fraction of the box's width
    and height applied to each side; e.g. ``0.1`` insets 10 % per side). The
    asset is then placed in the resulting inner rect according to ``fit``:

    * ``'stretch'`` — fill the inner rect exactly (aspect ignored).
    * ``'contain'`` — scale to fit *inside* the inner rect preserving
      ``aspect``; the asset is centered (letter/pillar-boxed).
    * ``'cover'``  — scale to *cover* the inner rect preserving ``aspect``; the
      asset is centered and the overflow is symmetric (caller clips to the box).

    ``aspect`` is ``(aw, ah)`` describing the asset's native aspect ratio. When
    ``aspect`` is ``None`` or non-positive, or ``fit`` is unknown, the behavior
    falls back to ``'stretch'`` (the inner rect itself). ``inner_padding`` is
    clamped to ``[0, 0.49]`` so the inner rect never collapses or inverts.

    Returns the asset rect in absolute-pixel ``xyxy`` (NOT clamped to the image
    — the compositor clips ``'cover'`` overflow when drawing).
    """
    x1, y1, x2, y2 = normalize_box(*box)
    bw = x2 - x1
    bh = y2 - y1

    pad = _clampf(_f(inner_padding), 0.0, 0.49)
    ix1 = x1 + bw * pad
    iy1 = y1 + bh * pad
    ix2 = x2 - bw * pad
    iy2 = y2 - bh * pad
    iw = ix2 - ix1
    ih = iy2 - iy1

    inner = (ix1, iy1, ix2, iy2)

    # Degenerate inner rect or stretch / missing aspect -> just the inner rect.
    if iw <= 0.0 or ih <= 0.0:
        return inner
    if fit not in ("contain", "cover"):
        return inner
    if aspect is None:
        return inner
    try:
        aw = _f(aspect[0])
        ah = _f(aspect[1])
    except (TypeError, IndexError):
        return inner
    if aw <= 0.0 or ah <= 0.0:
        return inner

    asset_ratio = aw / ah           # width / height
    inner_ratio = iw / ih

    if fit == "contain":
        # Fit inside: limited by the tighter dimension.
        if asset_ratio > inner_ratio:
            # asset relatively wider -> width-bound, letterbox top/bottom
            out_w = iw
            out_h = iw / asset_ratio
        else:
            # asset relatively taller -> height-bound, pillarbox left/right
            out_h = ih
            out_w = ih * asset_ratio
    else:  # 'cover'
        # Cover: limited by the looser dimension; overflow on the other axis.
        if asset_ratio > inner_ratio:
            # asset relatively wider -> height-bound, overflow horizontally
            out_h = ih
            out_w = ih * asset_ratio
        else:
            # asset relatively taller -> width-bound, overflow vertically
            out_w = iw
            out_h = iw / asset_ratio

    # Center the scaled asset within the inner rect.
    icx = (ix1 + ix2) * 0.5
    icy = (iy1 + iy2) * 0.5
    rx1 = icx - out_w * 0.5
    ry1 = icy - out_h * 0.5
    return (rx1, ry1, rx1 + out_w, ry1 + out_h)
