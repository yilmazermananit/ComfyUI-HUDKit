"""procedural_hud.py — vector HUD primitives (brief §9).

Schema-agnostic PIL vector primitives that draw a "target / tracking-reticle"
visual language onto a **straight-alpha** ``RGBA`` canvas. Every primitive draws
*in place* onto an :class:`PIL.ImageDraw.ImageDraw` and is **internally clipped to
the canvas bounds** so a box that runs off-frame never raises and never paints
outside the layer.

Coordinate standard (matches the rest of ``yea_hud``): absolute-pixel **xyxy**
``(x1, y1, x2, y2)`` with a top-left origin. Colors are passed as
``color_rgba = (r, g, b, a)`` with each channel a float in ``0..1``; the alpha is
*baked into the layer* (PIL ``ImageDraw`` over an RGBA image replaces destination
pixels, so each primitive's own opacity ends up in the layer and the compositor
later alpha-composites the whole overlay over the base exactly once — see
``docs/INTERFACES.md`` "## compositor.py").

Allowed deps only: stdlib, ``numpy``, ``PIL`` (Pillow), ``torch`` (unused here but
permitted), ``from .types import ...`` and the sibling ``geometry`` module.
NO ComfyUI imports, NO network / subprocess / eval / file writes. Pure
deterministic compute (the only randomness is the caller-supplied
``random.Random`` used for optional glitch / registration variation, which is
itself seeded deterministically upstream via ``yea_hud.utils.rng_for``).

Public API (binding contract — ``docs/INTERFACES.md`` "## procedural_hud.py")::

    new_rgba_layer(W, H) -> PIL.Image.Image
    draw_corners(draw, box, color_rgba, line_width, corner_length_ratio=0.25, corner_gap=0)
    draw_outline(draw, box, color_rgba, line_width, glitch_segments=0, rng=None)
    draw_solid(draw, box, color_rgba)
    draw_crosshair(draw, center, size, color_rgba, line_width, gap, dot=True, ticks=False)
    draw_registration(draw, box, color_rgba, density, rng=None)
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from . import geometry
from .types import Box

__all__ = [
    "new_rgba_layer",
    "draw_corners",
    "draw_outline",
    "draw_solid",
    "draw_crosshair",
    "draw_registration",
]


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public contract)
# ---------------------------------------------------------------------------

def _f(v) -> float:
    """Coerce a scalar to a finite float; ``None``/NaN/Inf -> ``0.0``.

    Mirrors ``geometry._f`` so a single bad coordinate or colour channel can
    never poison the rasterizer (PIL would raise on NaN/Inf coordinates).
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
    """Clamp ``v`` into ``[lo, hi]`` (bounds may be given in any order)."""
    if hi < lo:
        lo, hi = hi, lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _rgba_255(color_rgba: Sequence[float]) -> Tuple[int, int, int, int]:
    """Convert a float ``(r, g, b, a)`` in ``0..1`` to an int ``0..255`` RGBA tuple.

    Defensive about length and bad values: missing channels default to ``0``
    except alpha which defaults to fully-opaque (``255``); each channel is
    coerced via :func:`_f`, clamped to ``0..1`` and rounded. Accepts a 3-tuple
    (``rgb``) — alpha then defaults to opaque. Anything longer is truncated to
    four channels. This is what gets *baked* into the straight-alpha layer.
    """
    if color_rgba is None:
        return (0, 0, 0, 0)
    try:
        seq = list(color_rgba)
    except TypeError:
        return (0, 0, 0, 0)
    r = _f(seq[0]) if len(seq) > 0 else 0.0
    g = _f(seq[1]) if len(seq) > 1 else 0.0
    b = _f(seq[2]) if len(seq) > 2 else 0.0
    a = _f(seq[3]) if len(seq) > 3 else 1.0
    return (
        int(round(_clampf(r, 0.0, 1.0) * 255.0)),
        int(round(_clampf(g, 0.0, 1.0) * 255.0)),
        int(round(_clampf(b, 0.0, 1.0) * 255.0)),
        int(round(_clampf(a, 0.0, 1.0) * 255.0)),
    )


def _canvas_size(draw: "ImageDraw.ImageDraw") -> Tuple[int, int]:
    """Return ``(W, H)`` of the image backing an ``ImageDraw`` handle.

    PIL exposes the bound image via ``draw._image`` (or ``draw.im`` on some
    builds). We probe both so the primitives can clip to the real canvas
    instead of trusting the caller-supplied box. Falls back to a sentinel large
    size only if neither attribute exists (then PIL's own bounds clamp us).
    """
    img = getattr(draw, "_image", None)
    if img is not None and getattr(img, "size", None) is not None:
        w, h = img.size
        return int(w), int(h)
    im = getattr(draw, "im", None)
    if im is not None and getattr(im, "size", None) is not None:
        w, h = im.size
        return int(w), int(h)
    # Last-resort sentinel: PIL still clips drawing to its true bounds, so this
    # only affects our own pre-clip math (which then becomes a no-op clamp).
    return (1 << 30, 1 << 30)


def _prepared_box(draw: "ImageDraw.ImageDraw", box: Box) -> Optional[Tuple[float, float, float, float]]:
    """Normalize + clamp a box to the canvas; ``None`` if degenerate/off-frame.

    Returns the canonical, on-canvas ``xyxy`` (floats) or ``None`` when there is
    nothing to draw (box ``None``, zero canvas, or the clamped box has no
    positive extent). Centralizes the "clip to bounds internally" guarantee.
    """
    if box is None:
        return None
    W, H = _canvas_size(draw)
    if W <= 0 or H <= 0:
        return None
    x1, y1, x2, y2 = geometry.clamp_box(tuple(box), W, H)
    if (x2 - x1) <= 0.0 or (y2 - y1) <= 0.0:
        return None
    return (x1, y1, x2, y2)


def _iround(v: float) -> int:
    """Round to the nearest int (PIL wants integer pixel coordinates)."""
    return int(round(_f(v)))


def _safe_line_width(line_width) -> int:
    """A drawable integer line width (``>= 1``)."""
    lw = _iround(line_width)
    return lw if lw >= 1 else 1


def _as_rng(rng) -> random.Random:
    """Return a usable ``random.Random``.

    If ``rng`` is ``None`` (or not a Random) a *fresh, fixed-seed* Random is
    returned so output stays deterministic even when a caller forgets to pass
    one. Real determinism per-track/-frame comes from ``utils.rng_for`` upstream.
    """
    if isinstance(rng, random.Random):
        return rng
    return random.Random(0)


def _clip_segment(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    W: float,
    H: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Liang–Barsky clip of a line segment to the rectangle ``[0,W] x [0,H]``.

    Returns the clipped ``(p0, p1)`` or ``None`` if the segment lies entirely
    outside the canvas. Used so glitch / crosshair strokes whose width could
    bleed past the edge are pre-trimmed (belt-and-braces with PIL's own clamp).
    """
    x0, y0 = _f(p0[0]), _f(p0[1])
    x1, y1 = _f(p1[0]), _f(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - 0.0, W - x0, y0 - 0.0, H - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None  # parallel and outside
        else:
            t = qi / pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    nx0 = x0 + u1 * dx
    ny0 = y0 + u1 * dy
    nx1 = x0 + u2 * dx
    ny1 = y0 + u2 * dy
    return ((nx0, ny0), (nx1, ny1))


def _line(
    draw: "ImageDraw.ImageDraw",
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    color: Tuple[int, int, int, int],
    width: int,
    W: float,
    H: float,
) -> None:
    """Draw a single clipped line segment with integer endpoints.

    No-op when the segment is fully off-canvas or collapses to a zero-length
    after clipping (PIL would still draw a dot for a zero-length line — we skip
    it so an off-frame stroke leaves nothing behind).
    """
    clipped = _clip_segment(p0, p1, W, H)
    if clipped is None:
        return
    (cx0, cy0), (cx1, cy1) = clipped
    draw.line(
        [(_iround(cx0), _iround(cy0)), (_iround(cx1), _iround(cy1))],
        fill=color,
        width=width,
    )


# ---------------------------------------------------------------------------
# Layer factory
# ---------------------------------------------------------------------------

def new_rgba_layer(W: int, H: int) -> "Image.Image":
    """Create a fully-transparent straight-alpha ``RGBA`` layer.

    Parameters
    ----------
    W, H : int
        Layer width / height in pixels. Coerced to ``int``; values ``< 1`` are
        raised to ``1`` so the call never fails on a degenerate frame (PIL
        rejects zero-sized images).

    Returns
    -------
    PIL.Image.Image
        Mode ``'RGBA'``, every pixel ``(0, 0, 0, 0)`` — i.e. transparent with
        undefined (black) colour, the canonical straight-alpha empty overlay the
        HUD primitives draw onto.
    """
    w = int(W) if int(W) >= 1 else 1
    h = int(H) if int(H) >= 1 else 1
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def draw_corners(
    draw: "ImageDraw.ImageDraw",
    box: Box,
    color_rgba: Sequence[float],
    line_width: int,
    corner_length_ratio: float = 0.25,
    corner_gap: int = 0,
) -> None:
    """Draw four L-shaped corner brackets around ``box`` (brief §9.2).

    The classic reticle look: each corner of the box gets a short horizontal +
    vertical stroke, leaving the middle of each edge open. Drawn *in place* onto
    ``draw`` and clipped to the canvas.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Handle over a straight-alpha RGBA canvas.
    box : Box
        Target box ``xyxy``. Normalized + clamped to the canvas internally; a
        degenerate / off-frame box is a no-op.
    color_rgba : (r, g, b, a)
        Stroke colour, floats ``0..1``; alpha baked into the layer.
    line_width : int
        Stroke thickness in pixels (``>= 1``).
    corner_length_ratio : float, default 0.25
        Length of each bracket arm as a fraction of the box's *shorter* side,
        clamped to ``(0, 0.45]`` so the two brackets meeting on an edge always
        leave a visible gap in the middle (the defining look of corner brackets
        vs. a full outline). The arm is additionally capped at ~45 % of each
        available edge so it can never reach the opposite bracket.
    corner_gap : int, default 0
        Inset (px) of each bracket from the exact corner toward the box centre,
        so the L sits slightly *inside* the box (a small gap reads as "locked
        on"). Clamped so it can never collapse the arms.

    Returns
    -------
    None
    """
    prepared = _prepared_box(draw, box)
    if prepared is None:
        return
    x1, y1, x2, y2 = prepared
    color = _rgba_255(color_rgba)
    if color[3] <= 0:
        return
    lw = _safe_line_width(line_width)
    W, H = _canvas_size(draw)

    bw = x2 - x1
    bh = y2 - y1
    short = min(bw, bh)

    ratio = _clampf(_f(corner_length_ratio), 1e-4, 0.45)
    # Arm length: fraction of the shorter side, but never more than ~45 % of
    # either edge so the two brackets sharing an edge always leave a gap in the
    # middle (otherwise corners would degenerate into a full outline).
    arm = ratio * short
    arm = min(arm, bw * 0.45, bh * 0.45)
    if arm <= 0.0:
        return

    # Gap moves the bracket inward from the corner; clamp so arms stay positive.
    gap = _clampf(_f(corner_gap), 0.0, max(0.0, min(bw, bh) * 0.5 - 1.0))
    ix1 = x1 + gap
    iy1 = y1 + gap
    ix2 = x2 - gap
    iy2 = y2 - gap
    if (ix2 - ix1) <= 0.0 or (iy2 - iy1) <= 0.0:
        return

    # Re-cap arm against the (possibly inset) edges, keeping the mid-edge gap.
    arm = min(arm, (ix2 - ix1) * 0.45, (iy2 - iy1) * 0.45)
    if arm <= 0.0:
        return

    # Top-left
    _line(draw, (ix1, iy1), (ix1 + arm, iy1), color, lw, W, H)
    _line(draw, (ix1, iy1), (ix1, iy1 + arm), color, lw, W, H)
    # Top-right
    _line(draw, (ix2, iy1), (ix2 - arm, iy1), color, lw, W, H)
    _line(draw, (ix2, iy1), (ix2, iy1 + arm), color, lw, W, H)
    # Bottom-left
    _line(draw, (ix1, iy2), (ix1 + arm, iy2), color, lw, W, H)
    _line(draw, (ix1, iy2), (ix1, iy2 - arm), color, lw, W, H)
    # Bottom-right
    _line(draw, (ix2, iy2), (ix2 - arm, iy2), color, lw, W, H)
    _line(draw, (ix2, iy2), (ix2, iy2 - arm), color, lw, W, H)


def draw_outline(
    draw: "ImageDraw.ImageDraw",
    box: Box,
    color_rgba: Sequence[float],
    line_width: int,
    glitch_segments: int = 0,
    rng: Optional[random.Random] = None,
) -> None:
    """Draw the rectangular outline of ``box``, optionally "glitched" (brief §9.3).

    With ``glitch_segments <= 0`` this is a clean rectangle border. With a
    positive ``glitch_segments`` each of the four edges is broken into that many
    pieces and a deterministic subset of pieces is *dropped* (using ``rng``),
    giving a flickering/incomplete-scanline HUD outline. Drawn *in place* and
    clipped to the canvas.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Handle over a straight-alpha RGBA canvas.
    box : Box
        Target box ``xyxy``. Normalized + clamped internally; degenerate / off
        -frame -> no-op.
    color_rgba : (r, g, b, a)
        Stroke colour, floats ``0..1``.
    line_width : int
        Stroke thickness in pixels (``>= 1``).
    glitch_segments : int, default 0
        Number of pieces per edge. ``<= 1`` draws a solid (un-broken) rectangle.
        Larger values create more, finer broken segments.
    rng : random.Random | None
        Source of the deterministic glitch pattern. If ``None`` a fixed-seed
        Random is used so the same inputs always yield the same outline. The rng
        is only consumed when ``glitch_segments > 1``.

    Returns
    -------
    None
    """
    prepared = _prepared_box(draw, box)
    if prepared is None:
        return
    x1, y1, x2, y2 = prepared
    color = _rgba_255(color_rgba)
    if color[3] <= 0:
        return
    lw = _safe_line_width(line_width)
    W, H = _canvas_size(draw)

    segs = int(_f(glitch_segments))

    # --- clean rectangle outline -----------------------------------------
    if segs <= 1:
        # Draw the four edges as clipped lines (rather than draw.rectangle) so
        # the corners share the same width semantics as the glitched path and
        # the off-canvas clamp is identical.
        _line(draw, (x1, y1), (x2, y1), color, lw, W, H)  # top
        _line(draw, (x2, y1), (x2, y2), color, lw, W, H)  # right
        _line(draw, (x2, y2), (x1, y2), color, lw, W, H)  # bottom
        _line(draw, (x1, y2), (x1, y1), color, lw, W, H)  # left
        return

    # --- glitched (broken) outline ---------------------------------------
    r = _as_rng(rng)

    def _broken_edge(p_start: Tuple[float, float], p_end: Tuple[float, float]) -> None:
        sx, sy = p_start
        ex, ey = p_end
        for k in range(segs):
            t0 = k / segs
            t1 = (k + 1) / segs
            # Deterministic per-piece keep decision. ~30% dropout reads as a
            # convincing glitch while keeping the box legible. Always draw the
            # first piece of each edge so corners stay anchored.
            keep = (k == 0) or (r.random() > 0.30)
            if not keep:
                continue
            a = (sx + (ex - sx) * t0, sy + (ey - sy) * t0)
            b = (sx + (ex - sx) * t1, sy + (ey - sy) * t1)
            _line(draw, a, b, color, lw, W, H)

    _broken_edge((x1, y1), (x2, y1))  # top
    _broken_edge((x2, y1), (x2, y2))  # right
    _broken_edge((x2, y2), (x1, y2))  # bottom
    _broken_edge((x1, y2), (x1, y1))  # left


def draw_solid(draw: "ImageDraw.ImageDraw", box: Box, color_rgba: Sequence[float]) -> None:
    """Fill ``box`` with a solid translucent block (brief §9.4).

    The fill opacity is whatever is already baked into ``color_rgba`` alpha (the
    caller multiplies in ``fill_opacity`` upstream). Drawn *in place* and clipped
    to the canvas via the prepared, on-frame box.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Handle over a straight-alpha RGBA canvas.
    box : Box
        Region to fill, ``xyxy``. Normalized + clamped internally; degenerate /
        off-frame -> no-op.
    color_rgba : (r, g, b, a)
        Fill colour incl. opacity, floats ``0..1``.

    Returns
    -------
    None
    """
    prepared = _prepared_box(draw, box)
    if prepared is None:
        return
    x1, y1, x2, y2 = prepared
    color = _rgba_255(color_rgba)
    if color[3] <= 0:
        return
    # ImageDraw.rectangle fills inclusive of both corners; use rounded int
    # corners. The box is already clamped on-canvas so PIL never clips here.
    draw.rectangle(
        [(_iround(x1), _iround(y1)), (_iround(x2), _iround(y2))],
        fill=color,
    )


def draw_crosshair(
    draw: "ImageDraw.ImageDraw",
    center: Tuple[float, float],
    size: int,
    color_rgba: Sequence[float],
    line_width: int,
    gap: int,
    dot: bool = True,
    ticks: bool = False,
) -> None:
    """Draw a target crosshair at ``center`` (brief §9.5).

    A plus made of four arms (up/down/left/right) that stop short of the centre
    by ``gap`` pixels, optionally with a centre ``dot`` and small end ``ticks``
    (short cross-strokes at each arm tip). Drawn *in place*; every stroke is
    clipped to the canvas, so a crosshair near (or off) the edge simply loses the
    out-of-bounds arms.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Handle over a straight-alpha RGBA canvas.
    center : (cx, cy)
        Crosshair centre in absolute pixels.
    size : int
        Half-length of each arm (distance from centre to arm tip), in pixels.
        ``<= 0`` -> no-op (nothing to draw).
    color_rgba : (r, g, b, a)
        Stroke colour, floats ``0..1``.
    line_width : int
        Stroke thickness in pixels (``>= 1``).
    gap : int
        Radius of the empty centre region; arms start at ``gap`` from the
        centre. Clamped to ``[0, size]`` (a gap >= size leaves no arm, only the
        optional dot/ticks).
    dot : bool, default True
        Draw a small filled centre dot.
    ticks : bool, default False
        Draw a short perpendicular tick at each arm tip.

    Returns
    -------
    None
    """
    if center is None:
        return
    try:
        cx = _f(center[0])
        cy = _f(center[1])
    except (TypeError, IndexError):
        return

    sz = _f(size)
    if sz <= 0.0:
        return
    color = _rgba_255(color_rgba)
    if color[3] <= 0:
        return
    lw = _safe_line_width(line_width)
    W, H = _canvas_size(draw)
    if W <= 0 or H <= 0:
        return

    g = _clampf(_f(gap), 0.0, sz)

    # Four arms (only drawn where the inner gap is shorter than the arm).
    if sz - g > 0.0:
        _line(draw, (cx, cy - g), (cx, cy - sz), color, lw, W, H)  # up
        _line(draw, (cx, cy + g), (cx, cy + sz), color, lw, W, H)  # down
        _line(draw, (cx - g, cy), (cx - sz, cy), color, lw, W, H)  # left
        _line(draw, (cx + g, cy), (cx + sz, cy), color, lw, W, H)  # right

    # End ticks: short perpendicular strokes at each arm tip.
    if ticks:
        tick = max(2.0, sz * 0.25)
        # vertical arm tips -> horizontal ticks
        _line(draw, (cx - tick, cy - sz), (cx + tick, cy - sz), color, lw, W, H)
        _line(draw, (cx - tick, cy + sz), (cx + tick, cy + sz), color, lw, W, H)
        # horizontal arm tips -> vertical ticks
        _line(draw, (cx - sz, cy - tick), (cx - sz, cy + tick), color, lw, W, H)
        _line(draw, (cx + sz, cy - tick), (cx + sz, cy + tick), color, lw, W, H)

    # Centre dot.
    if dot:
        # Dot radius scales gently with line width; minimum 1px so it's visible.
        rad = max(1.0, lw * 0.75)
        dx1, dy1, dx2, dy2 = cx - rad, cy - rad, cx + rad, cy + rad
        # Only draw if any of the dot is on-canvas (clamp the bbox; skip if it
        # collapses entirely off-frame).
        ddx1 = _clampf(dx1, 0.0, W)
        ddy1 = _clampf(dy1, 0.0, H)
        ddx2 = _clampf(dx2, 0.0, W)
        ddy2 = _clampf(dy2, 0.0, H)
        if (ddx2 - ddx1) >= 0.0 and (ddy2 - ddy1) >= 0.0 and 0.0 <= cx <= W and 0.0 <= cy <= H:
            draw.ellipse(
                [(_iround(dx1), _iround(dy1)), (_iround(dx2), _iround(dy2))],
                fill=color,
            )


def draw_registration(
    draw: "ImageDraw.ImageDraw",
    box: Box,
    color_rgba: Sequence[float],
    density: float,
    rng: Optional[random.Random] = None,
) -> None:
    """Scatter micro registration marks around ``box`` (brief §9.6).

    Small technical marks — micro **diamonds**, **plus** signs and **ticks** —
    placed at the box corners and along its edges, with the *count* governed by
    ``density``. This is the fine "machine-vision" detailing layer. Mark glyph
    and (subtle) jitter are chosen deterministically from ``rng`` so the pattern
    is reproducible per track/frame. Drawn *in place*; every glyph is clipped to
    the canvas.

    Parameters
    ----------
    draw : PIL.ImageDraw.ImageDraw
        Handle over a straight-alpha RGBA canvas.
    box : Box
        Anchor box ``xyxy``. Normalized + clamped internally; degenerate / off
        -frame -> no-op.
    color_rgba : (r, g, b, a)
        Mark colour, floats ``0..1``.
    density : float
        ``0`` -> nothing. Otherwise clamped to ``(0, 1]`` and mapped to a number
        of edge marks per side (``0`` density draws only the four corner marks;
        density ``1`` packs the most). Corner marks are always drawn when
        ``density > 0``.
    rng : random.Random | None
        Deterministic source for glyph choice + micro jitter. ``None`` -> fixed
        -seed Random (still deterministic).

    Returns
    -------
    None
    """
    d = _f(density)
    if d <= 0.0:
        return
    prepared = _prepared_box(draw, box)
    if prepared is None:
        return
    x1, y1, x2, y2 = prepared
    color = _rgba_255(color_rgba)
    if color[3] <= 0:
        return
    W, H = _canvas_size(draw)
    r = _as_rng(rng)

    d = _clampf(d, 0.0, 1.0)
    bw = x2 - x1
    bh = y2 - y1

    # Mark size scales with the box's short side but stays "micro" (2..6 px).
    short = min(bw, bh)
    msize = _clampf(short * 0.03, 2.0, 6.0)

    # --- glyph painters ---------------------------------------------------
    def _diamond(px: float, py: float, s: float) -> None:
        pts = [
            (px, py - s),
            (px + s, py),
            (px, py + s),
            (px - s, py),
        ]
        # Clip: skip glyphs whose centre is off-canvas (PIL still clamps the
        # polygon, but skipping keeps partial glyphs from smearing the edge).
        if not (0.0 <= px <= W and 0.0 <= py <= H):
            return
        draw.polygon([(_iround(a), _iround(b)) for a, b in pts], fill=color)

    def _plus(px: float, py: float, s: float) -> None:
        _line(draw, (px - s, py), (px + s, py), color, 1, W, H)
        _line(draw, (px, py - s), (px, py + s), color, 1, W, H)

    def _tick(px: float, py: float, s: float, vertical: bool) -> None:
        if vertical:
            _line(draw, (px, py - s), (px, py + s), color, 1, W, H)
        else:
            _line(draw, (px - s, py), (px + s, py), color, 1, W, H)

    glyphs = ("diamond", "plus", "tick")

    def _mark(px: float, py: float, vertical_tick: bool) -> None:
        # micro jitter (deterministic), kept within a sub-pixel-ish range so
        # marks read as "registration" not "noise".
        jx = (r.random() - 0.5) * msize * 0.4
        jy = (r.random() - 0.5) * msize * 0.4
        gx = px + jx
        gy = py + jy
        kind = glyphs[r.randrange(len(glyphs))]
        if kind == "diamond":
            _diamond(gx, gy, msize)
        elif kind == "plus":
            _plus(gx, gy, msize)
        else:
            _tick(gx, gy, msize, vertical_tick)

    # --- corner marks (always) -------------------------------------------
    _mark(x1, y1, vertical_tick=False)
    _mark(x2, y1, vertical_tick=False)
    _mark(x1, y2, vertical_tick=False)
    _mark(x2, y2, vertical_tick=False)

    # --- edge marks (count from density) ---------------------------------
    # density 0 -> 0 interior marks per edge; density 1 -> up to 4 per edge.
    per_edge = int(round(d * 4.0))
    if per_edge <= 0:
        return

    for k in range(1, per_edge + 1):
        t = k / (per_edge + 1)  # interior fractions, never on the corner
        # top / bottom edges -> vertical ticks; left / right -> horizontal
        _mark(x1 + bw * t, y1, vertical_tick=True)   # top
        _mark(x1 + bw * t, y2, vertical_tick=True)   # bottom
        _mark(x1, y1 + bh * t, vertical_tick=False)  # left
        _mark(x2, y1 + bh * t, vertical_tick=False)  # right
