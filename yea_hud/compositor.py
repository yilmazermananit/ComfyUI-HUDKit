"""compositor.py — per-frame HUD draw orchestration (brief §17).

This is **the integration point** of ComfyUI-YEAHUD: the seam where the
schema-agnostic business logic (tracking, style, placement, captions, assets)
is finally rasterized over the real video frames. The node layer
(``nodes/hud_renderer.py``) builds, per frame, a list of
:class:`yea_hud.types.RenderItem` and hands the whole batch here together
with an :class:`yea_hud.assets.AssetBank`, the resolved palette and a flat
``params`` dict; :func:`render_frames` returns a brand-new IMAGE tensor.

Schema-agnostic / Cloud-safe
----------------------------
Imports only the Python standard library, :mod:`numpy`, :mod:`PIL` (Pillow),
:mod:`torch`, and sibling ``yea_hud`` modules. There are **no** ComfyUI
imports (no ``comfy`` / ``comfy_api`` / ``nodes``), no network, no subprocess,
no ``eval``/``exec``, no file writes, no arbitrary-path reads — pure
deterministic compute. The only objects that cross the ComfyUI boundary are the
``torch`` IMAGE tensor (in and out) and the duck-typed ``progress`` /
``interrupt`` callables, which are used purely behaviourally
(``progress.update(1)`` / ``interrupt()``).

Tensor / array contract (see ``docs/INTERFACES.md`` "## compositor.py")
----------------------------------------------------------------------
* Input ``frames`` is a ``torch`` IMAGE tensor ``[B, H, W, 3]`` (or
  ``[B, H, W, 4]``) of float in ``0..1`` — the ComfyUI IMAGE batch convention
  (``docs/runtime_findings.md`` §5). It is **never** mutated in place.
* Output is a fresh ``torch.Tensor`` IMAGE ``[B, H, W, 3]`` ``float32`` in
  ``0..1``; the batch length ``B`` is always preserved.
* For each frame the base image is moved to CPU **once** as numpy
  ``float32 [H, W, 3]``; a **single** straight-alpha RGBA overlay
  (``float32 [H, W, 4]``) is built with PIL and alpha-composited over the base
  **once**; the result is converted back to a tensor **once** and the frames are
  ``torch.stack``-ed.
* An empty plan frame (no :class:`RenderItem`) copies the base frame unchanged
  (no overlay built, no detection passthrough — brief §25.4).

Draw order (brief §17, per frame, back-to-front)
------------------------------------------------
For every :class:`RenderItem` in plan order:

1. **solid blocks**     — translucent fill (``draw_solid``);
2. **frame assets / procedural frames** — the user's recolored frame PNG, else
   the vector primitive (corner brackets / outline);
3. **crosshair**        — the target/reticle PNG, else the procedural crosshair;
4. **label strips**     — the solid caption-strip background + text (a single
   :func:`yea_hud.text_render.render_label_strip` array already contains
   both, so steps "label strips" and "text" of the contract are produced in one
   call and blitted here);
5. **micro details**    — registration marks (``draw_registration``).

Per-:class:`RenderItem` micro-variation is honoured: ``jitter`` offsets the
placement box, ``opacity_mul`` scales the *whole* item's alpha, ``line_dropout``
suppresses the line/box primitive for that frame, and ``fill_flash`` /
``target_flash`` momentarily boost the fill / target presence.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from . import geometry
from . import procedural_hud
from . import text_render
from . import palette as palette_mod
from .types import RenderItem

__all__ = [
    "render_frames",
    "composite_rgba_over",
]


# ---------------------------------------------------------------------------
# Small numeric helpers (kept local; mirror the conventions in sibling modules)
# ---------------------------------------------------------------------------

def _finite(value, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float; ``None``/NaN/Inf/garbage -> ``default``."""
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out or out in (float("inf"), float("-inf")):
        return float(default)
    return out


def _clamp01(value: float) -> float:
    """Clamp a scalar into the closed unit interval ``[0.0, 1.0]``."""
    v = _finite(value, 0.0)
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _param(params: Optional[dict], key: str, default):
    """Read ``params[key]`` defensively, falling back to ``default``.

    ``params`` may be ``None`` or missing keys (the node always supplies the
    full set, but the compositor must not assume it). The value is returned
    verbatim — callers coerce/clamp as appropriate.
    """
    if not isinstance(params, dict):
        return default
    if key not in params:
        return default
    return params[key]


# ---------------------------------------------------------------------------
# Tensor <-> numpy boundary (the only torch-touching code)
# ---------------------------------------------------------------------------

def _frame_to_numpy_rgb(frame: "torch.Tensor") -> np.ndarray:
    """Move one IMAGE frame tensor ``[H, W, C]`` to CPU numpy ``float32 [H, W, 3]``.

    Done **once** per frame. Accepts ``C`` of 3 (RGB) or 4 (RGBA — alpha is
    dropped, RGB taken) or 1 (grayscale -> replicated). The result is a fresh,
    contiguous, writable ``float32`` array clipped to ``0..1`` (the base onto
    which the overlay is composited). The source tensor is never mutated
    (``detach`` + a copy on conversion).
    """
    arr = frame.detach().to("cpu").contiguous().numpy()
    if arr.ndim != 3:
        raise ValueError(
            f"render_frames: expected a per-frame [H,W,C] tensor, got shape {arr.shape}"
        )
    arr = arr.astype(np.float32, copy=False)
    c = arr.shape[2]
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif c >= 3:
        arr = arr[:, :, :3]
    else:  # c == 2 or 0 — degenerate; pad/clip to 3 channels defensively.
        if c == 2:
            arr = np.repeat(arr[:, :, :1], 3, axis=2)
        else:
            raise ValueError(f"render_frames: unsupported channel count {c}")
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False))


# ---------------------------------------------------------------------------
# Straight-alpha compositing
# ---------------------------------------------------------------------------

def composite_rgba_over(base_rgb_np, overlay_rgba_np) -> np.ndarray:
    """Alpha-composite a straight-alpha RGBA overlay over an RGB base (source-over).

    Implements the standard *straight* (non-premultiplied) "source-over" blend::

        out_rgb = overlay_rgb * a + base_rgb * (1 - a)

    where ``a`` is the overlay's per-pixel alpha in ``0..1``. The base is treated
    as fully opaque (alpha 1), so the composite stays opaque and the returned
    array is a 3-channel RGB image (the HUD never makes the video transparent).

    Args:
        base_rgb_np: base image ``[H, W, 3]`` float in ``0..1`` (an ``[H, W, 4]``
            base is accepted — its alpha is ignored and the RGB used).
        overlay_rgba_np: straight-alpha overlay ``[H, W, 4]`` float in ``0..1``
            (an ``[H, W, 3]`` overlay is treated as fully opaque). Must share the
            base's ``(H, W)``.

    Returns:
        ``np.ndarray`` ``float32 [H, W, 3]`` in ``0..1`` — a fresh array (the
        inputs are not mutated).

    Raises:
        ValueError: if the spatial dimensions of base and overlay differ, or a
            shape cannot be interpreted as an image / overlay.
    """
    base = np.asarray(base_rgb_np)
    over = np.asarray(overlay_rgba_np)

    if base.ndim != 3 or base.shape[2] < 3:
        raise ValueError(
            f"composite_rgba_over: base must be [H,W,>=3], got {base.shape}"
        )
    if over.ndim != 3 or over.shape[2] < 3:
        raise ValueError(
            f"composite_rgba_over: overlay must be [H,W,>=3], got {over.shape}"
        )
    if base.shape[0] != over.shape[0] or base.shape[1] != over.shape[1]:
        raise ValueError(
            "composite_rgba_over: base/overlay spatial dims differ "
            f"({base.shape[:2]} vs {over.shape[:2]})"
        )

    base_rgb = np.clip(base[:, :, :3].astype(np.float32, copy=False), 0.0, 1.0)
    over_rgb = np.clip(over[:, :, :3].astype(np.float32, copy=False), 0.0, 1.0)

    if over.shape[2] >= 4:
        alpha = np.clip(over[:, :, 3:4].astype(np.float32, copy=False), 0.0, 1.0)
    else:
        # No alpha channel on the overlay -> fully opaque.
        alpha = np.ones((over.shape[0], over.shape[1], 1), dtype=np.float32)

    out = over_rgb * alpha + base_rgb * (1.0 - alpha)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0).astype(np.float32, copy=False))


def _blit_rgba(layer: "Image.Image", patch_rgba: np.ndarray, x: int, y: int) -> None:
    """Source-over a straight-alpha RGBA numpy ``patch`` onto the PIL ``layer``.

    ``layer`` is the single per-frame straight-alpha RGBA overlay (mode
    ``'RGBA'``); ``patch_rgba`` is an asset / label-strip array
    (``float32 [h, w, 4]`` in ``0..1``). The patch is clipped to the overlay
    bounds — negative ``(x, y)`` (patch hanging off the top/left) is handled by
    cropping the source, and overflow on the right/bottom by PIL's own
    ``alpha_composite`` destination clipping — and composited with proper
    *straight*-alpha source-over so the single overlay accumulates correctly
    (vector primitives and array patches share this one buffer, preserving the
    brief §17 per-item draw order).

    Off-frame or empty patches are silent no-ops. Mutates ``layer`` in place.
    """
    if patch_rgba is None:
        return
    patch = np.asarray(patch_rgba, dtype=np.float32)
    if patch.ndim != 3 or patch.shape[2] != 4 or patch.size == 0:
        return

    W, H = layer.size  # PIL size is (W, H)
    ph, pw = patch.shape[0], patch.shape[1]
    if ph <= 0 or pw <= 0 or W <= 0 or H <= 0:
        return

    x = int(round(_finite(x, 0.0)))
    y = int(round(_finite(y, 0.0)))

    # Crop the source for negative offsets so ``dest`` stays non-negative (PIL's
    # alpha_composite ``dest`` must be >= 0; it clips the right/bottom itself).
    src_x0 = 0
    src_y0 = 0
    dest_x = x
    dest_y = y
    if dest_x < 0:
        src_x0 = -dest_x
        dest_x = 0
    if dest_y < 0:
        src_y0 = -dest_y
        dest_y = 0
    if src_x0 >= pw or src_y0 >= ph:
        return  # entirely off the top/left
    if dest_x >= W or dest_y >= H:
        return  # entirely off the bottom/right

    sub = np.clip(patch[src_y0:ph, src_x0:pw, :], 0.0, 1.0)
    if sub.size == 0:
        return
    u8 = np.ascontiguousarray((sub * 255.0 + 0.5).astype(np.uint8))
    patch_img = Image.fromarray(u8, mode="RGBA")
    # alpha_composite performs straight-alpha source-over of patch_img onto layer
    # at (dest_x, dest_y), clipping any right/bottom overflow.
    layer.alpha_composite(patch_img, dest=(dest_x, dest_y))


def _scale_patch_opacity(patch: Optional[np.ndarray], opacity_mul: float) -> Optional[np.ndarray]:
    """Return a copy of an RGBA ``patch`` with its alpha scaled by ``opacity_mul``.

    Used so a frame/target asset honours a :class:`RenderItem`'s per-frame
    ``opacity_mul`` without mutating the (possibly cached) source array.
    ``None`` in -> ``None`` out; ``opacity_mul`` of (effectively) ``1.0`` returns
    the patch unchanged (no copy needed — the blitter only reads it).
    """
    if patch is None:
        return None
    mul = _clamp01(opacity_mul)
    if abs(mul - 1.0) < 1e-6:
        return patch
    out = np.array(patch, dtype=np.float32, copy=True)
    out[:, :, 3] *= mul
    return out


# ---------------------------------------------------------------------------
# Per-item geometry helpers
# ---------------------------------------------------------------------------

def _jittered_box(box, jitter, W: int, H: int):
    """Offset ``box`` by an integer ``(dx, dy)`` jitter and clamp to the frame."""
    x1, y1, x2, y2 = geometry.normalize_box(*box)
    try:
        dx = int(jitter[0])
        dy = int(jitter[1])
    except (TypeError, ValueError, IndexError):
        dx = dy = 0
    return geometry.clamp_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), W, H)


def _color_rgba(rgb: Sequence[float], alpha: float) -> Tuple[float, float, float, float]:
    """Build a ``(r, g, b, a)`` float tuple from an RGB triple + alpha (all clamped)."""
    try:
        r = _clamp01(rgb[0])
        g = _clamp01(rgb[1])
        b = _clamp01(rgb[2])
    except (TypeError, IndexError):
        r = g = b = 0.0
    return (r, g, b, _clamp01(alpha))


# ---------------------------------------------------------------------------
# Per-item drawing — the heart of the draw-order contract
# ---------------------------------------------------------------------------

def _crosshair_size(box, target_scale: float) -> int:
    """Half-arm length (px) of a procedural crosshair for ``box``.

    Scales with the box's shorter side by ``target_scale`` so a small box gets a
    small reticle and a large box a large one, with a sensible minimum (4 px) so
    the crosshair is always visible. ``target_scale`` is clamped to ``(0, 1]``
    (a non-positive value falls back to the ``0.18`` default). The returned value
    is the crosshair's *half*-arm length, i.e. the full span is ``~2 * size``.
    """
    x1, y1, x2, y2 = geometry.normalize_box(*box)
    short = min(max(0.0, x2 - x1), max(0.0, y2 - y1))
    scale = _clamp01(target_scale)
    if scale <= 0.0:
        scale = 0.18
    return int(max(4, round(0.5 * short * scale)))


def _draw_item(
    layer: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    item: RenderItem,
    *,
    assets,
    palette,
    params: dict,
    font_path: Optional[str],
    W: int,
    H: int,
) -> None:
    """Draw a single :class:`RenderItem` onto the shared per-frame RGBA overlay.

    Strictly follows the brief §17 draw order (solid -> frame asset / procedural
    frame -> crosshair -> label strip + text -> micro details). ``layer`` is the
    single straight-alpha RGBA PIL overlay and ``draw`` is an ``ImageDraw`` bound
    to that *same* image, so vector primitives (drawn via ``draw``) and array
    patches (asset / label strips, blitted via :func:`_blit_rgba`) accumulate
    into one buffer in the exact per-item order. All per-item micro-variation
    (jitter / opacity / dropout / flashes) is applied here.
    """
    style = item.style

    # Resolve effective placement (jittered + clamped) once.
    box = _jittered_box(item.box, item.jitter, W, H)
    if geometry.is_degenerate(box, min_size=1.0):
        # Nothing meaningful to draw for a collapsed box, but a crosshair-only or
        # label-only item could still anchor at its centre; keep going so those
        # paths (which clip themselves) still run on a thin box. We only bail when
        # the box has truly zero extent on BOTH axes.
        x1, y1, x2, y2 = box
        if (x2 - x1) <= 0.0 and (y2 - y1) <= 0.0:
            return

    opacity_mul = _clamp01(item.opacity_mul)
    if opacity_mul <= 0.0:
        return  # fully transparent this frame -> draw nothing

    primitive = style.primitive_style
    color_rgb = item.color_rgb

    # Resolved global opacities (params), pre-multiplied by the item's per-frame
    # opacity so the WHOLE item fades together under micro-variation.
    frame_opacity = _clamp01(_param(params, "frame_opacity", 1.0))
    fill_opacity = _clamp01(_param(params, "fill_opacity", 0.30))
    label_opacity = _clamp01(_param(params, "label_opacity", 0.92))
    line_width = int(max(1, _finite(_param(params, "line_width", 3), 3.0)))
    target_scale = _finite(_param(params, "target_scale", 0.18), 0.18)
    label_scale = _finite(_param(params, "label_scale", 1.0), 1.0)
    label_text_color_mode = _param(params, "label_text_color_mode", "auto")
    detail_density = _clamp01(_param(params, "detail_density", 0.0))

    glitch_phase = _finite(getattr(style, "glitch_phase", 0.0), 0.0)

    # ----- 1. SOLID BLOCKS ------------------------------------------------
    # Drawn first (furthest back) so brackets / text sit on top of the wash.
    if style.use_fill or primitive == "solid":
        fa = fill_opacity
        if item.fill_flash:
            # A one-frame flash pushes the fill toward fully present.
            fa = _clamp01(fa + (1.0 - fa) * 0.6)
        fa = _clamp01(fa * opacity_mul)
        if fa > 0.0:
            procedural_hud.draw_solid(draw, box, _color_rgba(color_rgb, fa))

    # ----- 2. FRAME ASSET / PROCEDURAL FRAME ------------------------------
    # The user's recolored frame PNG wins when this track was assigned one and
    # the bank can render it; otherwise the vector primitive (corners/outline).
    # 'target_only' draws no box/frame here (its visual is the crosshair).
    # 'line_dropout' suppresses the box/frame for this one frame (a flicker).
    drew_asset = False
    frame_asset_id = int(getattr(style, "frame_asset_id", -1))
    wants_asset = (
        frame_asset_id >= 0
        and primitive not in ("target_only",)
        and not item.line_dropout
    )
    if wants_asset and assets is not None:
        rect = geometry.asset_rect(
            box,
            getattr(assets, "fit_frame", "stretch"),
            0.0,
            None,
        )
        rx1, ry1, rx2, ry2 = geometry.normalize_box(*rect)
        out_w = int(round(max(1.0, rx2 - rx1)))
        out_h = int(round(max(1.0, ry2 - ry1)))
        patch = None
        try:
            patch = assets.frame_rgba(
                frame_asset_id, out_w, out_h, color_rgb, frame_opacity
            )
        except Exception:  # noqa: BLE001 — a bad asset must never break the batch
            patch = None
        if patch is not None:
            patch = _scale_patch_opacity(patch, opacity_mul)
            # The bank snaps to an 8px bucket, so the patch size may differ from
            # (out_w,out_h); centre it within the placement rect.
            ph, pw = patch.shape[0], patch.shape[1]
            bx = int(round(rx1 + ((rx2 - rx1) - pw) * 0.5))
            by = int(round(ry1 + ((ry2 - ry1) - ph) * 0.5))
            _blit_rgba(layer, patch, bx, by)
            drew_asset = True

    if not drew_asset and not item.line_dropout:
        # Vector primitive fallback / the chosen procedural style.
        if primitive == "corners":
            procedural_hud.draw_corners(
                draw, box, _color_rgba(color_rgb, frame_opacity * opacity_mul), line_width
            )
        elif primitive == "outline":
            # Glitch segments derived deterministically from the track's stable
            # glitch personality (0 phase -> clean outline).
            segs = int(round(glitch_phase * 6.0)) if glitch_phase > 0.0 else 0
            r = None
            if segs > 1:
                from .utils import rng_for

                r = rng_for(item.track_id, item.order_index, "outline_glitch")
            procedural_hud.draw_outline(
                draw,
                box,
                _color_rgba(color_rgb, frame_opacity * opacity_mul),
                line_width,
                glitch_segments=segs,
                rng=r,
            )
        elif primitive == "frame_asset":
            # Wanted a frame asset but none was renderable (mask-only slot / OOR):
            # degrade to corner brackets so the track is still anchored.
            procedural_hud.draw_corners(
                draw, box, _color_rgba(color_rgb, frame_opacity * opacity_mul), line_width
            )
        # primitive == 'target_only' -> intentionally nothing here.

    # ----- 3. CROSSHAIR ---------------------------------------------------
    if style.use_target or primitive == "target_only":
        cx, cy = geometry.target_center(box)
        target_asset_id = int(getattr(style, "target_asset_id", -1))
        # A target flash boosts the crosshair opacity for one frame.
        t_op = frame_opacity
        if item.target_flash:
            t_op = _clamp01(t_op + (1.0 - t_op) * 0.6)
        t_op = _clamp01(t_op * opacity_mul)

        drew_target_asset = False
        if target_asset_id >= 0 and assets is not None and t_op > 0.0:
            # Size the reticle PNG to the crosshair span.
            span = max(8, min(_crosshair_size(box, target_scale) * 2, int(H * 0.10)))
            patch = None
            try:
                patch = assets.target_rgba(
                    target_asset_id, span, span, color_rgb, t_op
                )
            except Exception:  # noqa: BLE001
                patch = None
            if patch is not None:
                patch = _scale_patch_opacity(patch, opacity_mul)
                ph, pw = patch.shape[0], patch.shape[1]
                _blit_rgba(layer, patch, int(round(cx - pw * 0.5)), int(round(cy - ph * 0.5)))
                drew_target_asset = True

        if not drew_target_asset and t_op > 0.0:
            size = min(_crosshair_size(box, target_scale), int(H * 0.05))
            gap = max(1, int(round(size * 0.35)))
            procedural_hud.draw_crosshair(
                draw,
                (cx, cy),
                size,
                _color_rgba(color_rgb, t_op),
                line_width,
                gap,
                dot=True,
                ticks=True,
            )

    # ----- 4. LABEL STRIP + TEXT -----------------------------------------
    # A single strip array carries BOTH the solid background and the text (the
    # contract's "label strips" then "text" steps), so it is produced once and
    # blitted on top of the box/crosshair.
    label_text = item.label_text
    if style.use_caption and label_text and getattr(item, "label_visible", True):
        # Font size scales with the box height (clamped) and the user label_scale.
        # Child / micro_tag labels render a touch smaller for clear hierarchy.
        x1, y1, x2, y2 = box
        label_style_name = getattr(item, "label_style", "solid_bar")
        # Uniform SMALL labels (editorial HUD): size scales with FRAME height, NOT box
        # height, so a large box never gets a giant caption. Tight cap; micro tags smaller.
        _frac = 0.016 if label_style_name == "micro_tag" else 0.019
        _lo, _hi = (9, 15) if label_style_name == "micro_tag" else (11, 21)
        font_size = int(min(_hi, max(_lo, round(H * _frac * max(0.4, label_scale)))))
        font = text_render.get_font(font_size, font_path=font_path)

        # Truncate to a width budget: the pre-resolved label rect (from the
        # renderer's placement engine) if present, else the box / frame width.
        bw = max(0.0, x2 - x1)
        pre = getattr(item, "label_rect", None)
        if pre is not None:
            max_text_w = int(max(8, (pre[2] - pre[0]) - 6))
        else:
            max_text_w = int(max(8, (bw if bw > 8.0 else float(W)) - 8))
        fitted = text_render.truncate_ellipsis(label_text, font, max_text_w)

        # Strip background: a child/category override colour when provided, else the
        # track colour. Text is the readable contrast colour. label_style controls
        # how solid the bar is (solid -> transparent -> text-only micro tag).
        strip_bg = getattr(item, "label_color_rgb", None) or color_rgb
        text_rgb = palette_mod.contrast_text_color(strip_bg, mode=label_text_color_mode)
        strip_alpha = _clamp01(label_opacity * opacity_mul)
        if label_style_name == "transparent_bar":
            strip_alpha *= 0.35
        elif label_style_name in ("minimal_text", "micro_tag"):
            strip_alpha = 0.0          # text only, no bar
            text_rgb = strip_bg        # coloured text on minimal styles

        strip = text_render.render_label_strip(
            None,
            text=fitted,
            font=font,
            fg_rgb=text_rgb,
            bg_rgb=strip_bg,
            bg_opacity=strip_alpha,
        )
        # Placement: honour the renderer's pre-resolved rect (collision/temporal
        # engine); otherwise fall back to the track's preferred side, clamped.
        sh, sw = strip.shape[0], strip.shape[1]
        if pre is not None:
            rx1, ry1 = float(pre[0]), float(pre[1])
        else:
            side = getattr(style, "caption_side", "top")
            rect, _resolved = geometry.label_rect(box, sw, sh, W, H, side)
            rx1, ry1, _rx2, _ry2 = rect
        _blit_rgba(layer, strip, int(round(rx1)), int(round(ry1)))

    # ----- 5. MICRO DETAILS (registration marks) -------------------------
    # The finest "machine-vision" layer, drawn last so it reads on top. Skipped
    # for target-only items (which have no box edges to register against) and
    # when the box was dropped this frame.
    if (
        detail_density > 0.0
        and not item.line_dropout
        and primitive != "target_only"
        and not geometry.is_degenerate(box, min_size=4.0)
    ):
        from .utils import rng_for

        reg_rng = rng_for(item.track_id, item.order_index, "registration")
        reg_alpha = _clamp01(frame_opacity * opacity_mul)
        if reg_alpha > 0.0:
            procedural_hud.draw_registration(
                draw, box, _color_rgba(color_rgb, reg_alpha), detail_density, rng=reg_rng
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_frames(
    frames,
    plans: List[List[RenderItem]],
    *,
    assets,
    palette,
    params: dict,
    font_path: Optional[str],
    progress=None,
    interrupt=None,
) -> "torch.Tensor":
    """Render the HUD overlay for every frame and return a new IMAGE tensor.

    This is the integration point between the schema-agnostic ``yea_hud``
    logic and ComfyUI's tensor world. For each frame the base image is moved to
    CPU numpy once, a single straight-alpha RGBA overlay is built in the brief
    §17 draw order, alpha-composited over the base once, converted back to a
    tensor once, and finally all frames are stacked. The input tensor is never
    mutated; the output preserves the batch length and is ``float32`` in
    ``0..1``.

    Args:
        frames: ``torch`` IMAGE tensor ``[B, H, W, 3]`` (or ``[B, H, W, 4]``) of
            float in ``0..1`` (ComfyUI IMAGE batch). ``B`` may be any ``>= 0``.
        plans: A per-frame list (length ``B``) of :class:`RenderItem` lists. An
            empty inner list means "draw nothing on this frame" (the base frame
            is passed through unchanged). If ``plans`` is shorter than ``B`` the
            missing frames are treated as empty; extra plan entries are ignored.
        assets: An :class:`yea_hud.assets.AssetBank` (or any object exposing
            ``frame_rgba`` / ``target_rgba`` / ``fit_frame``); may be ``None`` —
            then only procedural primitives are drawn.
        palette: The resolved palette (list of RGB triples). Carried for API
            symmetry / future use; per-item colours arrive pre-resolved on each
            :class:`RenderItem` (``color_rgb``).
        params: Flat dict of render parameters. Recognised keys (all optional,
            with safe defaults): ``frame_opacity``, ``fill_opacity``,
            ``label_opacity``, ``line_width``, ``target_scale``, ``label_scale``,
            ``label_text_color_mode``, ``detail_density``.
        font_path: Optional explicit font path for label strips (forwarded to
            :func:`yea_hud.text_render.get_font`; ``None`` uses the bundled /
            system / default chain).
        progress: Optional progress reporter exposing ``update(1)`` (e.g. a
            ComfyUI ``ProgressBar``); called once per frame. Duck-typed; any
            error from it is swallowed so progress reporting never breaks a
            render.
        interrupt: Optional zero-arg callable checked once per frame *before*
            drawing it; if it raises, the exception propagates (cooperative
            cancellation — ComfyUI passes
            ``throw_exception_if_processing_interrupted``).

    Returns:
        ``torch.Tensor`` IMAGE ``[B, H, W, 3]`` ``float32`` in ``0..1`` on the
        same device as ``frames`` (CPU compute, moved back to the input device),
        with the batch length preserved. For ``B == 0`` an empty
        ``[0, 0, 0, 3]`` tensor is returned.

    Raises:
        TypeError: if ``frames`` is ``None``.
        Whatever ``interrupt()`` raises (cooperative cancellation).
    """
    if frames is None:
        raise TypeError("render_frames: frames is None")

    # Remember the source device/dtype so we hand back a tensor on the same
    # device the node gave us (the node may have frames on GPU/MPS).
    src_device = getattr(frames, "device", None)

    B = int(frames.shape[0])
    if B <= 0:
        # Preserve a well-formed empty IMAGE batch.
        return torch.zeros((0, 0, 0, 3), dtype=torch.float32)

    plans = plans if plans is not None else []
    n_plans = len(plans)

    out_frames: List["torch.Tensor"] = []

    for f in range(B):
        # Cooperative cancellation: check BEFORE doing the frame's work.
        if interrupt is not None:
            interrupt()  # may raise -> propagate (do not swallow)

        # (1) base image to CPU numpy — exactly once per frame.
        base_rgb = _frame_to_numpy_rgb(frames[f])
        H, W = base_rgb.shape[0], base_rgb.shape[1]

        # The plan for this frame (missing -> empty; non-list -> empty).
        items: List[RenderItem] = []
        if f < n_plans:
            cand = plans[f]
            if isinstance(cand, (list, tuple)):
                items = [it for it in cand if isinstance(it, RenderItem)]

        if not items:
            # Empty plan frame -> base unchanged (a fresh contiguous copy so the
            # output never aliases the input tensor's backing buffer).
            result_np = np.ascontiguousarray(base_rgb)
        else:
            # (2) build ONE straight-alpha RGBA overlay for the whole frame. Both
            # the vector primitives (via ``draw``) and the asset / label-strip
            # arrays (via ``_blit_rgba``) accumulate into this single PIL layer,
            # so the per-item draw order from brief §17 is preserved exactly.
            layer = procedural_hud.new_rgba_layer(W, H)
            draw = ImageDraw.Draw(layer)

            for item in items:
                _draw_item(
                    layer,
                    draw,
                    item,
                    assets=assets,
                    palette=palette,
                    params=params,
                    font_path=font_path,
                    W=W,
                    H=H,
                )

            # Materialise the single overlay to numpy ONCE and alpha-composite it
            # over the base ONCE.
            overlay_np = np.ascontiguousarray(np.asarray(layer, dtype=np.float32) / 255.0)
            result_np = composite_rgba_over(base_rgb, overlay_np)

        # (3) back to a tensor — exactly once per frame.
        out_frames.append(torch.from_numpy(np.ascontiguousarray(result_np)).float())

        # Per-frame progress tick (never let a reporter error break the render).
        if progress is not None:
            try:
                progress.update(1)
            except Exception:  # noqa: BLE001
                pass

    out = torch.stack(out_frames, dim=0)
    if src_device is not None:
        try:
            out = out.to(src_device)
        except Exception:  # noqa: BLE001 — device move must never break output
            pass
    return out
