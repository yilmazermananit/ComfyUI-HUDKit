"""Tests for ``yonder_hud.compositor`` (brief §17 / §25.4 compositor cases).

Runnable two ways:

    /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_compositor.py -q
    /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_compositor.py

The ``__main__`` block runs every ``test_*`` function, prints a summary, and
``sys.exit(1)`` on any failure. No ComfyUI, no network, no model files — only
small synthetic numpy / torch fixtures.

Coverage (brief §25.4 + the compositor contract in ``docs/INTERFACES.md``):

* output shape ``[B,H,W,3]``, dtype ``float32``, range ``0..1``, batch length;
* no-detection passthrough (empty plan frame == base unchanged), and a fully
  empty plans list;
* input tensor is NEVER mutated in place;
* RGBA ``[B,H,W,4]`` input -> RGB ``[B,H,W,3]`` output;
* ``composite_rgba_over`` straight-alpha source-over math + shape + errors;
* placement: a drawn item actually changes pixels inside its box region;
* fill opacity: ``use_fill`` honours ``fill_opacity`` (lower opacity -> closer to
  base), and ``fill_flash`` boosts it;
* per-item ``opacity_mul`` (0 -> nothing drawn; partial -> lighter overlay);
* per-item ``line_dropout`` suppresses the vector box;
* per-item ``jitter`` shifts the overlay and never escapes the frame;
* edge clipping: a box / crosshair near or off the frame edge never raises and
  never paints outside ``[0,W)x[0,H)``;
* label flip: a top-hugging caption is relocated on-frame (never clipped off);
* frame-asset + target-asset rendering paths (recolored PNG over the box);
* ``B == 0`` -> well-formed empty ``[0,0,0,3]`` tensor;
* progress ``.update(1)`` per frame; ``interrupt()`` propagates; plans shorter /
  longer than ``B`` handled (missing -> empty, extra -> ignored).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Make the package importable when run as a bare script from the repo root or
# from within the tests/ directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from yonder_hud import palette as palette_mod  # noqa: E402
from yonder_hud.assets import AssetBank  # noqa: E402
from yonder_hud.compositor import composite_rgba_over, render_frames  # noqa: E402
from yonder_hud.types import RenderItem, TrackStyle  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PALETTE = palette_mod.resolve_palette(None)

DEFAULT_PARAMS = {
    "frame_opacity": 1.0,
    "fill_opacity": 0.30,
    "label_opacity": 0.92,
    "line_width": 3,
    "target_scale": 0.18,
    "label_scale": 1.0,
    "label_text_color_mode": "auto",
    "detail_density": 0.0,
}


def _empty_assets(max_cache_items: int = 16) -> AssetBank:
    """An AssetBank with no connected slots (procedural-only)."""
    return AssetBank(
        [None] * 6,
        [None] * 2,
        trim=True,
        fit_frame="stretch",
        fit_target="contain",
        max_cache_items=max_cache_items,
    )


def _white_on_transparent_pair(h: int = 16, w: int = 16):
    """A Load-Image-style ``(IMAGE, MASK)`` pair: white square on transparent.

    The PNG is white RGB everywhere; the alpha is 1 inside a central square and 0
    outside. ComfyUI's Load Image emits ``MASK = 1 - alpha`` (see
    ``yonder_hud.alpha`` docstring), so we build the mask that way; the asset
    bank then recovers the true alpha as ``1 - mask``.
    """
    img = torch.ones((1, h, w, 3), dtype=torch.float32)  # white
    alpha = torch.zeros((h, w), dtype=torch.float32)
    alpha[h // 4 : h - h // 4, w // 4 : w - w // 4] = 1.0
    mask = (1.0 - alpha).unsqueeze(0)  # [1, h, w], ComfyUI MASK convention
    return img, mask


def _asset_bank_with_one_each() -> AssetBank:
    """An AssetBank with one connected frame asset and one target asset."""
    img, mask = _white_on_transparent_pair()
    return AssetBank(
        [(img, mask)] + [None] * 5,
        [(img, mask)] + [None] * 1,
        trim=True,
        fit_frame="stretch",
        fit_target="contain",
        max_cache_items=32,
    )


def _gray_frames(b: int, h: int, w: int, value: float = 0.5) -> torch.Tensor:
    """A ``[b,h,w,3]`` float32 tensor filled with ``value``."""
    return torch.full((b, h, w, 3), float(value), dtype=torch.float32)


def _style(**kw) -> TrackStyle:
    """A TrackStyle with sensible defaults, overridable by keyword."""
    base = dict(
        color_id=0,
        primitive_style="corners",
        frame_asset_id=-1,
        target_asset_id=-1,
        show=True,
        use_fill=False,
        use_target=False,
        use_caption=False,
        caption_side="top",
        glitch_phase=0.0,
    )
    base.update(kw)
    return TrackStyle(**base)


# Keyword names that belong on a TrackStyle (not on a RenderItem). When passed to
# the ``_item`` helper they are routed onto the style for convenience.
_STYLE_KWARGS = frozenset(
    (
        "color_id", "primitive_style", "frame_asset_id", "target_asset_id",
        "show", "use_fill", "use_target", "use_caption", "caption_side",
        "glitch_phase",
    )
)


def _item(box, *, color=(1.0, 0.0, 0.0), label="", **kw) -> RenderItem:
    """A RenderItem with a corners style by default; box is xyxy.

    Convenience: any ``TrackStyle`` field passed as a keyword (e.g. ``use_fill``)
    is folded onto the style. An explicit ``style=`` wins for fields it already
    sets; remaining style-kwargs override on a copy.
    """
    explicit_style = kw.pop("style", None)
    style_kw = {k: kw.pop(k) for k in list(kw) if k in _STYLE_KWARGS}
    if explicit_style is None:
        style_kw.setdefault("use_caption", bool(label))
        style = _style(**style_kw)
    else:
        style = explicit_style
        # Apply any extra style-kwargs that the explicit style did not pin.
        if style_kw:
            import dataclasses

            style = dataclasses.replace(style, **style_kw)
    fields = dict(
        track_id=kw.pop("track_id", 1),
        order_index=kw.pop("order_index", 1),
        box=tuple(float(v) for v in box),
        color_rgb=color,
        style=style,
        label_text=label,
    )
    fields.update(kw)  # remaining kwargs are genuine RenderItem fields
    return RenderItem(**fields)


def _render(frames, plans, *, assets=None, params=None, progress=None, interrupt=None):
    """Convenience wrapper with the default palette / params / empty assets."""
    return render_frames(
        frames,
        plans,
        assets=assets if assets is not None else _empty_assets(),
        palette=PALETTE,
        params=params if params is not None else DEFAULT_PARAMS,
        font_path=None,
        progress=progress,
        interrupt=interrupt,
    )


class _Progress:
    """Minimal ProgressBar stand-in counting ``update(n)`` calls."""

    def __init__(self) -> None:
        self.total = 0
        self.calls = 0

    def update(self, n) -> None:
        self.total += int(n)
        self.calls += 1


# ---------------------------------------------------------------------------
# composite_rgba_over — straight-alpha source-over
# ---------------------------------------------------------------------------

def test_composite_rgba_over_math_and_shape():
    base = np.full((4, 4, 3), 0.2, dtype=np.float32)
    overlay = np.zeros((4, 4, 4), dtype=np.float32)
    overlay[:, :, 0] = 1.0  # red
    overlay[:, :, 3] = 0.5  # half alpha
    out = composite_rgba_over(base, overlay)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32
    # out = over*a + base*(1-a): R = 1*0.5 + 0.2*0.5 = 0.6 ; G,B = 0 + 0.2*0.5 = 0.1
    assert abs(float(out[0, 0, 0]) - 0.6) < 1e-5
    assert abs(float(out[0, 0, 1]) - 0.1) < 1e-5
    assert abs(float(out[0, 0, 2]) - 0.1) < 1e-5


def test_composite_rgba_over_transparent_is_identity():
    base = np.random.RandomState(0).rand(6, 5, 3).astype(np.float32)
    overlay = np.zeros((6, 5, 4), dtype=np.float32)  # alpha 0 everywhere
    out = composite_rgba_over(base, overlay)
    assert np.allclose(out, base, atol=1e-6)


def test_composite_rgba_over_opaque_replaces():
    base = np.full((3, 3, 3), 0.9, dtype=np.float32)
    overlay = np.zeros((3, 3, 4), dtype=np.float32)
    overlay[:, :, :3] = 0.1
    overlay[:, :, 3] = 1.0  # fully opaque
    out = composite_rgba_over(base, overlay)
    assert np.allclose(out, 0.1, atol=1e-6)


def test_composite_rgba_over_does_not_mutate_inputs():
    base = np.full((3, 3, 3), 0.4, dtype=np.float32)
    overlay = np.zeros((3, 3, 4), dtype=np.float32)
    overlay[:, :, 0] = 1.0
    overlay[:, :, 3] = 0.7
    base_copy = base.copy()
    overlay_copy = overlay.copy()
    _ = composite_rgba_over(base, overlay)
    assert np.array_equal(base, base_copy)
    assert np.array_equal(overlay, overlay_copy)


def test_composite_rgba_over_rgb_overlay_treated_opaque():
    base = np.full((2, 2, 3), 0.5, dtype=np.float32)
    overlay_rgb = np.zeros((2, 2, 3), dtype=np.float32)  # no alpha channel
    out = composite_rgba_over(base, overlay_rgb)
    assert np.allclose(out, 0.0, atol=1e-6)  # opaque black


def test_composite_rgba_over_shape_mismatch_raises():
    base = np.zeros((4, 4, 3), dtype=np.float32)
    overlay = np.zeros((3, 4, 4), dtype=np.float32)
    raised = False
    try:
        composite_rgba_over(base, overlay)
    except ValueError:
        raised = True
    assert raised


def test_composite_rgba_over_clips_out_of_range():
    base = np.full((2, 2, 3), 2.0, dtype=np.float32)  # over-bright base
    overlay = np.zeros((2, 2, 4), dtype=np.float32)
    out = composite_rgba_over(base, overlay)
    assert float(out.max()) <= 1.0 + 1e-6
    assert float(out.min()) >= 0.0 - 1e-6


# ---------------------------------------------------------------------------
# render_frames — output contract (§25.4: output shape / range / batch length)
# ---------------------------------------------------------------------------

def test_output_shape_dtype_range():
    frames = _gray_frames(2, 24, 30, 0.5)
    plans = [[_item((5, 5, 25, 20), use_fill=True, style=_style(use_fill=True))], []]
    out = _render(frames, plans)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == (2, 24, 30, 3)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 - 1e-6
    assert float(out.max()) <= 1.0 + 1e-6


def test_batch_length_preserved_various_B():
    for b in (1, 3, 5):
        frames = _gray_frames(b, 16, 16, 0.5)
        plans = [[] for _ in range(b)]
        out = _render(frames, plans)
        assert out.shape[0] == b, f"batch {b} not preserved (got {out.shape[0]})"


def test_empty_batch_returns_wellformed_tensor():
    frames = torch.zeros((0, 5, 5, 3), dtype=torch.float32)
    out = _render(frames, [])
    assert tuple(out.shape) == (0, 0, 0, 3)
    assert out.dtype == torch.float32


def test_rgba_input_yields_rgb_output():
    frames = torch.full((1, 20, 20, 4), 0.3, dtype=torch.float32)
    frames[..., 3] = 1.0
    out = _render(frames, [[_item((2, 2, 18, 18))]])
    assert tuple(out.shape) == (1, 20, 20, 3)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# render_frames — no-detection passthrough (§25.4)
# ---------------------------------------------------------------------------

def test_empty_plan_frame_passthrough_identity():
    frames = torch.rand(3, 18, 22, 3, dtype=torch.float32)
    out = _render(frames, [[], [], []])
    assert torch.allclose(out, frames, atol=1e-6), "empty plans must pass frames through unchanged"


def test_fully_empty_plans_list_passthrough():
    # plans == [] (length 0) -> every frame is "missing" -> empty -> unchanged.
    frames = torch.rand(2, 12, 12, 3, dtype=torch.float32)
    out = _render(frames, [])
    assert torch.allclose(out, frames, atol=1e-6)


def test_mixed_empty_and_drawn_frames():
    frames = _gray_frames(3, 20, 20, 0.5)
    drawn = _item((2, 2, 18, 18), use_fill=True, style=_style(use_fill=True))
    out = _render(frames, [[], [drawn], []])
    # frame 0 and 2 unchanged, frame 1 changed.
    assert torch.allclose(out[0], frames[0], atol=1e-6)
    assert torch.allclose(out[2], frames[2], atol=1e-6)
    assert not torch.allclose(out[1], frames[1], atol=1e-6)


def test_plans_shorter_than_batch_remaining_empty():
    frames = _gray_frames(3, 16, 16, 0.5)
    drawn = _item((2, 2, 14, 14), use_fill=True, style=_style(use_fill=True))
    out = _render(frames, [[drawn]])  # only 1 plan for 3 frames
    assert out.shape[0] == 3
    assert not torch.allclose(out[0], frames[0], atol=1e-6)  # drawn
    assert torch.allclose(out[1], frames[1], atol=1e-6)      # missing -> empty
    assert torch.allclose(out[2], frames[2], atol=1e-6)


def test_plans_longer_than_batch_extra_ignored():
    frames = _gray_frames(1, 16, 16, 0.5)
    drawn = _item((2, 2, 14, 14), use_fill=True, style=_style(use_fill=True))
    # 3 plans for a 1-frame batch: extra plans are ignored, no error.
    out = _render(frames, [[drawn], [drawn], [drawn]])
    assert out.shape[0] == 1


# ---------------------------------------------------------------------------
# render_frames — never mutate input (contract)
# ---------------------------------------------------------------------------

def test_input_tensor_not_mutated():
    frames = _gray_frames(2, 20, 20, 0.5)
    before = frames.clone()
    _ = _render(frames, [[_item((3, 3, 17, 17), use_fill=True, style=_style(use_fill=True))], []])
    assert torch.equal(frames, before), "render_frames must not mutate the input tensor"


def test_output_not_aliasing_input_on_passthrough():
    # Even when a frame is passed through unchanged, the OUTPUT must be a distinct
    # buffer (so a later in-place op on the output cannot corrupt the input).
    frames = _gray_frames(1, 10, 10, 0.5)
    out = _render(frames, [[]])
    out[0, 0, 0, 0] = 0.123  # mutate output
    assert float(frames[0, 0, 0, 0]) == 0.5  # input untouched


# ---------------------------------------------------------------------------
# render_frames — placement (§25.4)
# ---------------------------------------------------------------------------

def test_placement_changes_pixels_inside_box():
    frames = _gray_frames(1, 40, 40, 0.5)
    box = (10, 10, 30, 30)
    out = _render(frames, [[_item(box, color=(1.0, 0.0, 0.0), use_fill=True,
                                  style=_style(use_fill=True))]])
    delta = (out[0] - frames[0]).abs().sum(dim=-1).numpy()
    # Significant change is concentrated inside (or on the border of) the box.
    inside = delta[12:28, 12:28]
    far_corner = delta[0:4, 0:4]  # well outside the box
    assert float(inside.sum()) > 0.0
    assert float(far_corner.sum()) == 0.0, "pixels far outside the box must be untouched"


def test_solid_fill_uses_track_color():
    frames = _gray_frames(1, 30, 30, 0.5)
    # Pure-red fill at full fill_opacity -> the red channel rises toward 1, the
    # green/blue channels fall toward 0 inside the box.
    params = dict(DEFAULT_PARAMS, fill_opacity=1.0)
    out = _render(frames, [[_item((5, 5, 25, 25), color=(1.0, 0.0, 0.0), use_fill=True,
                                  style=_style(use_fill=True))]], params=params)
    px = out[0, 15, 15]  # centre of the box
    assert float(px[0]) > 0.9   # red high
    assert float(px[1]) < 0.1   # green low
    assert float(px[2]) < 0.1   # blue low


# ---------------------------------------------------------------------------
# render_frames — fill opacity (§25.4)
# ---------------------------------------------------------------------------

def test_fill_opacity_scales_blend():
    frames = _gray_frames(1, 24, 24, 0.5)
    box = (4, 4, 20, 20)
    low = _render(frames, [[_item(box, color=(1.0, 1.0, 1.0), use_fill=True,
                                  style=_style(use_fill=True))]],
                  params=dict(DEFAULT_PARAMS, fill_opacity=0.2))
    high = _render(frames, [[_item(box, color=(1.0, 1.0, 1.0), use_fill=True,
                                   style=_style(use_fill=True))]],
                   params=dict(DEFAULT_PARAMS, fill_opacity=0.9))
    # White fill: higher opacity pushes the centre pixel closer to 1.0.
    c_low = float(low[0, 12, 12, 0])
    c_high = float(high[0, 12, 12, 0])
    assert c_high > c_low > 0.5, f"opacity should scale blend (low={c_low}, high={c_high})"
    # Expected values: 0.5*(1-a) + 1.0*a.
    assert abs(c_low - (0.5 * 0.8 + 0.2)) < 0.02
    assert abs(c_high - (0.5 * 0.1 + 0.9)) < 0.02


def test_fill_flash_boosts_fill():
    frames = _gray_frames(1, 24, 24, 0.5)
    box = (4, 4, 20, 20)
    params = dict(DEFAULT_PARAMS, fill_opacity=0.2)
    noflash = _render(frames, [[_item(box, color=(1.0, 1.0, 1.0), use_fill=True,
                                      style=_style(use_fill=True), fill_flash=False)]], params=params)
    flash = _render(frames, [[_item(box, color=(1.0, 1.0, 1.0), use_fill=True,
                                    style=_style(use_fill=True), fill_flash=True)]], params=params)
    assert float(flash[0, 12, 12, 0]) > float(noflash[0, 12, 12, 0])


# ---------------------------------------------------------------------------
# render_frames — per-item micro variation
# ---------------------------------------------------------------------------

def test_opacity_mul_zero_draws_nothing():
    frames = _gray_frames(1, 20, 20, 0.5)
    out = _render(frames, [[_item((2, 2, 18, 18), use_fill=True,
                                  style=_style(use_fill=True), opacity_mul=0.0)]])
    assert torch.allclose(out[0], frames[0], atol=1e-6), "opacity_mul=0 must draw nothing"


def test_opacity_mul_partial_is_lighter():
    frames = _gray_frames(1, 24, 24, 0.5)
    box = (4, 4, 20, 20)
    full = _render(frames, [[_item(box, color=(1.0, 1.0, 1.0), use_fill=True,
                                   style=_style(use_fill=True), opacity_mul=1.0)]],
                   params=dict(DEFAULT_PARAMS, fill_opacity=1.0))
    half = _render(frames, [[_item(box, color=(1.0, 1.0, 1.0), use_fill=True,
                                   style=_style(use_fill=True), opacity_mul=0.5)]],
                   params=dict(DEFAULT_PARAMS, fill_opacity=1.0))
    # A partial opacity_mul leaves the centre between base (0.5) and full white.
    c_full = float(full[0, 12, 12, 0])
    c_half = float(half[0, 12, 12, 0])
    assert c_full > c_half > 0.5


def test_line_dropout_suppresses_box():
    frames = _gray_frames(1, 30, 30, 0.5)
    box = (5, 5, 25, 25)
    drawn = _render(frames, [[_item(box, style=_style(primitive_style="corners"),
                                    line_dropout=False)]])
    dropped = _render(frames, [[_item(box, style=_style(primitive_style="corners"),
                                      line_dropout=True)]])
    # With no fill/target/caption, line_dropout removes the only drawn element.
    assert not torch.allclose(drawn[0], frames[0], atol=1e-6)
    assert torch.allclose(dropped[0], frames[0], atol=1e-6)


def test_jitter_shifts_overlay():
    frames = _gray_frames(1, 40, 40, 0.5)
    box = (10, 10, 30, 30)
    base = _render(frames, [[_item(box, use_fill=True, style=_style(use_fill=True),
                                   jitter=(0, 0))]])
    shifted = _render(frames, [[_item(box, use_fill=True, style=_style(use_fill=True),
                                      jitter=(6, 0))]])
    # The two overlays differ where the box moved.
    assert not torch.allclose(base[0], shifted[0], atol=1e-4)


# ---------------------------------------------------------------------------
# render_frames — edge clipping (§25.4)
# ---------------------------------------------------------------------------

def test_box_off_right_edge_no_overflow():
    frames = _gray_frames(1, 30, 30, 0.5)
    # Box extends past the right/bottom edge; must clip, never raise, never paint
    # outside the frame (the output is always [H,W,3] and clamped 0..1).
    out = _render(frames, [[_item((20, 20, 60, 60), use_fill=True,
                                  style=_style(use_fill=True))]])
    assert tuple(out.shape) == (1, 30, 30, 3)
    assert float(out.max()) <= 1.0 + 1e-6 and float(out.min()) >= -1e-6


def test_box_fully_off_frame_is_safe():
    frames = _gray_frames(1, 20, 20, 0.5)
    # A box entirely off-frame (negative) -> clamped to nothing -> passthrough.
    out = _render(frames, [[_item((-50, -50, -10, -10), use_fill=True,
                                  style=_style(use_fill=True))]])
    assert tuple(out.shape) == (1, 20, 20, 3)
    assert torch.allclose(out[0], frames[0], atol=1e-6)


def test_jitter_pushing_box_off_edge_no_overflow():
    frames = _gray_frames(1, 24, 24, 0.5)
    out = _render(frames, [[_item((2, 2, 20, 20), style=_style(primitive_style="corners"),
                                  jitter=(-100, -100))]])
    assert tuple(out.shape) == (1, 24, 24, 3)
    assert float(out.max()) <= 1.0 + 1e-6 and float(out.min()) >= -1e-6


def test_crosshair_near_edge_no_raise():
    frames = _gray_frames(1, 20, 20, 0.5)
    # target_only item whose centre is near the corner -> some arms clip off.
    style = _style(primitive_style="target_only", use_target=True)
    out = _render(frames, [[_item((0, 0, 6, 6), style=style)]],
                  params=dict(DEFAULT_PARAMS, target_scale=0.9))
    assert tuple(out.shape) == (1, 20, 20, 3)
    assert float(out.max()) <= 1.0 + 1e-6


def test_two_items_one_offframe_one_on():
    # Mirrors the original skeleton case: one off-top-left box + one off-bottom
    # -right box on a small frame must not crash and must stay in-range.
    frames = _gray_frames(1, 80, 80, 0.5)
    plans = [[
        _item((-30, -30, 50, 50), style=_style(primitive_style="corners")),
        _item((60, 60, 200, 200), use_fill=True, style=_style(use_fill=True)),
    ]]
    out = _render(frames, plans)
    assert tuple(out.shape) == (1, 80, 80, 3)
    assert float(out.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# render_frames — label flip (§25.4)
# ---------------------------------------------------------------------------

def test_label_top_hugging_box_stays_on_frame():
    # A caption with preferred side 'top' on a box hugging the TOP edge must be
    # relocated (flipped to bottom / clamped) so it is never clipped off-frame.
    frames = _gray_frames(1, 60, 120, 0.5)
    style = _style(use_caption=True, caption_side="top")
    item = _item((10, 0, 90, 20), color=(0.1, 0.1, 0.9), label="01. TOPLABEL", style=style)
    out = _render(frames, [[item]])
    delta = (out[0] - frames[0]).abs().sum(dim=-1).numpy()
    # The label drew *somewhere* (caption present) and stayed inside the frame.
    assert float(delta.sum()) > 0.0
    # Since the box hugs the top, the flipped strip should appear below the box.
    below = delta[20:40, :]
    assert float(below.sum()) > 0.0, "flipped caption strip should appear below the top-hugging box"


def test_label_bottom_hugging_box_stays_on_frame():
    frames = _gray_frames(1, 60, 120, 0.5)
    style = _style(use_caption=True, caption_side="bottom")
    # Box hugging the BOTTOM edge -> 'bottom' preference must flip up to stay on-frame.
    item = _item((10, 40, 90, 60), color=(0.1, 0.1, 0.9), label="01. BOTLABEL", style=style)
    out = _render(frames, [[item]])
    assert tuple(out.shape) == (1, 60, 120, 3)
    delta = (out[0] - frames[0]).abs().sum(dim=-1).numpy()
    assert float(delta.sum()) > 0.0
    # Output shape fixed at 60 rows -> by construction nothing was painted off-frame.


def test_long_label_near_top_no_crash():
    # A very long caption near the top edge must truncate / fit, never crash.
    frames = _gray_frames(1, 120, 200, 0.5)
    style = _style(use_caption=True, caption_side="top")
    item = _item((10, 2, 120, 40), label="01. PERSON WEARING A BLUE JACKET HOLDING A BALL",
                 style=style)
    out = _render(frames, [[item]])
    assert tuple(out.shape) == (1, 120, 200, 3)
    assert float(out.max()) <= 1.0 + 1e-6


def test_empty_label_draws_no_strip():
    # use_caption True but empty label_text -> no strip (only the box).
    frames = _gray_frames(1, 30, 30, 0.5)
    style = _style(use_caption=True, primitive_style="corners")
    with_label = _render(frames, [[_item((4, 8, 26, 24), label="HELLO", style=style)]])
    no_label = _render(frames, [[_item((4, 8, 26, 24), label="", style=style)]])
    # The labelled render changes strictly more pixels than the unlabelled one.
    d_with = float((with_label[0] - frames[0]).abs().sum())
    d_without = float((no_label[0] - frames[0]).abs().sum())
    assert d_with > d_without


# ---------------------------------------------------------------------------
# render_frames — asset rendering paths
# ---------------------------------------------------------------------------

def test_frame_asset_rendered_and_recolored():
    frames = _gray_frames(1, 40, 48, 0.3)
    assets = _asset_bank_with_one_each()
    assert assets.n_frame_assets == 1
    style = _style(primitive_style="frame_asset", frame_asset_id=0)
    item = _item((8, 8, 40, 32), color=(0.0, 1.0, 0.0), style=style)  # green tint
    out = _render(frames, [[item]], assets=assets)
    assert tuple(out.shape) == (1, 40, 48, 3)
    # The recolored white asset becomes green -> green channel hits ~1 somewhere.
    assert float(out[0, ..., 1].max()) > 0.9
    # And the asset never escapes the frame / range.
    assert float(out.max()) <= 1.0 + 1e-6 and float(out.min()) >= -1e-6


def test_target_asset_rendered():
    frames = _gray_frames(1, 40, 40, 0.3)
    assets = _asset_bank_with_one_each()
    assert assets.n_target_assets == 1
    style = _style(primitive_style="target_only", use_target=True, target_asset_id=0)
    item = _item((10, 10, 30, 30), color=(0.0, 0.0, 1.0), style=style)  # blue reticle
    out = _render(frames, [[item]], assets=assets, params=dict(DEFAULT_PARAMS, target_scale=0.5))
    assert float(out[0, ..., 2].max()) > 0.9  # blue present
    assert float(out.max()) <= 1.0 + 1e-6


def test_mask_only_frame_asset_falls_back_to_corners():
    # A mask-only frame slot is eligible but renders to None (brief §16); a track
    # assigned 'frame_asset' on it must degrade to corner brackets, not vanish.
    _img, mask = _white_on_transparent_pair()
    assets = AssetBank(
        [(None, mask)] + [None] * 5,  # mask connected, image not
        [None] * 2,
        trim=True,
        fit_frame="stretch",
        fit_target="contain",
        max_cache_items=16,
    )
    assert assets.n_frame_assets == 1
    assert assets.frame_rgba(0, 16, 16, (1.0, 0.0, 0.0), 1.0) is None  # not rendered
    frames = _gray_frames(1, 30, 30, 0.5)
    style = _style(primitive_style="frame_asset", frame_asset_id=0)
    out = _render(frames, [[_item((5, 5, 25, 25), color=(1.0, 0.0, 0.0), style=style)]], assets=assets)
    # Something WAS drawn (the corner-bracket fallback).
    assert not torch.allclose(out[0], frames[0], atol=1e-6)


def test_none_assets_argument_uses_procedural_only():
    # assets=None must not crash; a frame_asset style degrades to procedural.
    frames = _gray_frames(1, 30, 30, 0.5)
    style = _style(primitive_style="frame_asset", frame_asset_id=0)
    out = render_frames(
        frames,
        [[_item((5, 5, 25, 25), color=(1.0, 0.0, 0.0), style=style)]],
        assets=None,
        palette=PALETTE,
        params=DEFAULT_PARAMS,
        font_path=None,
    )
    assert tuple(out.shape) == (1, 30, 30, 3)
    assert not torch.allclose(out[0], frames[0], atol=1e-6)  # corner fallback drew


# ---------------------------------------------------------------------------
# render_frames — progress + interrupt + determinism
# ---------------------------------------------------------------------------

def test_progress_called_once_per_frame():
    frames = _gray_frames(4, 12, 12, 0.5)
    prog = _Progress()
    _ = _render(frames, [[], [], [], []], progress=prog)
    assert prog.calls == 4
    assert prog.total == 4


def test_progress_error_does_not_break_render():
    class BadProgress:
        def update(self, n):
            raise RuntimeError("progress backend exploded")

    frames = _gray_frames(2, 10, 10, 0.5)
    out = _render(frames, [[], []], progress=BadProgress())
    assert tuple(out.shape) == (2, 10, 10, 3)  # render still completed


def test_interrupt_propagates():
    class _Stop(Exception):
        pass

    def boom():
        raise _Stop()

    frames = _gray_frames(2, 10, 10, 0.5)
    raised = False
    try:
        _render(frames, [[], []], interrupt=boom)
    except _Stop:
        raised = True
    assert raised, "interrupt() raising must propagate out of render_frames"


def test_interrupt_checked_before_first_frame():
    # An interrupt that fires immediately should be checked before frame 0.
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("stop")

    frames = _gray_frames(3, 8, 8, 0.5)
    try:
        _render(frames, [[], [], []], interrupt=boom)
    except RuntimeError:
        pass
    assert calls["n"] >= 1


def test_deterministic_same_inputs_same_output():
    frames = _gray_frames(2, 24, 24, 0.5)
    style = _style(use_fill=True, use_caption=True, primitive_style="outline", glitch_phase=0.5)
    plans = [[_item((4, 6, 20, 22), color=(1.0, 0.2, 0.1), label="01. PERSON",
                    style=style, jitter=(1, -1))], []]
    a = _render(frames, plans)
    b = _render(frames, plans)
    assert torch.equal(a, b), "render must be deterministic for identical inputs"


def test_none_frames_raises_type_error():
    raised = False
    try:
        _render(None, [[]])
    except TypeError:
        raised = True
    assert raised


def test_missing_params_uses_safe_defaults():
    # params={} must not crash; defaults apply.
    frames = _gray_frames(1, 20, 20, 0.5)
    out = render_frames(
        frames,
        [[_item((3, 3, 17, 17), use_fill=True, style=_style(use_fill=True))]],
        assets=_empty_assets(),
        palette=PALETTE,
        params={},
        font_path=None,
    )
    assert tuple(out.shape) == (1, 20, 20, 3)


def test_item_with_show_false_style_still_drawn_if_present():
    # The compositor draws whatever RenderItems it is given; show-gating happens
    # UPSTREAM (renderer node), so an item present in the plan IS drawn even if
    # its style.show is False. This documents the boundary precisely.
    frames = _gray_frames(1, 20, 20, 0.5)
    style = _style(use_fill=True, show=False)
    out = _render(frames, [[_item((3, 3, 17, 17), color=(1.0, 1.0, 1.0), style=style)]],
                  params=dict(DEFAULT_PARAMS, fill_opacity=1.0))
    assert not torch.allclose(out[0], frames[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Script runner (works without pytest)
# ---------------------------------------------------------------------------

def _run_all() -> int:
    g = globals()
    tests = sorted(name for name in g if name.startswith("test_") and callable(g[name]))
    passed = 0
    failed = 0
    failures = []
    for name in tests:
        try:
            g[name]()
            passed += 1
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 — test harness needs broad catch
            failed += 1
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"compositor tests: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFAILURES:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
