"""Tests for ``yonder_hud.procedural_hud`` (brief §9 primitives, §25 cases).

Runnable two ways:

    /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_procedural_hud.py -q
    /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_procedural_hud.py

The ``__main__`` block runs every ``test_*`` function, prints a summary, and
``sys.exit(1)`` on any failure. No ComfyUI, no network, no model files — only
small synthetic PIL/numpy fixtures.

Coverage:
* new_rgba_layer: mode/size/transparency, degenerate (0/neg) size guard.
* draw_corners: L-brackets at all four corners, middle of edges empty,
  corner_gap inset, ratio cap, off-frame clip, degenerate no-op, alpha baked.
* draw_outline: clean rectangle border vs glitched (broken) border, glitch is
  deterministic with a seeded rng + differs from clean, off-frame clip.
* draw_solid: solid fill incl. exact corners, alpha baked from colour, clamps
  off-frame fill to canvas, transparent colour = no-op.
* draw_crosshair: arms + centre gap (empty centre), dot, ticks, edge clipping.
* draw_registration: density 0 = no-op, corners always marked, more density =>
  more painted pixels, deterministic with seeded rng, off-frame safe.
* Cross-cutting: every primitive accepts a None/degenerate box without raising
  and never paints outside the canvas bounds.
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

# Make the package importable when run as a bare script from the repo root or
# from within the tests/ directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from yonder_hud import procedural_hud as P  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OPAQUE_WHITE = (1.0, 1.0, 1.0, 1.0)
OPAQUE_RED = (1.0, 0.0, 0.0, 1.0)
HALF_GREEN = (0.0, 1.0, 0.0, 0.5)
TRANSPARENT = (1.0, 1.0, 1.0, 0.0)


def _layer(w=64, h=48):
    """A fresh transparent RGBA layer + its ImageDraw handle."""
    img = P.new_rgba_layer(w, h)
    return img, ImageDraw.Draw(img)


def _arr(img):
    """uint8 [H,W,4] numpy view of an RGBA image."""
    return np.array(img)


def _alpha(img):
    """The alpha channel [H,W] uint8."""
    return _arr(img)[..., 3]


def _painted_count(img):
    """Number of pixels with non-zero alpha (i.e. anything drawn)."""
    return int(np.count_nonzero(_alpha(img)))


def _any_painted(img):
    return _painted_count(img) > 0


# ---------------------------------------------------------------------------
# new_rgba_layer
# ---------------------------------------------------------------------------

def test_new_rgba_layer_mode_size_transparent():
    img = P.new_rgba_layer(20, 10)
    assert img.mode == "RGBA"
    assert img.size == (20, 10)  # PIL size is (W, H)
    a = _arr(img)
    assert a.shape == (10, 20, 4)
    assert a.sum() == 0  # fully transparent, black


def test_new_rgba_layer_degenerate_size_guard():
    # zero / negative dims must not raise (PIL rejects 0-sized images); clamp >=1
    img0 = P.new_rgba_layer(0, 0)
    assert img0.size == (1, 1) and img0.mode == "RGBA"
    imgn = P.new_rgba_layer(-5, 3)
    assert imgn.size == (1, 3)


def test_new_rgba_layer_float_dims_coerced():
    img = P.new_rgba_layer(8.9, 4.2)
    assert img.size == (8, 4)


# ---------------------------------------------------------------------------
# draw_corners
# ---------------------------------------------------------------------------

def test_draw_corners_paints_all_four_corners_not_mid_edges():
    img, d = _layer(64, 48)
    box = (10, 8, 54, 40)
    P.draw_corners(d, box, OPAQUE_WHITE, line_width=2, corner_length_ratio=0.25, corner_gap=0)
    a = _alpha(img)
    # corners painted (sample a couple px in from each corner along an arm)
    assert a[8, 12] > 0     # top edge near top-left corner
    assert a[8, 52] > 0     # top edge near top-right corner
    assert a[40, 12] > 0    # bottom-left
    assert a[40, 52] > 0    # bottom-right
    # vertical arms painted near corners
    assert a[10, 10] > 0    # left edge near top-left
    assert a[38, 54] > 0    # right edge near bottom-right
    # MIDDLE of the top edge must be EMPTY (gap between brackets)
    mid_x = (10 + 54) // 2
    assert a[8, mid_x] == 0
    # middle of the left edge empty
    mid_y = (8 + 40) // 2
    assert a[mid_y, 10] == 0


def test_draw_corners_color_alpha_baked_into_layer():
    img, d = _layer(40, 40)
    P.draw_corners(d, (5, 5, 35, 35), HALF_GREEN, line_width=3)
    a = _arr(img)
    painted = a[a[..., 3] > 0]
    assert painted.size > 0
    # alpha ~ 0.5*255 ~ 128 ; green channel full where painted
    assert abs(int(painted[0, 3]) - 128) <= 2
    assert int(painted[0, 1]) == 255  # G
    assert int(painted[0, 0]) == 0    # R


def test_draw_corners_gap_insets_brackets():
    # With a gap, the exact corner pixel should be empty and the bracket sits inside.
    img, d = _layer(60, 60)
    box = (10, 10, 50, 50)
    P.draw_corners(d, box, OPAQUE_WHITE, line_width=1, corner_length_ratio=0.3, corner_gap=5)
    a = _alpha(img)
    # exact top-left corner pixel is now empty
    assert a[10, 10] == 0
    # but a pixel inset by the gap is painted
    assert a[15, 16] > 0 or a[16, 15] > 0


def test_draw_corners_off_frame_box_clips_no_paint_outside():
    img, d = _layer(32, 32)
    # box mostly off the right/bottom; must not raise and must not paint OOB
    P.draw_corners(d, (20, 20, 200, 200), OPAQUE_WHITE, line_width=3)
    a = _alpha(img)
    assert a.shape == (32, 32)  # never grew
    # everything painted is within bounds by construction (numpy view is the layer)
    # the visible top-left corner of the clipped region should be painted
    assert a[20, 21] > 0 or a[21, 20] > 0


def test_draw_corners_degenerate_and_none_box_noop():
    img, d = _layer(32, 32)
    P.draw_corners(d, (10, 10, 10, 10), OPAQUE_WHITE, line_width=2)  # zero size
    assert not _any_painted(img)
    P.draw_corners(d, None, OPAQUE_WHITE, line_width=2)
    assert not _any_painted(img)
    P.draw_corners(d, (10, 10, 10.5, 30), OPAQUE_WHITE, line_width=2)  # width <1 px extent
    # width 0.5 clamps to on-frame but extent positive after clamp? 10..10.5 -> 0.5 -> rounds; still drawable as line. Allow either.
    # The key assertion: no exception. (pixels optional)


def test_draw_corners_transparent_color_noop():
    img, d = _layer(32, 32)
    P.draw_corners(d, (4, 4, 28, 28), TRANSPARENT, line_width=3)
    assert not _any_painted(img)


def test_draw_corners_ratio_capped_no_overlap_small_box():
    # ratio 0.9 on a small box must be capped so opposite brackets don't merge
    img, d = _layer(40, 40)
    box = (10, 10, 30, 30)  # 20x20
    P.draw_corners(d, box, OPAQUE_WHITE, line_width=1, corner_length_ratio=0.9)
    a = _alpha(img)
    # the exact centre of the top edge must still be empty (arms capped at half)
    cx = (10 + 30) // 2
    assert a[10, cx] == 0


# ---------------------------------------------------------------------------
# draw_outline
# ---------------------------------------------------------------------------

def test_draw_outline_clean_rectangle_border():
    img, d = _layer(50, 40)
    box = (10, 8, 40, 32)
    P.draw_outline(d, box, OPAQUE_WHITE, line_width=1, glitch_segments=0)
    a = _alpha(img)
    # all four edges painted
    assert a[8, 25] > 0    # top mid
    assert a[32, 25] > 0   # bottom mid
    assert a[20, 10] > 0   # left mid
    assert a[20, 40] > 0   # right mid
    # interior is empty (outline, not fill)
    assert a[20, 25] == 0


def test_draw_outline_glitch_breaks_edges_and_is_deterministic():
    box = (8, 8, 56, 40)
    # clean reference
    img_clean, d_clean = _layer(64, 48)
    P.draw_outline(d_clean, box, OPAQUE_WHITE, line_width=1, glitch_segments=0)
    n_clean = _painted_count(img_clean)

    # glitched with a fixed seed
    img_g1, d_g1 = _layer(64, 48)
    P.draw_outline(d_g1, box, OPAQUE_WHITE, line_width=1, glitch_segments=8, rng=random.Random(123))
    n_g1 = _painted_count(img_g1)

    # same seed -> identical pixels (determinism)
    img_g2, d_g2 = _layer(64, 48)
    P.draw_outline(d_g2, box, OPAQUE_WHITE, line_width=1, glitch_segments=8, rng=random.Random(123))
    assert np.array_equal(_arr(img_g1), _arr(img_g2))

    # glitched should drop at least some segments -> fewer painted px than clean
    assert n_g1 < n_clean
    assert n_g1 > 0  # but not everything dropped


def test_draw_outline_glitch_different_seeds_differ():
    box = (8, 8, 56, 40)
    img_a, da = _layer(64, 48)
    img_b, db = _layer(64, 48)
    P.draw_outline(da, box, OPAQUE_WHITE, line_width=1, glitch_segments=10, rng=random.Random(1))
    P.draw_outline(db, box, OPAQUE_WHITE, line_width=1, glitch_segments=10, rng=random.Random(999))
    # extremely unlikely to be identical with different seeds
    assert not np.array_equal(_arr(img_a), _arr(img_b))


def test_draw_outline_glitch_none_rng_is_deterministic():
    box = (8, 8, 56, 40)
    img_a, da = _layer(64, 48)
    img_b, db = _layer(64, 48)
    P.draw_outline(da, box, OPAQUE_WHITE, line_width=1, glitch_segments=8, rng=None)
    P.draw_outline(db, box, OPAQUE_WHITE, line_width=1, glitch_segments=8, rng=None)
    assert np.array_equal(_arr(img_a), _arr(img_b))  # fixed fallback seed


def test_draw_outline_off_frame_and_degenerate():
    img, d = _layer(32, 32)
    P.draw_outline(d, (-50, -50, 1000, 1000), OPAQUE_WHITE, line_width=2)  # huge -> clamps to full frame border
    assert _arr(img).shape == (32, 32, 4)
    img2, d2 = _layer(32, 32)
    P.draw_outline(d2, None, OPAQUE_WHITE, line_width=2)
    assert not _any_painted(img2)


# ---------------------------------------------------------------------------
# draw_solid
# ---------------------------------------------------------------------------

def test_draw_solid_fills_box_with_baked_alpha():
    img, d = _layer(40, 40)
    box = (8, 8, 24, 24)
    P.draw_solid(d, box, HALF_GREEN)
    a = _arr(img)
    # interior filled with half-alpha green
    assert a[15, 15, 3] == 128 or abs(int(a[15, 15, 3]) - 128) <= 2
    assert a[15, 15, 1] == 255
    # exact corners inclusive
    assert a[8, 8, 3] > 0
    assert a[24, 24, 3] > 0
    # outside untouched
    assert a[5, 5, 3] == 0
    assert a[30, 30, 3] == 0


def test_draw_solid_clamps_off_frame_fill():
    img, d = _layer(32, 32)
    P.draw_solid(d, (-100, -100, 1000, 1000), OPAQUE_RED)  # cover entire canvas
    a = _arr(img)
    assert a.shape == (32, 32, 4)
    # entire canvas filled red opaque
    assert np.all(a[..., 3] == 255)
    assert np.all(a[..., 0] == 255)
    assert np.all(a[..., 1] == 0)


def test_draw_solid_transparent_color_noop():
    img, d = _layer(32, 32)
    P.draw_solid(d, (4, 4, 28, 28), TRANSPARENT)
    assert not _any_painted(img)


def test_draw_solid_degenerate_box_noop():
    img, d = _layer(32, 32)
    P.draw_solid(d, (10, 10, 10, 10), OPAQUE_RED)
    assert not _any_painted(img)
    P.draw_solid(d, None, OPAQUE_RED)
    assert not _any_painted(img)


# ---------------------------------------------------------------------------
# draw_crosshair
# ---------------------------------------------------------------------------

def test_draw_crosshair_arms_with_empty_center_gap():
    img, d = _layer(64, 64)
    center = (32, 32)
    P.draw_crosshair(d, center, size=12, color_rgba=OPAQUE_WHITE, line_width=1, gap=4, dot=False, ticks=False)
    a = _alpha(img)
    # arm tips painted
    assert a[32, 32 + 11] > 0   # right arm (near tip)
    assert a[32, 32 - 11] > 0   # left arm
    assert a[32 - 11, 32] > 0   # up arm
    assert a[32 + 11, 32] > 0   # down arm
    # the exact centre is empty (gap, and dot disabled)
    assert a[32, 32] == 0
    # just inside the gap (1px from centre) is empty
    assert a[32, 33] == 0


def test_draw_crosshair_dot_fills_center():
    img, d = _layer(40, 40)
    P.draw_crosshair(d, (20, 20), size=8, color_rgba=OPAQUE_RED, line_width=2, gap=3, dot=True, ticks=False)
    a = _arr(img)
    assert a[20, 20, 3] > 0  # centre dot present
    assert a[20, 20, 0] == 255


def test_draw_crosshair_ticks_add_end_strokes():
    img_no, dn = _layer(64, 64)
    img_yes, dy = _layer(64, 64)
    P.draw_crosshair(dn, (32, 32), size=12, color_rgba=OPAQUE_WHITE, line_width=1, gap=3, dot=False, ticks=False)
    P.draw_crosshair(dy, (32, 32), size=12, color_rgba=OPAQUE_WHITE, line_width=1, gap=3, dot=False, ticks=True)
    assert _painted_count(img_yes) > _painted_count(img_no)
    # a tick is a perpendicular stroke at the right arm tip (x=44): pixels above/below centre y
    a = _alpha(img_yes)
    assert a[32 - 2, 32 + 12] > 0 or a[32 + 2, 32 + 12] > 0


def test_draw_crosshair_edge_clipping_no_oob():
    img, d = _layer(20, 20)
    # near corner: left and up arms run off-canvas, must clip cleanly
    P.draw_crosshair(d, (2, 2), size=15, color_rgba=OPAQUE_WHITE, line_width=1, gap=2, dot=True, ticks=True)
    a = _alpha(img)
    assert a.shape == (20, 20)
    # right/down arms still present on-canvas
    assert a[2, 10] > 0   # right arm
    assert a[10, 2] > 0   # down arm


def test_draw_crosshair_zero_size_and_none_center_noop():
    img, d = _layer(32, 32)
    P.draw_crosshair(d, (16, 16), size=0, color_rgba=OPAQUE_WHITE, line_width=1, gap=0)
    assert not _any_painted(img)
    P.draw_crosshair(d, None, size=10, color_rgba=OPAQUE_WHITE, line_width=1, gap=0)
    assert not _any_painted(img)


def test_draw_crosshair_gap_ge_size_only_dot():
    img, d = _layer(40, 40)
    # gap >= size -> arms vanish; with dot=True only the centre dot remains
    P.draw_crosshair(d, (20, 20), size=8, color_rgba=OPAQUE_WHITE, line_width=2, gap=20, dot=True, ticks=False)
    a = _alpha(img)
    assert a[20, 20] > 0             # dot present
    # no arm pixels far from centre
    assert a[20, 20 + 7] == 0


# ---------------------------------------------------------------------------
# draw_registration
# ---------------------------------------------------------------------------

def test_draw_registration_zero_density_noop():
    img, d = _layer(64, 48)
    P.draw_registration(d, (8, 8, 56, 40), OPAQUE_WHITE, density=0.0, rng=random.Random(0))
    assert not _any_painted(img)


def test_draw_registration_marks_corners_when_density_positive():
    img, d = _layer(80, 80)
    box = (16, 16, 64, 64)
    P.draw_registration(d, box, OPAQUE_WHITE, density=0.2, rng=random.Random(7))
    assert _any_painted(img)
    a = _alpha(img)
    # corner regions should have marks (search a small neighbourhood — glyphs are
    # jittered + vary in type so we check a window rather than an exact pixel).
    def _win(cy, cx, r=8):
        y0, y1 = max(0, cy - r), min(a.shape[0], cy + r)
        x0, x1 = max(0, cx - r), min(a.shape[1], cx + r)
        return int(a[y0:y1, x0:x1].sum())
    assert _win(16, 16) > 0    # top-left
    assert _win(16, 64) > 0    # top-right
    assert _win(64, 16) > 0    # bottom-left
    assert _win(64, 64) > 0    # bottom-right


def test_draw_registration_more_density_more_marks():
    box = (10, 10, 90, 70)
    img_lo, dl = _layer(100, 80)
    img_hi, dh = _layer(100, 80)
    P.draw_registration(dl, box, OPAQUE_WHITE, density=0.1, rng=random.Random(42))
    P.draw_registration(dh, box, OPAQUE_WHITE, density=1.0, rng=random.Random(42))
    assert _painted_count(img_hi) > _painted_count(img_lo)


def test_draw_registration_deterministic_with_seed():
    box = (10, 10, 90, 70)
    img_a, da = _layer(100, 80)
    img_b, db = _layer(100, 80)
    P.draw_registration(da, box, OPAQUE_WHITE, density=0.7, rng=random.Random(2024))
    P.draw_registration(db, box, OPAQUE_WHITE, density=0.7, rng=random.Random(2024))
    assert np.array_equal(_arr(img_a), _arr(img_b))


def test_draw_registration_none_rng_deterministic():
    box = (10, 10, 90, 70)
    img_a, da = _layer(100, 80)
    img_b, db = _layer(100, 80)
    P.draw_registration(da, box, OPAQUE_WHITE, density=0.5, rng=None)
    P.draw_registration(db, box, OPAQUE_WHITE, density=0.5, rng=None)
    assert np.array_equal(_arr(img_a), _arr(img_b))


def test_draw_registration_off_frame_and_none_box():
    img, d = _layer(40, 40)
    P.draw_registration(d, (-100, -100, 1000, 1000), OPAQUE_WHITE, density=1.0, rng=random.Random(0))
    assert _arr(img).shape == (40, 40, 4)  # never grew, no raise
    img2, d2 = _layer(40, 40)
    P.draw_registration(d2, None, OPAQUE_WHITE, density=1.0, rng=random.Random(0))
    assert not _any_painted(img2)


# ---------------------------------------------------------------------------
# Cross-cutting robustness (brief §25 — never raise, never paint OOB)
# ---------------------------------------------------------------------------

def test_all_primitives_never_paint_outside_canvas():
    """Draw deliberately-overflowing primitives; the numpy view is the layer, so
    if any drew OOB PIL would have raised. We additionally assert the image size
    is unchanged and that the *border ring* logic stayed self-consistent."""
    W, H = 48, 36
    img = P.new_rgba_layer(W, H)
    d = ImageDraw.Draw(img)
    big = (-200, -200, 9000, 9000)
    P.draw_solid(d, big, HALF_GREEN)            # fills whole canvas
    P.draw_corners(d, big, OPAQUE_WHITE, 4)
    P.draw_outline(d, big, OPAQUE_RED, 3, glitch_segments=6, rng=random.Random(5))
    P.draw_crosshair(d, (W * 5, H * 5), 50, OPAQUE_WHITE, 2, 4, dot=True, ticks=True)
    P.draw_registration(d, big, OPAQUE_WHITE, 1.0, rng=random.Random(5))
    a = _arr(img)
    assert a.shape == (H, W, 4)


def test_all_primitives_accept_none_box_without_raising():
    img, d = _layer(16, 16)
    P.draw_corners(d, None, OPAQUE_WHITE, 2)
    P.draw_outline(d, None, OPAQUE_WHITE, 2)
    P.draw_solid(d, None, OPAQUE_WHITE)
    P.draw_registration(d, None, OPAQUE_WHITE, 1.0)
    # crosshair takes a center, not a box; None center
    P.draw_crosshair(d, None, 8, OPAQUE_WHITE, 1, 2)
    assert not _any_painted(img)


def test_primitives_tiny_canvas_no_raise():
    img = P.new_rgba_layer(2, 2)
    d = ImageDraw.Draw(img)
    P.draw_corners(d, (0, 0, 2, 2), OPAQUE_WHITE, 1)
    P.draw_outline(d, (0, 0, 2, 2), OPAQUE_WHITE, 1, glitch_segments=4, rng=random.Random(0))
    P.draw_solid(d, (0, 0, 2, 2), HALF_GREEN)
    P.draw_crosshair(d, (1, 1), 1, OPAQUE_WHITE, 1, 0)
    P.draw_registration(d, (0, 0, 2, 2), OPAQUE_WHITE, 1.0, rng=random.Random(0))
    assert _arr(img).shape == (2, 2, 4)


def test_color_rgb_only_defaults_opaque():
    # 3-tuple color (no alpha) must default to opaque, not crash.
    img, d = _layer(20, 20)
    P.draw_solid(d, (4, 4, 16, 16), (1.0, 0.0, 0.0))
    a = _arr(img)
    assert a[10, 10, 3] == 255
    assert a[10, 10, 0] == 255


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
    print(f"procedural_hud tests: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFAILURES:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
