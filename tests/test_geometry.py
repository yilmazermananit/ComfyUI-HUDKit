"""Unit tests for yea_hud.geometry.

Runnable BOTH ways:

* pytest:   python -m pytest tests/test_geometry.py -q
* script:   python tests/test_geometry.py     (runs every test, prints a
            summary, sys.exit(1) on any failure)

No ComfyUI, no network, no model files. Tiny synthetic numpy/torch fixtures
only (and most cases are plain python tuples — geometry is pure math).
"""

from __future__ import annotations

import math
import os
import sys

# Make the package importable regardless of CWD when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402  (synthetic-fixture dtype coverage)
import torch  # noqa: E402  (verify 0-d tensor scalars are accepted)

from yea_hud import geometry as G  # noqa: E402
from yea_hud.types import Box  # noqa: E402  (alias identity check)


EPS = 1e-6


def _approx(a, b, eps=EPS):
    return abs(float(a) - float(b)) <= eps


def _box_approx(a, b, eps=EPS):
    return len(a) == 4 == len(b) and all(_approx(x, y, eps) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Conversions + round-trip
# ---------------------------------------------------------------------------

def test_xywh_to_xyxy_basic():
    assert _box_approx(G.xywh_to_xyxy(10, 20, 30, 40), (10, 20, 40, 60))


def test_xyxy_to_xywh_basic():
    assert _box_approx(G.xyxy_to_xywh(10, 20, 40, 60), (10, 20, 30, 40))


def test_conversion_roundtrip():
    box = (12.5, 7.25, 88.0, 130.75)
    x, y, w, h = G.xyxy_to_xywh(*box)
    assert _box_approx(G.xywh_to_xyxy(x, y, w, h), box)
    # and back the other way
    xywh = (3.0, 4.0, 50.0, 22.0)
    xyxy = G.xywh_to_xyxy(*xywh)
    assert _box_approx(G.xyxy_to_xywh(*xyxy), xywh)


def test_xywh_negative_extent_is_tolerated():
    # negative w/h should NOT be silently normalized by the converter
    assert _box_approx(G.xywh_to_xyxy(10, 10, -5, -8), (10, 10, 5, 2))


# ---------------------------------------------------------------------------
# normalize / clamp / degenerate
# ---------------------------------------------------------------------------

def test_normalize_swaps_reversed():
    assert _box_approx(G.normalize_box(40, 60, 10, 20), (10, 20, 40, 60))
    assert _box_approx(G.normalize_box(10, 60, 40, 20), (10, 20, 40, 60))
    # already-canonical box unchanged
    assert _box_approx(G.normalize_box(1, 2, 3, 4), (1, 2, 3, 4))


def test_clamp_box_clips_to_bounds():
    assert _box_approx(G.clamp_box((-5, -10, 120, 250), 100, 200), (0, 0, 100, 200))
    # reversed input still clips correctly (normalized first)
    assert _box_approx(G.clamp_box((120, 250, -5, -10), 100, 200), (0, 0, 100, 200))
    # interior box untouched
    assert _box_approx(G.clamp_box((10, 20, 30, 40), 100, 200), (10, 20, 30, 40))


def test_clamp_box_zero_dim_collapses():
    assert _box_approx(G.clamp_box((10, 10, 20, 20), 0, 100), (0, 0, 0, 0))
    assert _box_approx(G.clamp_box((10, 10, 20, 20), 100, 0), (0, 0, 0, 0))


def test_is_degenerate():
    assert G.is_degenerate((10, 10, 10.5, 30)) is True       # width 0.5 < 1
    assert G.is_degenerate((10, 10, 30, 10.5)) is True       # height 0.5 < 1
    assert G.is_degenerate((10, 10, 30, 30)) is False
    assert G.is_degenerate(None) is True
    # custom min_size
    assert G.is_degenerate((0, 0, 4, 4), min_size=5.0) is True
    assert G.is_degenerate((0, 0, 6, 6), min_size=5.0) is False
    # reversed but large enough -> not degenerate
    assert G.is_degenerate((40, 40, 10, 10)) is False


# ---------------------------------------------------------------------------
# padding (pixels + percent + negative + clamp)
# ---------------------------------------------------------------------------

def test_pad_box_pixels_adds_each_side():
    # interior box, no clamp interference
    assert _box_approx(G.pad_box((50, 50, 100, 100), 10, 5, 1000, 1000, "pixels"),
                       (40, 45, 110, 105))


def test_pad_box_percent_uses_box_fraction():
    # box 100x80 wide; 10% of 100 = 10 each side X, 25% of 80 = 20 each side Y
    out = G.pad_box((0 + 100, 0 + 100, 200, 180), 0.1, 0.25, 10000, 10000, "percent")
    assert _box_approx(out, (90, 80, 210, 200))


def test_pad_box_clamps_to_frame():
    # padding pushes past the frame edges -> clamped
    out = G.pad_box((2, 2, 8, 8), 10, 10, 20, 20, "pixels")
    assert _box_approx(out, (0, 0, 18, 18))


def test_pad_box_negative_inset_never_inverts():
    # negative padding shrinks; over-shrink must not produce a reversed box
    out = G.pad_box((10, 10, 14, 14), -5, -5, 100, 100, "pixels")
    x1, y1, x2, y2 = out
    assert x2 >= x1 and y2 >= y1


def test_pad_box_unknown_mode_defaults_pixels():
    a = G.pad_box((50, 50, 100, 100), 10, 5, 1000, 1000, "pixels")
    b = G.pad_box((50, 50, 100, 100), 10, 5, 1000, 1000, "bogus-mode")
    assert _box_approx(a, b)


# ---------------------------------------------------------------------------
# area / iou / centroid
# ---------------------------------------------------------------------------

def test_area():
    assert _approx(G.area((0, 0, 10, 20)), 200.0)
    assert _approx(G.area((10, 20, 0, 0)), 200.0)   # reversed
    assert _approx(G.area((5, 5, 5, 5)), 0.0)       # degenerate point


def test_iou_identical_is_one():
    assert _approx(G.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)


def test_iou_no_overlap_is_zero():
    assert _approx(G.iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
    # touching edges (zero-width intersection) -> 0
    assert _approx(G.iou((0, 0, 10, 10), (10, 0, 20, 10)), 0.0)


def test_iou_half_overlap():
    # A=[0,0,10,10] area100, B=[5,0,15,10] area100, inter=[5,0,10,10] area50
    # union = 100+100-50 = 150 -> iou = 50/150 = 1/3
    assert _approx(G.iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)


def test_iou_degenerate_union_zero():
    assert _approx(G.iou((5, 5, 5, 5), (5, 5, 5, 5)), 0.0)


def test_iou_in_unit_range():
    val = G.iou((0, 0, 100, 100), (10, 10, 90, 90))
    assert 0.0 <= val <= 1.0


def test_centroid_and_dist():
    assert _box_approx((*G.centroid((0, 0, 10, 20)),), (5, 10)) is False  # sanity: not a box len
    cx, cy = G.centroid((0, 0, 10, 20))
    assert _approx(cx, 5.0) and _approx(cy, 10.0)
    assert _approx(G.centroid_dist((0, 0, 0, 0), (3, 4, 3, 4)), 5.0)
    # centroid of a reversed box equals centroid of normalized box
    assert G.centroid((10, 20, 0, 0)) == G.centroid((0, 0, 10, 20))


# ---------------------------------------------------------------------------
# smoothing -> alpha, ema, adaptive
# ---------------------------------------------------------------------------

def test_smoothing_to_alpha_endpoints_and_monotonic():
    assert _approx(G.smoothing_to_alpha(0.0), 1.0)
    assert _approx(G.smoothing_to_alpha(0.9), 0.1)
    assert _approx(G.smoothing_to_alpha(1.0), 0.02)   # floor
    assert _approx(G.smoothing_to_alpha(2.0), 0.02)   # clamped above floor
    assert _approx(G.smoothing_to_alpha(-1.0), 1.0)   # clamped ceiling
    # monotonic non-increasing in smoothing
    vals = [G.smoothing_to_alpha(s / 10.0) for s in range(0, 11)]
    assert all(vals[i] >= vals[i + 1] - EPS for i in range(len(vals) - 1))


def test_ema_box_first_sight_returns_current():
    cur = (10.0, 20.0, 30.0, 40.0)
    assert _box_approx(G.ema_box(cur, None, 0.3), cur)


def test_ema_box_blend():
    cur = (10.0, 10.0, 20.0, 20.0)
    prev = (0.0, 0.0, 0.0, 0.0)
    # alpha 0.25 -> 25% toward current
    assert _box_approx(G.ema_box(cur, prev, 0.25), (2.5, 2.5, 5.0, 5.0))
    # alpha 1.0 -> exactly current; alpha 0.0 -> exactly previous
    assert _box_approx(G.ema_box(cur, prev, 1.0), cur)
    assert _box_approx(G.ema_box(cur, prev, 0.0), prev)


def test_ema_box_alpha_clamped():
    cur = (10.0, 10.0, 20.0, 20.0)
    prev = (0.0, 0.0, 0.0, 0.0)
    assert _box_approx(G.ema_box(cur, prev, 5.0), cur)     # >1 clamps to 1
    assert _box_approx(G.ema_box(cur, prev, -5.0), prev)   # <0 clamps to 0


def test_adaptive_alpha_motion_increases_alpha():
    base = 0.2
    prev = (0.0, 0.0, 10.0, 10.0)
    frame_diag = math.hypot(100, 100)
    # no motion -> base
    assert _approx(G.adaptive_alpha(base, prev, prev, frame_diag), base)
    # small motion -> slightly above base
    small = G.adaptive_alpha(base, (1, 0, 11, 10), prev, frame_diag)
    # large motion (>= one diagonal) -> alpha 1.0
    large = G.adaptive_alpha(base, (1000, 1000, 1010, 1010), prev, frame_diag)
    assert base <= small <= large
    assert _approx(large, 1.0)


def test_adaptive_alpha_monotonic_in_motion():
    base = 0.1
    prev = (0.0, 0.0, 10.0, 10.0)
    diag = 100.0
    last = -1.0
    for d in range(0, 130, 10):
        a = G.adaptive_alpha(base, (d, 0, 10 + d, 10), prev, diag)
        assert a >= last - EPS
        last = a
    assert _approx(last, 1.0)


def test_adaptive_alpha_no_history_or_bad_diag():
    base = 0.3
    assert _approx(G.adaptive_alpha(base, (0, 0, 1, 1), None, 100.0), base)
    assert _approx(G.adaptive_alpha(base, (0, 0, 1, 1), (5, 5, 6, 6), 0.0), base)
    assert _approx(G.adaptive_alpha(base, (0, 0, 1, 1), (5, 5, 6, 6), -10.0), base)


# ---------------------------------------------------------------------------
# importance
# ---------------------------------------------------------------------------

def test_importance_area_ratio():
    # 50x40 box in 100x100 frame -> 2000/10000 = 0.2
    assert _approx(G.importance((0, 0, 50, 40), 100, 100), 0.2)


def test_importance_zero_frame():
    assert _approx(G.importance((0, 0, 10, 10), 0, 100), 0.0)


def test_importance_center_bias_boost():
    box = (40, 40, 60, 60)        # centered 20x20 in 100x100, area 400 -> base 0.04
    base = G.importance(box, 100, 100, center_bias=0.0)
    biased = G.importance(box, 100, 100, center_bias=1.0)
    assert _approx(base, 0.04)
    # dead-center box: full (1+bias) boost -> base*2
    assert _approx(biased, base * 2.0)


def test_importance_center_bias_corner_smaller_than_center():
    corner = (0, 0, 20, 20)
    center = (40, 40, 60, 60)     # same size, centered
    ic = G.importance(corner, 100, 100, center_bias=1.0)
    ce = G.importance(center, 100, 100, center_bias=1.0)
    assert ce > ic   # center boosted more than corner for equal area


# ---------------------------------------------------------------------------
# target_center
# ---------------------------------------------------------------------------

def test_target_center_is_centroid():
    assert G.target_center((0, 0, 10, 20)) == G.centroid((0, 0, 10, 20))


# ---------------------------------------------------------------------------
# label_rect — MUST keep labels inside the frame
# ---------------------------------------------------------------------------

def test_label_rect_prefers_top_when_room():
    # box well below the top edge, plenty of room above
    rect, side = G.label_rect((50, 100, 150, 200), text_w=60, text_h=14, W=400, H=400, side="top")
    assert side == "top"
    rx1, ry1, rx2, ry2 = rect
    # strip directly above the box, left-aligned
    assert _approx(rx1, 50) and _approx(ry2, 100)
    assert _approx(rx2 - rx1, 60) and _approx(ry2 - ry1, 14)


def test_label_rect_flips_to_bottom_when_top_clipped():
    # box hugging the top edge -> top strip would go off-frame -> flip to bottom
    rect, side = G.label_rect((50, 2, 150, 80), text_w=60, text_h=14, W=400, H=400, side="top")
    assert side == "bottom"
    _, ry1, _, _ = rect
    assert _approx(ry1, 80)   # sits just below the box's bottom edge


def test_label_rect_flips_to_top_when_bottom_clipped():
    # box hugging the bottom edge, preferred bottom -> flip to top
    rect, side = G.label_rect((50, 300, 150, 398), text_w=60, text_h=14, W=400, H=400, side="bottom")
    assert side == "top"
    _, ry1, _, ry2 = rect
    assert _approx(ry2, 300)   # sits just above the box top


def test_label_rect_right_align_on_horizontal_overflow():
    # box near right edge: left-aligned strip overflows -> right-align to box right
    rect, _ = G.label_rect((350, 100, 390, 200), text_w=80, text_h=14, W=400, H=400, side="top")
    rx1, _, rx2, _ = rect
    assert rx2 <= 400 + EPS              # fully on-frame
    assert _approx(rx2, 390)            # right edge aligned to box right


def test_label_rect_always_inside_frame_fuzz():
    # exhaustive-ish: many boxes/sizes, strip must never leave [0,W]x[0,H]
    W, H = 320, 240
    rng = np.random.default_rng(12345)
    for _ in range(400):
        x1 = float(rng.integers(-20, W + 20))
        y1 = float(rng.integers(-20, H + 20))
        bw = float(rng.integers(1, 150))
        bh = float(rng.integers(1, 150))
        tw = int(rng.integers(1, 120))
        th = int(rng.integers(1, 40))
        side = "top" if rng.integers(0, 2) == 0 else "bottom"
        rect, resolved = G.label_rect((x1, y1, x1 + bw, y1 + bh), tw, th, W, H, side)
        rx1, ry1, rx2, ry2 = rect
        assert rx1 >= -EPS and ry1 >= -EPS
        assert rx2 <= W + EPS and ry2 <= H + EPS
        assert resolved in ("top", "bottom")


def test_label_rect_huge_strip_clamped_not_offscreen():
    # strip taller than the frame can't fit either side -> clamped on-frame
    rect, side = G.label_rect((10, 10, 50, 30), text_w=10, text_h=500, W=100, H=100, side="top")
    rx1, ry1, rx2, ry2 = rect
    assert ry1 >= -EPS and rx1 >= -EPS
    # top edge clamped to 0 since it cannot fit below the frame height
    assert _approx(ry1, 0.0)
    assert side in ("top", "bottom")


def test_label_rect_strip_wider_than_frame_clamped_left():
    rect, _ = G.label_rect((10, 50, 40, 90), text_w=500, text_h=14, W=100, H=200, side="top")
    rx1, _, _, _ = rect
    assert _approx(rx1, 0.0)


# ---------------------------------------------------------------------------
# asset_rect — stretch / contain / cover + inner padding
# ---------------------------------------------------------------------------

def test_asset_rect_stretch_fills_box():
    assert _box_approx(G.asset_rect((0, 0, 100, 50), "stretch", 0.0, (1, 1)),
                       (0, 0, 100, 50))
    # stretch ignores aspect entirely
    assert _box_approx(G.asset_rect((0, 0, 100, 50), "stretch", 0.0, (3, 7)),
                       (0, 0, 100, 50))


def test_asset_rect_inner_padding_insets():
    # 10% padding on a 100x100 box -> inset 10 each side
    assert _box_approx(G.asset_rect((0, 0, 100, 100), "stretch", 0.1, None),
                       (10, 10, 90, 90))


def test_asset_rect_contain_letterbox():
    # square asset (1:1) inside a wide 200x100 inner rect -> height-bound 100x100, centered
    rect = G.asset_rect((0, 0, 200, 100), "contain", 0.0, (1, 1))
    rx1, ry1, rx2, ry2 = rect
    assert _approx(rx2 - rx1, 100) and _approx(ry2 - ry1, 100)
    assert _approx((rx1 + rx2) / 2, 100) and _approx((ry1 + ry2) / 2, 50)  # centered
    # the asset stays WITHIN the box for contain
    assert rx1 >= -EPS and ry1 >= -EPS and rx2 <= 200 + EPS and ry2 <= 100 + EPS


def test_asset_rect_cover_overflows_one_axis():
    # square asset covering a wide 200x100 rect -> width-bound 200x200, overflows vertically
    rect = G.asset_rect((0, 0, 200, 100), "cover", 0.0, (1, 1))
    rx1, ry1, rx2, ry2 = rect
    assert _approx(rx2 - rx1, 200) and _approx(ry2 - ry1, 200)
    assert _approx((rx1 + rx2) / 2, 100) and _approx((ry1 + ry2) / 2, 50)  # centered
    # vertical overflow beyond the box (centered): top < 0, bottom > 100
    assert ry1 < 0 and ry2 > 100


def test_asset_rect_contain_tall_asset_pillarbox():
    # tall asset (1:2 -> w:h) inside a square 100x100 -> width 50, height 100
    rect = G.asset_rect((0, 0, 100, 100), "contain", 0.0, (1, 2))
    rx1, ry1, rx2, ry2 = rect
    assert _approx(rx2 - rx1, 50) and _approx(ry2 - ry1, 100)
    assert _approx((rx1 + rx2) / 2, 50)


def test_asset_rect_bad_aspect_falls_back_to_inner():
    # aspect None / non-positive -> inner rect (stretch)
    inner = (10, 10, 90, 90)
    assert _box_approx(G.asset_rect((0, 0, 100, 100), "contain", 0.1, None), inner)
    assert _box_approx(G.asset_rect((0, 0, 100, 100), "contain", 0.1, (0, 5)), inner)
    assert _box_approx(G.asset_rect((0, 0, 100, 100), "cover", 0.1, (5, 0)), inner)


def test_asset_rect_unknown_fit_falls_back_to_inner():
    inner = (5, 5, 95, 95)
    assert _box_approx(G.asset_rect((0, 0, 100, 100), "bogus", 0.05, (1, 1)), inner)


def test_asset_rect_padding_clamped():
    # huge padding clamped to <0.49 so inner rect never inverts
    rect = G.asset_rect((0, 0, 100, 100), "stretch", 5.0, None)
    rx1, ry1, rx2, ry2 = rect
    assert rx2 >= rx1 and ry2 >= ry1


# ---------------------------------------------------------------------------
# dtype / scalar robustness (numpy + torch scalars, NaN/Inf, None)
# ---------------------------------------------------------------------------

def test_accepts_numpy_and_torch_scalars():
    box_np = (np.float32(0), np.int64(0), np.float64(10), np.float32(20))
    assert _approx(G.area(box_np), 200.0)
    box_t = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(10.0), torch.tensor(20.0))
    cx, cy = G.centroid(box_t)
    assert _approx(cx, 5.0) and _approx(cy, 10.0)


def test_nonfinite_coords_are_sanitized():
    # NaN / Inf map to 0.0 rather than propagating
    out = G.normalize_box(float("nan"), float("inf"), 10, 20)
    assert all(math.isfinite(v) for v in out)
    # area with a NaN coord doesn't blow up
    val = G.area((float("nan"), 0, 10, 20))
    assert math.isfinite(val)


def test_box_alias_is_tuple_type():
    # types.Box is the tuple alias the contract advertises; a returned Box is a tuple
    out = G.xywh_to_xyxy(0, 0, 1, 1)
    assert isinstance(out, tuple) and len(out) == 4
    # alias exists and is the 4-float tuple type
    assert Box is not None


# ---------------------------------------------------------------------------
# Script runner (so the file works without pytest installed)
# ---------------------------------------------------------------------------

def _collect_tests():
    g = globals()
    return [(name, g[name]) for name in sorted(g) if name.startswith("test_") and callable(g[name])]


def _main() -> int:
    tests = _collect_tests()
    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — report every failure
            import traceback
            failed.append(name)
            print(f"FAIL  {name}: {exc.__class__.__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")
    print("-" * 60)
    print(f"geometry: {passed}/{len(tests)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
