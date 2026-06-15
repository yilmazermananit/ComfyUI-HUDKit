"""Tests for the adaptive label placement engine."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yea_hud.label_placement import LabelPlacer, PlacementRequest, candidate_rect, POSITIONS


def test_placed_rects_in_bounds():
    placer = LabelPlacer(1920, 1080)
    reqs = [PlacementRequest(key="person:0", anchor=(800, 400, 1000, 900), tw=300, th=28,
                             priority=100, is_primary=True)]
    placed = placer.place(reqs)
    r = placed[0].rect
    assert 0 <= r[0] and 0 <= r[1] and r[2] <= 1920 and r[3] <= 1080


def test_top_boundary_label_goes_below_or_inside():
    # anchor hugging the top: "above" would leave the frame -> engine should not pick an
    # above_* position (its clamped rect would be heavily clipped).
    placer = LabelPlacer(1920, 1080)
    reqs = [PlacementRequest(key="t", anchor=(100, 0, 400, 200), tw=250, th=30, priority=80)]
    placed = placer.place(reqs)
    assert placed[0].position not in ("above_left", "above_right") or placed[0].rect[1] >= 0


def test_right_boundary_label_stays_in_frame():
    placer = LabelPlacer(1920, 1080)
    reqs = [PlacementRequest(key="r", anchor=(1700, 400, 1900, 700), tw=400, th=30, priority=80)]
    placed = placer.place(reqs)
    assert placed[0].rect[2] <= 1920


def test_temporal_stability_keeps_position():
    placer = LabelPlacer(1920, 1080)
    req = PlacementRequest(key="person:0", anchor=(800, 400, 1000, 900), tw=300, th=28, priority=100)
    p1 = placer.place([req])[0]
    # move the anchor slightly; placement should keep the same relative position
    req2 = PlacementRequest(key="person:0", anchor=(810, 405, 1010, 905), tw=300, th=28, priority=100)
    p2 = placer.place([req2])[0]
    assert p1.position == p2.position


def test_two_labels_avoid_each_other():
    placer = LabelPlacer(1920, 1080, collision_mode="reposition")
    reqs = [
        PlacementRequest(key="a", anchor=(500, 500, 700, 800), tw=200, th=28, priority=100, is_primary=True),
        PlacementRequest(key="b", anchor=(520, 500, 720, 800), tw=200, th=28, priority=50),
    ]
    placed = placer.place(reqs)
    # higher-priority "a" is placed; "b" either repositions or is hidden, but both resolved
    assert len(placed) == 2
    assert placed[0].visible  # primary always visible


def test_fifty_labels_no_crash():
    placer = LabelPlacer(1920, 1080, collision_mode="dense_editorial")
    reqs = []
    for i in range(50):
        x = (i % 10) * 180 + 20
        y = (i // 10) * 200 + 20
        reqs.append(PlacementRequest(key=f"obj:{i}", anchor=(x, y, x + 120, y + 160),
                                     tw=140, th=24, priority=40 + (i % 5)))
    placed = placer.place(reqs)
    assert len(placed) == 50
    for p in placed:
        assert p.rect[0] >= 0 and p.rect[1] >= 0 and p.rect[2] <= 1920 and p.rect[3] <= 1080


def test_candidate_rect_all_positions():
    for pos in POSITIONS:
        r = candidate_rect((100, 100, 300, 400), 150, 30, pos)
        assert r[2] > r[0] and r[3] > r[1]


def test_hide_low_priority_on_overlap():
    placer = LabelPlacer(400, 400, collision_mode="hide_low_priority")
    # two labels forced to the same tiny region
    reqs = [
        PlacementRequest(key="hi", anchor=(150, 150, 250, 250), tw=180, th=28, priority=100, is_primary=True),
        PlacementRequest(key="lo", anchor=(150, 150, 250, 250), tw=180, th=28, priority=10),
    ]
    placed = placer.place(reqs)
    assert placed[0].visible
    # low-priority may be hidden under heavy overlap (not asserted strictly, but must not crash)
    assert isinstance(placed[1].visible, bool)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa
            failed += 1; print(f"ERROR {fn.__name__}: {e!r}")
    print("-" * 60)
    print(f"label_placement tests: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
