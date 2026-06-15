"""Unit tests for yonder_hud.bbox_adapter.

Runnable BOTH ways:

* pytest:   python -m pytest tests/test_bbox_adapter.py -q
* script:   python tests/test_bbox_adapter.py   (runs every test, prints a
            summary, sys.exit(1) on any failure)

No ComfyUI, no network, no model files. Tiny synthetic numpy/torch fixtures
that exactly mirror the native BOUNDING_BOX value SAM3_Detect builds (findings
§3): per-frame list[list[{"x","y","width","height","score"}]] of absolute-pixel
xywh dicts, plus the single-dict / list[dict] broadcast shapes and the brief
§4.3 defensive shapes (xyxy dicts, 4-tuples, normalized coords).
"""

from __future__ import annotations

import os
import sys

# Make the package importable regardless of CWD when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402  (defensive ndarray-box coverage)
import torch  # noqa: E402  (defensive tensor-box coverage)

from yonder_hud import bbox_adapter as BA  # noqa: E402
from yonder_hud.types import Detection  # noqa: E402


EPS = 1e-4


def _approx(a, b, eps=EPS):
    return abs(float(a) - float(b)) <= eps


def _box_approx(det: Detection, xyxy, eps=EPS):
    return (_approx(det.x1, xyxy[0], eps) and _approx(det.y1, xyxy[1], eps)
            and _approx(det.x2, xyxy[2], eps) and _approx(det.y2, xyxy[3], eps))


def _xywh(x, y, w, h, **extra):
    """Build a native xywh box dict exactly like SAM3_Detect does."""
    d = {"x": float(x), "y": float(y), "width": float(w), "height": float(h)}
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# §25.1 — the three native shapes
# ---------------------------------------------------------------------------

def test_single_dict_broadcasts_to_all_frames():
    box = _xywh(10, 20, 30, 40, score=0.9)
    out = BA.adapt_bboxes(box, B=3, W=200, H=200)
    assert len(out) == 3
    for f in range(3):
        assert len(out[f]) == 1
        det = out[f][0]
        assert det.frame_index == f
        assert det.source_index == 0
        assert _box_approx(det, (10, 20, 40, 60))   # x+w, y+h
        assert _approx(det.score, 0.9)
        # broadcast path feeds the IoU tracker, so no native track id
        assert det.source_track_id is None
        assert det.mask_index is None


def test_list_of_dicts_broadcasts_to_all_frames():
    boxes = [_xywh(0, 0, 10, 10), _xywh(50, 60, 20, 20)]
    out = BA.adapt_bboxes(boxes, B=2, W=300, H=300)
    assert len(out) == 2
    for f in range(2):
        assert len(out[f]) == 2
        assert _box_approx(out[f][0], (0, 0, 10, 10))
        assert _box_approx(out[f][1], (50, 60, 70, 80))
        assert out[f][0].source_index == 0
        assert out[f][1].source_index == 1
        assert out[f][0].frame_index == f


def test_list_of_lists_per_frame():
    # Exactly SAM3_Detect's output shape: list[list[dict]] with score.
    bboxes = [
        [_xywh(1, 1, 5, 5, score=0.8)],                 # frame 0: 1 det
        [_xywh(10, 10, 5, 5), _xywh(20, 20, 5, 5)],     # frame 1: 2 dets
        [],                                              # frame 2: empty
    ]
    out = BA.adapt_bboxes(bboxes, B=3, W=100, H=100)
    assert len(out) == 3
    assert len(out[0]) == 1 and _box_approx(out[0][0], (1, 1, 6, 6))
    assert out[0][0].frame_index == 0
    assert len(out[1]) == 2
    assert _box_approx(out[1][0], (10, 10, 15, 15))
    assert _box_approx(out[1][1], (20, 20, 25, 25))
    assert out[1][1].frame_index == 1 and out[1][1].source_index == 1
    assert out[2] == []


# ---------------------------------------------------------------------------
# Broadcasting rules (brief §7) — NEVER auto-repeat
# ---------------------------------------------------------------------------

def test_list_list_shorter_than_B_pads_empty_never_repeats():
    bboxes = [
        [_xywh(0, 0, 10, 10)],
        [_xywh(5, 5, 10, 10)],
    ]
    out = BA.adapt_bboxes(bboxes, B=5, W=100, H=100)
    assert len(out) == 5
    assert len(out[0]) == 1 and len(out[1]) == 1
    # Frames 2,3,4 must be EMPTY (not a repeat of frame 1).
    assert out[2] == [] and out[3] == [] and out[4] == []


def test_list_list_longer_than_B_ignores_extra():
    bboxes = [
        [_xywh(0, 0, 10, 10)],
        [_xywh(5, 5, 10, 10)],
        [_xywh(7, 7, 10, 10)],
        [_xywh(9, 9, 10, 10)],
    ]
    out = BA.adapt_bboxes(bboxes, B=2, W=100, H=100)
    assert len(out) == 2
    assert _box_approx(out[0][0], (0, 0, 10, 10))
    assert _box_approx(out[1][0], (5, 5, 15, 15))


def test_single_dict_to_one_frame():
    out = BA.adapt_bboxes(_xywh(1, 2, 3, 4), B=1, W=50, H=50)
    assert len(out) == 1 and len(out[0]) == 1
    assert _box_approx(out[0][0], (1, 2, 4, 6))


# ---------------------------------------------------------------------------
# Defensive shapes (brief §4.3)
# ---------------------------------------------------------------------------

def test_xyxy_dict_detected():
    box = {"x1": 10, "y1": 20, "x2": 40, "y2": 60}
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert _box_approx(out[0][0], (10, 20, 40, 60))


def test_xyxy_dict_minmax_keys():
    box = {"xmin": 5, "ymin": 6, "xmax": 25, "ymax": 36}
    out = BA.adapt_bboxes([box], B=1, W=100, H=100)
    assert _box_approx(out[0][0], (5, 6, 25, 36))


def test_width_height_preferred_over_xmax_when_both_present():
    # If a dict carries BOTH width/height and x2/y2, native xywh wins.
    box = {"x": 10, "y": 10, "width": 5, "height": 5, "x2": 999, "y2": 999}
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert _box_approx(out[0][0], (10, 10, 15, 15))


def test_4tuple_xyxy_heuristic():
    # 3rd>1st and 4th>2nd -> treated as xyxy per the brief's rule.
    out = BA.adapt_bboxes([(10.0, 20.0, 40.0, 60.0)], B=1, W=100, H=100)
    assert _box_approx(out[0][0], (10, 20, 40, 60))


def test_4tuple_xywh_when_extent_form():
    # 3rd/4th are width/height that do NOT push past the corner (x=0 corner):
    # (0, 0, 30, 40): 30>0 and 40>0 -> heuristic says xyxy here, giving same box.
    # Use a corner where width<x to force the xywh branch: (50, 50, 10, 10)
    # -> 3rd(10) !> 1st(50) -> xywh -> (50,50,60,60).
    out = BA.adapt_bboxes([[50.0, 50.0, 10.0, 10.0]], B=1, W=200, H=200)
    assert _box_approx(out[0][0], (50, 50, 60, 60))


def test_numpy_and_torch_4tuples():
    arr = np.array([50.0, 50.0, 10.0, 10.0], dtype=np.float32)
    ten = torch.tensor([50.0, 50.0, 10.0, 10.0])
    out_np = BA.adapt_bboxes([arr], B=1, W=200, H=200)
    out_t = BA.adapt_bboxes([ten], B=1, W=200, H=200)
    assert _box_approx(out_np[0][0], (50, 50, 60, 60))
    assert _box_approx(out_t[0][0], (50, 50, 60, 60))


def test_bare_4tuple_is_single_box_not_perframe():
    # A bare 4-number sequence must be ONE box broadcast to all frames,
    # never misread as a per-frame list of 4 scalars.
    out = BA.adapt_bboxes((50, 50, 10, 10), B=3, W=200, H=200)
    assert len(out) == 3
    for f in range(3):
        assert len(out[f]) == 1
        assert _box_approx(out[f][0], (50, 50, 60, 60))


def test_normalized_coords_scaled_to_pixels():
    # All coords <= 1.0 -> normalized -> *W/*H. xywh form.
    box = _xywh(0.1, 0.2, 0.3, 0.4)
    out = BA.adapt_bboxes(box, B=1, W=1000, H=500)
    # x=100, y=100, w=300->x2=400; h=200->y2=300
    assert _box_approx(out[0][0], (100, 100, 400, 300))


def test_normalized_xyxy_dict_scaled():
    box = {"x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8}
    out = BA.adapt_bboxes(box, B=1, W=1000, H=1000)
    assert _box_approx(out[0][0], (100, 200, 500, 800))


def test_mixed_normalized_and_pixel_boxes_per_box_detection():
    # One normalized box + one pixel box in the same frame list.
    boxes = [
        _xywh(0.1, 0.1, 0.2, 0.2),    # normalized -> 100,100,300,300 @ W=H=1000
        _xywh(500, 500, 100, 100),    # pixel -> 500,500,600,600
    ]
    out = BA.adapt_bboxes(boxes, B=1, W=1000, H=1000)
    assert len(out[0]) == 2
    assert _box_approx(out[0][0], (100, 100, 300, 300))
    assert _box_approx(out[0][1], (500, 500, 600, 600))


# ---------------------------------------------------------------------------
# Score / label carry-through
# ---------------------------------------------------------------------------

def test_score_carried_label_absent_for_sam3():
    # SAM3: score present, label absent.
    box = _xywh(10, 10, 20, 20, score=0.42)
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert _approx(out[0][0].score, 0.42)
    assert out[0][0].label is None


def test_label_carried_for_mediapipe_style():
    box = _xywh(10, 10, 20, 20, label="face", score=0.7)
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert out[0][0].label == "face"
    assert _approx(out[0][0].score, 0.7)


def test_empty_label_treated_as_absent():
    box = _xywh(10, 10, 20, 20, label="   ")
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert out[0][0].label is None


def test_nonfinite_score_dropped():
    box = _xywh(10, 10, 20, 20, score=float("nan"))
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert out[0][0].score is None


# ---------------------------------------------------------------------------
# Normalization / clamping / degenerate filtering
# ---------------------------------------------------------------------------

def test_reversed_xyxy_is_swapped():
    box = {"x1": 40, "y1": 60, "x2": 10, "y2": 20}
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    det = out[0][0]
    assert det.x2 >= det.x1 and det.y2 >= det.y1
    assert _box_approx(det, (10, 20, 40, 60))


def test_negative_width_xywh_swapped_after_normalize():
    box = _xywh(40, 40, -30, -20)   # x2=10, y2=20 -> normalize -> (10,20,40,40)
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    det = out[0][0]
    assert _box_approx(det, (10, 20, 40, 40))


def test_out_of_bounds_box_clamped():
    box = _xywh(-50, -50, 1000, 1000)   # xyxy=(-50,-50,950,950) -> clamp to 100x100
    out = BA.adapt_bboxes(box, B=1, W=100, H=100)
    assert _box_approx(out[0][0], (0, 0, 100, 100))


def test_degenerate_box_dropped_but_source_index_preserved():
    # frame has [valid, degenerate(0-size), valid] -> 2 dets with src_idx 0 and 2.
    boxes = [_xywh(0, 0, 10, 10), _xywh(50, 50, 0, 0), _xywh(80, 80, 10, 10)]
    out = BA.adapt_bboxes(boxes, B=1, W=100, H=100)
    assert len(out[0]) == 2
    assert out[0][0].source_index == 0
    assert out[0][1].source_index == 2   # the dropped degenerate consumed index 1
    assert _box_approx(out[0][1], (80, 80, 90, 90))


def test_subpixel_box_dropped():
    # width/height < 1px -> degenerate -> dropped.
    box = _xywh(10, 10, 0.5, 0.5)
    out = BA.adapt_bboxes(box, B=2, W=100, H=100)
    assert out[0] == [] and out[1] == []


# ---------------------------------------------------------------------------
# Empty / None / garbage / dimension edge cases
# ---------------------------------------------------------------------------

def test_none_yields_B_empty_frames():
    out = BA.adapt_bboxes(None, B=4, W=100, H=100)
    assert out == [[], [], [], []]


def test_empty_list_yields_B_empty_frames():
    out = BA.adapt_bboxes([], B=3, W=100, H=100)
    assert out == [[], [], []]


def test_list_of_empty_frames():
    out = BA.adapt_bboxes([[], [], []], B=3, W=100, H=100)
    assert out == [[], [], []]


def test_none_frame_placeholder_in_list_list():
    # SAM3 pads per-frame entries with None for frames with no boxes.
    bboxes = [[_xywh(0, 0, 10, 10)], None, [_xywh(5, 5, 10, 10)]]
    out = BA.adapt_bboxes(bboxes, B=3, W=100, H=100)
    assert len(out[0]) == 1
    assert out[1] == []
    assert len(out[2]) == 1


def test_B_zero_returns_empty_list():
    assert BA.adapt_bboxes(_xywh(0, 0, 10, 10), B=0, W=100, H=100) == []


def test_zero_dimensions_collapse_to_empty():
    box = _xywh(0, 0, 50, 50)
    assert BA.adapt_bboxes(box, B=2, W=0, H=100) == [[], []]
    assert BA.adapt_bboxes(box, B=2, W=100, H=0) == [[], []]


def test_garbage_input_yields_empty():
    assert BA.adapt_bboxes("not a box", B=2, W=100, H=100) == [[], []]
    assert BA.adapt_bboxes(12345, B=2, W=100, H=100) == [[], []]
    assert BA.adapt_bboxes({"foo": "bar"}, B=1, W=100, H=100) == [[]]


def test_input_not_mutated():
    boxes = [[_xywh(0, 0, 10, 10)]]
    before = repr(boxes)
    BA.adapt_bboxes(boxes, B=3, W=100, H=100)
    assert repr(boxes) == before


def test_negative_B_treated_as_zero():
    assert BA.adapt_bboxes(_xywh(0, 0, 10, 10), B=-3, W=100, H=100) == []


# ---------------------------------------------------------------------------
# summarize_bboxes
# ---------------------------------------------------------------------------

def test_summary_single_dict():
    s = BA.summarize_bboxes(_xywh(10, 20, 30, 40, score=0.9))
    assert s["kind"] == "dict"
    assert s["broadcast"] == "all_frames"
    assert s["total_boxes"] == 1
    assert s["has_score"] is True
    assert s["has_label"] is False
    assert s["coord_space"] == "pixel"
    assert "x" in s["keys"] and "width" in s["keys"]


def test_summary_list_list_counts_and_frames():
    bboxes = [
        [_xywh(0, 0, 5, 5)],
        [_xywh(0, 0, 5, 5), _xywh(1, 1, 5, 5)],
        [],
        [_xywh(2, 2, 5, 5)],
    ]
    s = BA.summarize_bboxes(bboxes, max_frames=2)
    assert s["kind"] == "list_list"
    assert s["broadcast"] == "per_frame"
    assert s["n_frames"] == 4
    assert s["total_boxes"] == 4
    assert s["per_frame_counts"] == [1, 2]      # truncated to max_frames=2
    assert s["frames_omitted"] == 2


def test_summary_normalized_detected():
    s = BA.summarize_bboxes(_xywh(0.1, 0.2, 0.3, 0.4))
    assert s["coord_space"] == "normalized"


def test_summary_mixed_coord_space():
    bboxes = [[_xywh(0.1, 0.1, 0.2, 0.2)], [_xywh(500, 500, 100, 100)]]
    s = BA.summarize_bboxes(bboxes)
    assert s["coord_space"] == "mixed"


def test_summary_label_detected():
    s = BA.summarize_bboxes([_xywh(0, 0, 5, 5, label="face")])
    assert s["has_label"] is True
    assert s["kind"] == "list_dict"


def test_summary_none_and_empty():
    sn = BA.summarize_bboxes(None)
    assert sn["kind"] == "none" and sn["broadcast"] == "none" and sn["total_boxes"] == 0
    se = BA.summarize_bboxes([])
    assert se["kind"] == "empty_list" and se["n_frames"] == 0


def test_summary_unknown_does_not_raise():
    s = BA.summarize_bboxes(object())
    assert s["kind"] == "unknown"
    assert s["total_boxes"] == 0
    assert isinstance(s["note"], str) and s["note"]


def test_summary_never_dumps_giant_data():
    # 10_000 boxes in one frame; summary must bound its scan (max_dets) and not
    # explode. We assert it returns fast-ish structurally (counts only).
    big = [[_xywh(i % 50, i % 50, 5, 5) for i in range(10000)]]
    s = BA.summarize_bboxes(big, max_dets=5)
    assert s["total_boxes"] == 10000          # counts are exact
    assert s["per_frame_counts"] == [10000]
    # keys came from sampling only; still the right key set
    assert "x" in s["keys"]


# ---------------------------------------------------------------------------
# Real SAM3-shaped fixture (mirrors nodes_sam3.py execute output construction)
# ---------------------------------------------------------------------------

def _sam3_like_output(boxes_per_frame, scores_per_frame):
    """Reproduce SAM3_Detect.execute's all_bbox_dicts construction (findings §3)."""
    all_bbox_dicts = []
    for frame_boxes, frame_scores in zip(boxes_per_frame, scores_per_frame):
        frame_bbox_dicts = []
        for (bx0, by0, bx1, by1), score in zip(frame_boxes, frame_scores):
            frame_bbox_dicts.append({
                "x": float(bx0), "y": float(by0),
                "width": float(bx1 - bx0), "height": float(by1 - by0),
                "score": float(score),
            })
        all_bbox_dicts.append(frame_bbox_dicts)
    return all_bbox_dicts


def test_sam3_shaped_output_roundtrips():
    # Two frames, mirroring kept_boxes (xyxy in model space) -> xywh dicts.
    boxes = [
        [(10, 20, 50, 80)],                 # frame 0: 1 box, xyxy
        [(100, 100, 140, 160), (5, 5, 25, 25)],
    ]
    scores = [[0.95], [0.88, 0.77]]
    native = _sam3_like_output(boxes, scores)
    out = BA.adapt_bboxes(native, B=2, W=640, H=480)
    assert len(out) == 2
    assert _box_approx(out[0][0], (10, 20, 50, 80))
    assert _approx(out[0][0].score, 0.95)
    assert _box_approx(out[1][0], (100, 100, 140, 160))
    assert _box_approx(out[1][1], (5, 5, 25, 25))
    assert _approx(out[1][1].score, 0.77)
    # summary agrees with structure
    s = BA.summarize_bboxes(native)
    assert s["kind"] == "list_list" and s["n_frames"] == 2 and s["total_boxes"] == 3
    assert s["has_score"] is True and s["has_label"] is False


def test_returns_detection_instances():
    out = BA.adapt_bboxes(_xywh(0, 0, 10, 10), B=1, W=100, H=100)
    assert isinstance(out[0][0], Detection)


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
    print(f"bbox_adapter: {passed}/{len(tests)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
