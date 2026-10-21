"""Tests for ``yea_hud.mask_adapter`` (brief §25, mask cases).

Covers the MASK shapes verified in ``docs/runtime_findings.md`` §4
(``[B,H,W]``, ``[H,W]``, ``[B,1,H,W]``, flattened ``[N,H,W]``), the numpy/torch
connected-component analysis (``min_area`` drop + ``merge_distance`` merge), and
the native ``BOUNDING_BOX`` value emitted by ``masks_to_native_bboxes`` (the
``list[list[{"x","y","width","height","score"}]]`` xywh shape — NO custom type).

Runnable two ways:

    python -m pytest tests/test_mask_adapter.py -q
    python tests/test_mask_adapter.py

The ``__main__`` block runs every ``test_*`` function, prints a summary, and
``sys.exit(1)`` on any failure. No ComfyUI, no network, no model files — only
small synthetic numpy/torch fixtures.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make the package importable when run as a bare script from the repo root or
# from within the tests/ directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402  (allowed dep; imported after sys.path fix)

from yea_hud import mask_adapter  # noqa: E402
from yea_hud.types import Detection  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _blob_mask(h, w, blobs):
    """Build a float32 [H,W] mask with rectangular blobs set to 1.0.

    ``blobs`` is a list of (y1, y2, x1, x2) half-open rectangles.
    """
    m = np.zeros((h, w), dtype=np.float32)
    for (y1, y2, x1, x2) in blobs:
        m[y1:y2, x1:x2] = 1.0
    return m


def _det_boxes(dets):
    """Return a set of integer xyxy tuples for easy comparison."""
    return {(int(d.x1), int(d.y1), int(d.x2), int(d.y2)) for d in dets}


# ---------------------------------------------------------------------------
# normalize_mask_batch — the §4 MASK shapes
# ---------------------------------------------------------------------------

def test_normalize_BHW_canonical():
    """[B,H,W] splits into B planes of [H,W] float32."""
    t = torch.zeros(3, 4, 5)
    t[0] = 0.0
    t[1] = 1.0
    t[2] = 0.5
    planes = mask_adapter.normalize_mask_batch(t, 3)
    assert isinstance(planes, list) and len(planes) == 3
    for p in planes:
        assert isinstance(p, np.ndarray) and p.dtype == np.float32 and p.shape == (4, 5)
    assert np.allclose(planes[0], 0.0)
    assert np.allclose(planes[1], 1.0)
    assert np.allclose(planes[2], 0.5)


def test_normalize_HW_single_B1():
    """[H,W] with B==1 -> one-element list."""
    m = np.ones((6, 7), dtype=np.float32)
    planes = mask_adapter.normalize_mask_batch(m, 1)
    assert len(planes) == 1
    assert planes[0].shape == (6, 7)
    assert np.allclose(planes[0], 1.0)


def test_normalize_HW_broadcast_to_B():
    """[H,W] with B>1 broadcasts the union mask to all B frames (copies)."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[1, 1] = 1.0
    planes = mask_adapter.normalize_mask_batch(m, 4)
    assert len(planes) == 4
    for p in planes:
        assert p.shape == (4, 4)
        assert np.isclose(p[1, 1], 1.0)
    # Independent buffers (mutating one must not touch the others).
    planes[0][0, 0] = 1.0
    assert planes[1][0, 0] == 0.0


def test_normalize_B1HW_channel_first():
    """[B,1,H,W] squeezes the channel and splits into B planes."""
    t = torch.zeros(2, 1, 3, 3)
    t[0, 0] = 0.25
    t[1, 0] = 0.75
    planes = mask_adapter.normalize_mask_batch(t, 2)
    assert len(planes) == 2
    assert planes[0].shape == (3, 3)
    assert np.allclose(planes[0], 0.25)
    assert np.allclose(planes[1], 0.75)


def test_normalize_HW1_trailing_channel():
    """[H,W,1] squeezes the trailing channel."""
    m = np.full((5, 5, 1), 0.4, dtype=np.float32)
    planes = mask_adapter.normalize_mask_batch(m, 1)
    assert len(planes) == 1 and planes[0].shape == (5, 5)
    assert np.allclose(planes[0], 0.4)


def test_normalize_flattened_NHW_returned_asis():
    """Flattened [N,H,W] with N != B is returned as-is (length N, not B)."""
    # SAM3 individual_masks: 5 instances across e.g. 2 frames -> [5,H,W].
    t = torch.zeros(5, 4, 4)
    for i in range(5):
        t[i] = float(i) / 5.0
    planes = mask_adapter.normalize_mask_batch(t, 2)  # B=2 but N=5
    assert len(planes) == 5  # NOT split to B; native length preserved
    for i, p in enumerate(planes):
        assert p.shape == (4, 4)
        assert np.allclose(p, float(i) / 5.0, atol=1e-4)


def test_normalize_1HW_broadcasts_like_HW():
    """[1,H,W] with B>1 broadcasts (single plane treated as union)."""
    t = torch.ones(1, 3, 3)
    planes = mask_adapter.normalize_mask_batch(t, 3)
    assert len(planes) == 3
    for p in planes:
        assert p.shape == (3, 3) and np.allclose(p, 1.0)


def test_normalize_none_returns_empty():
    """None (unconnected MASK socket) -> []."""
    assert mask_adapter.normalize_mask_batch(None, 4) == []


def test_normalize_zero_length_batch_empty():
    """[0,H,W] (SAM3 individual_masks with zero detections) -> []."""
    t = torch.zeros(0, 8, 8)
    assert mask_adapter.normalize_mask_batch(t, 1) == []


def test_normalize_uint8_scaled():
    """uint8 0..255 mask is scaled to 0..1."""
    m = np.zeros((2, 4, 4), dtype=np.uint8)
    m[0] = 255
    m[1] = 0
    planes = mask_adapter.normalize_mask_batch(m, 2)
    assert np.allclose(planes[0], 1.0)
    assert np.allclose(planes[1], 0.0)
    assert planes[0].dtype == np.float32


def test_normalize_bool_dtype():
    """Boolean mask maps False/True -> 0.0/1.0."""
    m = np.zeros((2, 3, 3), dtype=bool)
    m[0, 1, 1] = True
    planes = mask_adapter.normalize_mask_batch(m, 2)
    assert np.isclose(planes[0][1, 1], 1.0)
    assert np.isclose(planes[0][0, 0], 0.0)
    assert np.allclose(planes[1], 0.0)


def test_normalize_values_clipped():
    """Out-of-range float values are clipped to [0,1]; NaN/Inf sanitised."""
    m = np.array([[[2.0, -1.0], [np.nan, np.inf]]], dtype=np.float32)  # [1,2,2]
    planes = mask_adapter.normalize_mask_batch(m, 1)
    p = planes[0]
    assert p.shape == (2, 2)
    assert np.isclose(p[0, 0], 1.0)   # 2.0 -> 1
    assert np.isclose(p[0, 1], 0.0)   # -1.0 -> 0
    assert np.isclose(p[1, 0], 0.0)   # NaN -> 0
    assert np.isclose(p[1, 1], 1.0)   # +Inf -> 1


def test_normalize_bad_B_defaults_to_one():
    """A non-int / non-positive B is treated as 1 (defensive)."""
    m = np.ones((3, 3), dtype=np.float32)
    assert len(mask_adapter.normalize_mask_batch(m, 0)) == 1
    assert len(mask_adapter.normalize_mask_batch(m, None)) == 1  # type: ignore[arg-type]


def test_normalize_bad_4d_raises():
    """A non-squeezable 4-D shape is a hard error."""
    t = torch.zeros(2, 3, 4, 4)  # [B, C=3, H, W] — ambiguous, not a MASK layout
    raised = False
    try:
        mask_adapter.normalize_mask_batch(t, 2)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# mask_to_detections — connected components, min_area, merge_distance
# ---------------------------------------------------------------------------

def test_mask_to_detections_two_separate_blobs():
    """Two disjoint blobs -> two detections with correct xyxy (half-open)."""
    m = _blob_mask(10, 10, [(1, 3, 1, 3), (6, 9, 6, 9)])
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0)
    assert len(dets) == 2
    boxes = _det_boxes(dets)
    # blob1: rows 1..2, cols 1..2 -> xyxy (1,1,3,3); blob2: (6,6,9,9)
    assert (1, 1, 3, 3) in boxes
    assert (6, 6, 9, 9) in boxes
    # frame_index + source_index + mask_index wired
    assert all(d.frame_index == 0 for d in dets)
    assert sorted(d.source_index for d in dets) == [0, 1]
    assert all(d.mask_index == d.source_index for d in dets)
    assert all(d.score is None for d in dets)
    assert all(d.source_track_id is None for d in dets)


def test_mask_to_detections_min_area_drops_small():
    """A 1-pixel speck is dropped when min_area excludes it; big blob kept."""
    m = _blob_mask(10, 10, [(0, 1, 0, 1)])      # 1px speck
    m += _blob_mask(10, 10, [(4, 8, 4, 8)])     # 16px blob
    m = np.clip(m, 0.0, 1.0)
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=4, merge_distance=0)
    assert len(dets) == 1
    assert (4, 4, 8, 8) in _det_boxes(dets)


def test_mask_to_detections_min_area_zero_keeps_all():
    """min_area<=0 disables the area filter (keeps even 1px components)."""
    m = _blob_mask(6, 6, [(0, 1, 0, 1), (3, 4, 3, 4)])
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=0, merge_distance=0)
    assert len(dets) == 2


def test_mask_to_detections_merge_distance_joins_close():
    """Two close blobs with a 1px gap merge into one box when merge_distance>=1."""
    # blobs cols 1..2 and cols 4..5 on the same rows -> horizontal gap = 1px
    m = _blob_mask(8, 8, [(2, 5, 1, 3), (2, 5, 4, 6)])
    # Without merging: two detections.
    dets0 = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0)
    assert len(dets0) == 2
    # With merge_distance=1: a single enclosing box (1,2)-(6,5).
    dets1 = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=1)
    assert len(dets1) == 1
    assert (1, 2, 6, 5) in _det_boxes(dets1)


def test_mask_to_detections_merge_distance_respects_gap():
    """Blobs farther apart than merge_distance stay separate."""
    # gap between cols 1..2 (x2=3) and cols 6..7 (x1=6) is 6-3 = 3px
    m = _blob_mask(8, 10, [(2, 5, 1, 3), (2, 5, 6, 8)])
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=2)
    assert len(dets) == 2  # gap 3 > merge_distance 2 -> not merged


def test_mask_to_detections_threshold_strict_gt():
    """Foreground is value > threshold (strict)."""
    m = np.full((4, 4), 0.5, dtype=np.float32)
    # threshold 0.5 -> nothing strictly greater -> no detections
    assert mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0) == []
    # threshold 0.4 -> all pixels qualify -> one full-frame detection
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.4, min_area=1, merge_distance=0)
    assert len(dets) == 1
    assert (0, 0, 4, 4) in _det_boxes(dets)


def test_mask_to_detections_empty_mask():
    """All-zero mask -> no detections."""
    m = np.zeros((5, 5), dtype=np.float32)
    assert mask_adapter.mask_to_detections(m, 2, threshold=0.5, min_area=1, merge_distance=0) == []


def test_mask_to_detections_none_and_empty_array():
    """None or zero-size array -> []."""
    assert mask_adapter.mask_to_detections(None, 0, threshold=0.5, min_area=1, merge_distance=0) == []
    assert mask_adapter.mask_to_detections(
        np.zeros((0, 0), dtype=np.float32), 0, threshold=0.5, min_area=1, merge_distance=0
    ) == []


def test_mask_to_detections_accepts_torch_and_squeezes():
    """torch [1,H,W] and [H,W,1] inputs are squeezed and analysed."""
    base = _blob_mask(6, 6, [(1, 4, 1, 4)])
    t1 = torch.from_numpy(base).unsqueeze(0)       # [1,H,W]
    t2 = torch.from_numpy(base).unsqueeze(-1)      # [H,W,1]
    d1 = mask_adapter.mask_to_detections(t1, 0, threshold=0.5, min_area=1, merge_distance=0)
    d2 = mask_adapter.mask_to_detections(t2, 0, threshold=0.5, min_area=1, merge_distance=0)
    assert _det_boxes(d1) == {(1, 1, 4, 4)}
    assert _det_boxes(d2) == {(1, 1, 4, 4)}


def test_mask_to_detections_returns_detection_instances():
    """Output elements are real Detection dataclasses with xyxy guarantees."""
    m = _blob_mask(8, 8, [(2, 6, 2, 6)])
    dets = mask_adapter.mask_to_detections(m, 7, threshold=0.5, min_area=1, merge_distance=0)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.frame_index == 7
    assert d.x2 >= d.x1 and d.y2 >= d.y1
    assert d.width >= 1.0 and d.height >= 1.0


def test_mask_to_detections_diagonal_touch_not_connected():
    """4-connectivity: diagonally-touching pixels are SEPARATE components."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[1, 1] = 1.0
    m[2, 2] = 1.0  # touches (1,1) only diagonally
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0)
    assert len(dets) == 2  # not connected under 4-connectivity


def test_mask_to_detections_l_shape_single_component():
    """An L-shaped 4-connected region is one component with its full bbox."""
    m = np.zeros((5, 5), dtype=np.float32)
    m[1, 1] = 1.0
    m[2, 1] = 1.0
    m[2, 2] = 1.0  # 4-connected chain
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0)
    assert len(dets) == 1
    assert (1, 1, 3, 3) in _det_boxes(dets)


def test_mask_to_detections_numpy_matches_cv2_when_present():
    """The pure-numpy labeller produces the same boxes as the cv2 fast path.

    We force the numpy path by monkeypatching the cv2 helper to return None, run
    again with the real helper, and assert identical detection boxes regardless
    of whether cv2 is installed in the environment.
    """
    m = _blob_mask(12, 12, [(1, 4, 1, 4), (6, 11, 6, 11), (0, 2, 9, 11)])

    # Real path (may or may not use cv2 depending on the env).
    real = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0)

    # Forced numpy path.
    orig = mask_adapter._label_components_cv2
    mask_adapter._label_components_cv2 = lambda binary: None  # type: ignore[assignment]
    try:
        forced_numpy = mask_adapter.mask_to_detections(
            m, 0, threshold=0.5, min_area=1, merge_distance=0
        )
    finally:
        mask_adapter._label_components_cv2 = orig  # type: ignore[assignment]

    assert _det_boxes(real) == _det_boxes(forced_numpy)
    assert len(real) == 3


# ---------------------------------------------------------------------------
# masks_to_native_bboxes — native BOUNDING_BOX value (list[list[dict]] xywh)
# ---------------------------------------------------------------------------

def test_native_bboxes_shape_and_keys():
    """Output is list[list[dict]] with exactly the native xywh+score keys."""
    # [B=2, H, W]; frame 0 has one blob, frame 1 has none.
    m0 = _blob_mask(10, 10, [(2, 6, 3, 7)])
    m1 = np.zeros((10, 10), dtype=np.float32)
    batch = np.stack([m0, m1]).astype(np.float32)  # [2,10,10]
    out = mask_adapter.masks_to_native_bboxes(batch, threshold=0.5, min_area=1, merge_distance=0)
    assert isinstance(out, list) and len(out) == 2          # outer length == B
    assert isinstance(out[0], list) and len(out[0]) == 1
    assert out[1] == []                                      # empty frame kept
    d = out[0][0]
    assert set(d.keys()) == {"x", "y", "width", "height", "score"}
    assert all(isinstance(d[k], float) for k in d)
    # blob rows 2..5 cols 3..6 -> x=3,y=2,w=4,h=4
    assert d == {"x": 3.0, "y": 2.0, "width": 4.0, "height": 4.0, "score": 1.0}


def test_native_bboxes_single_HW_one_frame():
    """A bare [H,W] union mask yields exactly one frame (not broadcast)."""
    m = _blob_mask(8, 8, [(1, 4, 1, 4)])
    out = mask_adapter.masks_to_native_bboxes(m, threshold=0.5, min_area=1, merge_distance=0)
    assert len(out) == 1                       # outer length == 1, not some B
    assert len(out[0]) == 1
    assert out[0][0]["width"] == 3.0 and out[0][0]["height"] == 3.0


def test_native_bboxes_score_always_one():
    """Every emitted dict carries score == 1.0 (mask has no confidence)."""
    m = _blob_mask(6, 6, [(0, 2, 0, 2), (3, 5, 3, 5)])
    out = mask_adapter.masks_to_native_bboxes(m, threshold=0.5, min_area=1, merge_distance=0)
    assert len(out) == 1 and len(out[0]) == 2
    assert all(d["score"] == 1.0 for d in out[0])


def test_native_bboxes_none_returns_empty():
    """None mask -> []."""
    assert mask_adapter.masks_to_native_bboxes(None, threshold=0.5, min_area=1, merge_distance=0) == []


def test_native_bboxes_flattened_NHW_outer_length_N():
    """A flattened [N,H,W] mask yields N inner lists (one per plane)."""
    t = torch.zeros(3, 6, 6)
    t[0, 1:3, 1:3] = 1.0
    t[1] = 0.0
    t[2, 4:6, 4:6] = 1.0
    out = mask_adapter.masks_to_native_bboxes(t, threshold=0.5, min_area=1, merge_distance=0)
    assert len(out) == 3
    assert len(out[0]) == 1
    assert out[1] == []
    assert len(out[2]) == 1


def test_native_bboxes_B1HW_channel_first():
    """[B,1,H,W] is handled: outer length == B."""
    t = torch.zeros(2, 1, 8, 8)
    t[0, 0, 2:5, 2:5] = 1.0
    t[1, 0, 0:2, 6:8] = 1.0
    out = mask_adapter.masks_to_native_bboxes(t, threshold=0.5, min_area=1, merge_distance=0)
    assert len(out) == 2
    assert out[0][0] == {"x": 2.0, "y": 2.0, "width": 3.0, "height": 3.0, "score": 1.0}
    assert out[1][0] == {"x": 6.0, "y": 0.0, "width": 2.0, "height": 2.0, "score": 1.0}


def test_native_bboxes_min_area_and_merge_applied():
    """min_area + merge_distance flow through to the per-frame boxes."""
    # one big blob + one speck; specks dropped, big blob kept (no merge needed)
    m = _blob_mask(12, 12, [(0, 1, 0, 1), (4, 9, 4, 9)])
    out = mask_adapter.masks_to_native_bboxes(m, threshold=0.5, min_area=4, merge_distance=0)
    assert len(out) == 1 and len(out[0]) == 1
    assert out[0][0]["width"] == 5.0 and out[0][0]["height"] == 5.0

    # two close blobs -> merge into one with merge_distance
    m2 = _blob_mask(8, 8, [(2, 5, 1, 3), (2, 5, 4, 6)])
    out2 = mask_adapter.masks_to_native_bboxes(m2, threshold=0.5, min_area=1, merge_distance=1)
    assert len(out2[0]) == 1


def test_native_bboxes_xywh_matches_detection_xyxy():
    """The native xywh dicts are a faithful conversion of detection xyxy."""
    m = _blob_mask(10, 10, [(3, 8, 2, 9)])  # rows 3..7, cols 2..8
    dets = mask_adapter.mask_to_detections(m, 0, threshold=0.5, min_area=1, merge_distance=0)
    out = mask_adapter.masks_to_native_bboxes(m, threshold=0.5, min_area=1, merge_distance=0)
    d = dets[0]
    nb = out[0][0]
    assert nb["x"] == d.x1
    assert nb["y"] == d.y1
    assert nb["width"] == d.width
    assert nb["height"] == d.height


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
    print(f"mask_adapter tests: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFAILURES:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
