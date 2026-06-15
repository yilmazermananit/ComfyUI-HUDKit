"""Unit tests for yea_hud.track_adapter (PRIMARY tracking path).

Runnable BOTH ways:

* pytest:   python -m pytest tests/test_track_adapter.py -q
* script:   python tests/test_track_adapter.py   (runs every test, prints a
            summary, sys.exit(1) on any failure)

No ComfyUI, no network, no model files. Tiny synthetic torch/numpy fixtures
only. The packed-mask fixtures are built with a verbatim copy of the native
``pack_masks`` (comfy/ldm/sam3/tracker.py:194-198) so the round-trip through
``unpack_packed_masks`` is tested against the real bit layout, and the whole
SAM3_TRACK_DATA dict shape matches the source (tracker.py:1791/1802 +
nodes_sam3.py:311 which appends ``orig_size``).
"""

from __future__ import annotations

import os
import sys

# Make the package importable regardless of CWD when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from yea_hud import track_adapter as TA  # noqa: E402
from yea_hud.types import Detection  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _native_pack_masks(masks: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of comfy/ldm/sam3/tracker.py pack_masks (the source-of-truth
    bit layout). ``masks`` is ``[*, H, W]`` (W divisible by 8) -> ``[*, H, W//8]``
    uint8. Used to build realistic packed fixtures so tests bind to the real
    format, not our own reimplementation."""
    binary = masks > 0
    shifts = torch.arange(8, device=masks.device)
    return (binary.view(*masks.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def _native_unpack_masks(packed: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of comfy/ldm/sam3/tracker.py unpack_masks (source-of-truth)."""
    bits = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
    return (packed.unsqueeze(-1) & bits).bool().view(*packed.shape[:-1], -1)


def _rect_mask(h_m: int, w_m: int, x1: int, y1: int, x2: int, y2: int) -> torch.Tensor:
    """A bool ``[h_m, w_m]`` mask with a filled axis-aligned rect [x1,y1)..(x2,y2)
    in *mask* pixel coords (xyxy, half-open on the far edge)."""
    m = torch.zeros(h_m, w_m, dtype=torch.bool)
    m[y1:y2, x1:x2] = True
    return m


def _make_track_data(masks_FOHW: torch.Tensor, scores, orig_size, n_frames=None) -> dict:
    """Build a SAM3_TRACK_DATA dict from a dense ``[F, O, H_m, W_m]`` bool tensor
    by bit-packing it exactly like the native producer. ``W_m`` must be divisible
    by 8."""
    packed = _native_pack_masks(masks_FOHW)
    F = masks_FOHW.shape[0]
    return {
        "packed_masks": packed,
        "n_frames": F if n_frames is None else n_frames,
        "scores": list(scores),
        "orig_size": tuple(orig_size),
    }


# ---------------------------------------------------------------------------
# unpack_packed_masks
# ---------------------------------------------------------------------------

def test_unpack_matches_native_and_roundtrips():
    """Bit-exact with native unpack_masks AND inverts native pack_masks."""
    torch.manual_seed(1234)
    m = torch.rand(3, 4, 7, 24) > 0.5  # W=24 divisible by 8
    packed = _native_pack_masks(m)
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (3, 4, 7, 3)  # last dim 24//8 == 3

    native = _native_unpack_masks(packed)
    mine = TA.unpack_packed_masks(packed)
    assert mine.dtype == torch.bool
    assert torch.equal(mine, native)
    # round-trip pack -> unpack == original binary mask
    assert torch.equal(mine, m)
    # last dim expanded by 8
    assert tuple(mine.shape) == (3, 4, 7, 24)


def test_unpack_lsb_first_bit_order():
    """A single byte value 1 lights ONLY column 0 (LSB-first, like pack_masks)."""
    packed = torch.tensor([[1]], dtype=torch.uint8)  # shape [1,1]
    out = TA.unpack_packed_masks(packed)  # -> [1, 8] bool
    assert tuple(out.shape) == (1, 8)
    expected = torch.tensor([[True, False, False, False, False, False, False, False]])
    assert torch.equal(out, expected)

    packed2 = torch.tensor([[128]], dtype=torch.uint8)  # MSB only
    out2 = TA.unpack_packed_masks(packed2)
    expected2 = torch.tensor([[False, False, False, False, False, False, False, True]])
    assert torch.equal(out2, expected2)


def test_unpack_accepts_numpy_and_casts_dtype():
    """numpy input and a non-uint8 integer dtype are coerced, not rejected."""
    arr = np.array([[3]], dtype=np.int64)  # bits 0 and 1 set
    out = TA.unpack_packed_masks(arr)
    assert isinstance(out, torch.Tensor) and out.dtype == torch.bool
    expected = torch.tensor([[True, True, False, False, False, False, False, False]])
    assert torch.equal(out, expected)


def test_unpack_none_and_scalar_raise():
    """None and a 0-d scalar are programmer errors -> ValueError."""
    raised_none = False
    try:
        TA.unpack_packed_masks(None)
    except ValueError:
        raised_none = True
    assert raised_none

    raised_scalar = False
    try:
        TA.unpack_packed_masks(torch.tensor(5, dtype=torch.uint8))
    except ValueError:
        raised_scalar = True
    assert raised_scalar


# ---------------------------------------------------------------------------
# is_track_data
# ---------------------------------------------------------------------------

def test_is_track_data_true():
    td = {"packed_masks": None, "n_frames": 4, "scores": []}
    assert TA.is_track_data(td) is True
    td2 = {"packed_masks": torch.zeros(1, 1, 8, 1, dtype=torch.uint8), "n_frames": 1}
    assert TA.is_track_data(td2) is True


def test_is_track_data_false():
    assert TA.is_track_data(None) is False
    assert TA.is_track_data([]) is False
    assert TA.is_track_data({}) is False
    # Looks like a BOUNDING_BOX value (list[list[dict]]), not track data.
    assert TA.is_track_data([[{"x": 0, "y": 0, "width": 10, "height": 10}]]) is False
    # Missing one of the required keys.
    assert TA.is_track_data({"packed_masks": None}) is False
    assert TA.is_track_data({"n_frames": 3}) is False


# ---------------------------------------------------------------------------
# adapt_track_data — None / empty / malformed
# ---------------------------------------------------------------------------

def test_adapt_none_packed_masks():
    """packed_masks None -> ([[], ...]*n_frames, 0); identity slots = 0."""
    td = {"packed_masks": None, "n_frames": 5, "scores": []}
    per_frame, n_obj = TA.adapt_track_data(td)
    assert n_obj == 0
    assert len(per_frame) == 5
    assert all(f == [] for f in per_frame)


def test_adapt_none_packed_masks_zero_frames():
    td = {"packed_masks": None, "n_frames": 0, "scores": []}
    per_frame, n_obj = TA.adapt_track_data(td)
    assert per_frame == []
    assert n_obj == 0


def test_adapt_not_a_dict():
    per_frame, n_obj = TA.adapt_track_data([1, 2, 3])  # type: ignore[arg-type]
    assert per_frame == [] and n_obj == 0


def test_adapt_wrong_rank_degrades_gracefully():
    """A non-4D packed tensor degrades to empty frames (renderer falls back)."""
    td = {"packed_masks": torch.zeros(3, 8, 2, dtype=torch.uint8), "n_frames": 3, "scores": [0.9]}
    per_frame, n_obj = TA.adapt_track_data(td)
    assert n_obj == 0
    assert len(per_frame) == 3 and all(f == [] for f in per_frame)


def test_adapt_missing_n_frames_uses_packed_len():
    """No n_frames key -> output length follows packed frame dim."""
    masks = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    masks[0, 0] = _rect_mask(8, 8, 1, 1, 5, 5)
    td = {"packed_masks": _native_pack_masks(masks), "scores": [0.5], "orig_size": (8, 8)}
    per_frame, n_obj = TA.adapt_track_data(td)
    assert len(per_frame) == 4
    assert n_obj == 1


# ---------------------------------------------------------------------------
# adapt_track_data — single object, extent + scaling
# ---------------------------------------------------------------------------

def test_adapt_single_object_extent_no_scale():
    """1 frame, 1 object, orig_size == mask res -> exact pixel extent xyxy."""
    h_m, w_m = 16, 16
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 3, 4, 10, 12)  # x in [3,10), y in [4,12)
    td = _make_track_data(masks, scores=[0.77], orig_size=(h_m, w_m))

    per_frame, n_obj = TA.adapt_track_data(td)
    assert n_obj == 1
    assert len(per_frame) == 1
    dets = per_frame[0]
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection)
    # extent is half-open on far edge: cols 3..9 -> x1=3,x2=10 ; rows 4..11 -> y1=4,y2=12
    assert (d.x1, d.y1, d.x2, d.y2) == (3.0, 4.0, 10.0, 12.0)
    assert d.frame_index == 0
    assert d.source_index == 0
    assert d.source_track_id == 0   # persistent identity = object index
    assert d.mask_index == 0
    assert abs(d.score - 0.77) < 1e-6


def test_adapt_single_pixel_unit_box():
    """A single lit pixel -> unit box (never zero-area)."""
    h_m, w_m = 8, 8
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0, 5, 6] = True  # row 5, col 6
    td = _make_track_data(masks, scores=[1.0], orig_size=(h_m, w_m))
    per_frame, n_obj = TA.adapt_track_data(td)
    d = per_frame[0][0]
    assert (d.x1, d.y1, d.x2, d.y2) == (6.0, 5.0, 7.0, 6.0)


def test_adapt_scaling_to_orig_size():
    """Mask extent is scaled (H/H_m, W/W_m) into original pixels."""
    h_m, w_m = 16, 16
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 2, 4, 6, 8)  # x[2,6) y[4,8)
    # orig 4x in width, 8x in height
    H, W = 128, 64
    td = _make_track_data(masks, scores=[0.5], orig_size=(H, W))
    per_frame, _ = TA.adapt_track_data(td)
    d = per_frame[0][0]
    sx = W / w_m   # 4.0
    sy = H / h_m   # 8.0
    assert abs(d.x1 - 2 * sx) < 1e-6
    assert abs(d.x2 - 6 * sx) < 1e-6
    assert abs(d.y1 - 4 * sy) < 1e-6
    assert abs(d.y2 - 8 * sy) < 1e-6
    # clamped within frame
    assert 0 <= d.x1 <= W and 0 <= d.x2 <= W
    assert 0 <= d.y1 <= H and 0 <= d.y2 <= H


# ---------------------------------------------------------------------------
# adapt_track_data — occlusion / identity persistence (the headline feature)
# ---------------------------------------------------------------------------

def test_adapt_occlusion_skips_but_identity_persists():
    """Object absent on a frame (all-zero mask) is skipped there, but its index
    stays the same when it reappears — no renumbering."""
    h_m, w_m = 16, 16
    F, O = 3, 2
    masks = torch.zeros(F, O, h_m, w_m, dtype=torch.bool)
    # object 0 present on all 3 frames
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 5, 5)
    masks[1, 0] = _rect_mask(h_m, w_m, 2, 2, 6, 6)
    masks[2, 0] = _rect_mask(h_m, w_m, 3, 3, 7, 7)
    # object 1 present on frames 0 and 2, OCCLUDED on frame 1 (left all-zero)
    masks[0, 1] = _rect_mask(h_m, w_m, 10, 10, 14, 14)
    masks[2, 1] = _rect_mask(h_m, w_m, 11, 11, 15, 15)

    td = _make_track_data(masks, scores=[0.9, 0.4], orig_size=(h_m, w_m))
    per_frame, n_obj = TA.adapt_track_data(td)

    assert n_obj == 2
    assert len(per_frame) == 3

    # frame 0: both objects
    ids0 = sorted(d.source_track_id for d in per_frame[0])
    assert ids0 == [0, 1]
    # frame 1: only object 0 (object 1 occluded)
    ids1 = sorted(d.source_track_id for d in per_frame[1])
    assert ids1 == [0]
    # frame 2: object 1 reappears with SAME id 1 (not renumbered to 0/anything)
    ids2 = sorted(d.source_track_id for d in per_frame[2])
    assert ids2 == [0, 1]
    # the reappeared object keeps its own score (scores[1]) across the gap
    d_obj1_f2 = next(d for d in per_frame[2] if d.source_track_id == 1)
    assert abs(d_obj1_f2.score - 0.4) < 1e-6


def test_adapt_all_empty_frame_yields_empty_list():
    """A frame where every object is absent -> [] for that frame (occlusion)."""
    h_m, w_m = 8, 8
    masks = torch.zeros(2, 2, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 4, 4)
    masks[0, 1] = _rect_mask(h_m, w_m, 5, 5, 7, 7)
    # frame 1: both empty
    td = _make_track_data(masks, scores=[0.6, 0.6], orig_size=(h_m, w_m))
    per_frame, n_obj = TA.adapt_track_data(td)
    assert n_obj == 2
    assert len(per_frame[0]) == 2
    assert per_frame[1] == []


# ---------------------------------------------------------------------------
# adapt_track_data — area_threshold, scores edge cases, padding
# ---------------------------------------------------------------------------

def test_adapt_area_threshold_filters_small_blobs():
    """area_threshold drops objects whose lit area (mask res) is below it."""
    h_m, w_m = 16, 16
    masks = torch.zeros(1, 2, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 0, 0, 4, 4)  # area 16
    masks[0, 1, 8, 8] = True                        # area 1 (single pixel)
    td = _make_track_data(masks, scores=[0.9, 0.9], orig_size=(h_m, w_m))

    # default threshold 1 -> both kept
    pf_default, _ = TA.adapt_track_data(td)
    assert len(pf_default[0]) == 2

    # threshold 5 -> only the 16-px object survives
    pf_thr, _ = TA.adapt_track_data(td, area_threshold=5)
    assert len(pf_thr[0]) == 1
    assert pf_thr[0][0].source_track_id == 0


def test_adapt_area_threshold_coerced_to_min_one():
    """area_threshold <= 0 behaves like 1 (any lit pixel)."""
    h_m, w_m = 8, 8
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0, 3, 3] = True
    td = _make_track_data(masks, scores=[0.5], orig_size=(h_m, w_m))
    pf, _ = TA.adapt_track_data(td, area_threshold=0)
    assert len(pf[0]) == 1
    pf2, _ = TA.adapt_track_data(td, area_threshold=-10)
    assert len(pf2[0]) == 1


def test_adapt_scores_shorter_than_n_obj_gives_none():
    """If scores is shorter than n_obj, missing scores -> None (not crash)."""
    h_m, w_m = 8, 8
    masks = torch.zeros(1, 2, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 0, 0, 4, 4)
    masks[0, 1] = _rect_mask(h_m, w_m, 4, 4, 7, 7)
    td = _make_track_data(masks, scores=[0.9], orig_size=(h_m, w_m))  # only 1 score
    per_frame, n_obj = TA.adapt_track_data(td)
    assert n_obj == 2
    by_id = {d.source_track_id: d for d in per_frame[0]}
    assert abs(by_id[0].score - 0.9) < 1e-6
    assert by_id[1].score is None


def test_adapt_empty_scores_list():
    """scores == [] but masks present -> all scores None, detections still emit."""
    h_m, w_m = 8, 8
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 5, 5)
    td = {"packed_masks": _native_pack_masks(masks), "n_frames": 1, "scores": [], "orig_size": (h_m, w_m)}
    per_frame, n_obj = TA.adapt_track_data(td)
    assert n_obj == 1
    assert per_frame[0][0].score is None


def test_adapt_nan_score_becomes_none():
    h_m, w_m = 8, 8
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 5, 5)
    td = _make_track_data(masks, scores=[float("nan")], orig_size=(h_m, w_m))
    per_frame, _ = TA.adapt_track_data(td)
    assert per_frame[0][0].score is None


def test_adapt_reported_n_frames_pads_beyond_packed():
    """n_frames larger than packed frame dim -> extra frames are empty lists."""
    h_m, w_m = 8, 8
    masks = torch.zeros(2, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 5, 5)
    masks[1, 0] = _rect_mask(h_m, w_m, 2, 2, 6, 6)
    td = _make_track_data(masks, scores=[0.5], orig_size=(h_m, w_m), n_frames=5)
    per_frame, n_obj = TA.adapt_track_data(td)
    assert len(per_frame) == 5
    assert len(per_frame[0]) == 1 and len(per_frame[1]) == 1
    assert per_frame[2] == [] and per_frame[3] == [] and per_frame[4] == []


def test_adapt_missing_orig_size_falls_back_to_mask_res():
    """No orig_size -> render at mask resolution (1:1), no divide-by-unknown."""
    h_m, w_m = 16, 16
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 2, 3, 9, 11)
    td = {"packed_masks": _native_pack_masks(masks), "n_frames": 1, "scores": [0.5]}
    per_frame, _ = TA.adapt_track_data(td)
    d = per_frame[0][0]
    assert (d.x1, d.y1, d.x2, d.y2) == (2.0, 3.0, 9.0, 11.0)


def test_adapt_determinism():
    """Same input twice -> identical detection coordinates / ids / scores."""
    h_m, w_m = 16, 16
    torch.manual_seed(7)
    masks = torch.rand(3, 3, h_m, w_m) > 0.6
    td = _make_track_data(masks, scores=[0.1, 0.2, 0.3], orig_size=(48, 32))
    a, na = TA.adapt_track_data(td)
    b, nb = TA.adapt_track_data(td)
    assert na == nb
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert len(fa) == len(fb)
        for da, dbb in zip(fa, fb):
            assert da.xyxy == dbb.xyxy
            assert da.source_track_id == dbb.source_track_id
            assert da.score == dbb.score


def test_adapt_does_not_mutate_input():
    h_m, w_m = 8, 8
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 5, 5)
    packed = _native_pack_masks(masks)
    packed_clone = packed.clone()
    td = {"packed_masks": packed, "n_frames": 1, "scores": [0.5], "orig_size": (h_m, w_m)}
    TA.adapt_track_data(td)
    assert torch.equal(packed, packed_clone)  # input tensor untouched
    assert set(td.keys()) == {"packed_masks", "n_frames", "scores", "orig_size"}


# ---------------------------------------------------------------------------
# summarize_track_data
# ---------------------------------------------------------------------------

def test_summarize_full_dict():
    h_m, w_m = 16, 16
    F, O = 3, 2
    masks = torch.zeros(F, O, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 1, 1, 5, 5)
    masks[0, 1] = _rect_mask(h_m, w_m, 8, 8, 12, 12)
    masks[1, 0] = _rect_mask(h_m, w_m, 2, 2, 6, 6)  # only obj 0 on frame 1
    masks[2, 1] = _rect_mask(h_m, w_m, 9, 9, 13, 13)  # only obj 1 on frame 2
    td = _make_track_data(masks, scores=[0.91, 0.42], orig_size=(64, 48))

    s = TA.summarize_track_data(td)
    assert s["is_track_data"] is True
    assert s["n_frames"] == 3
    assert s["n_obj"] == 2
    assert s["packed_masks_shape"] == (3, 2, 16, 2)  # 16//8 == 2
    assert s["packed_masks_dtype"] == "torch.uint8"
    assert s["mask_resolution"] == (16, 16)
    assert s["orig_size"] == (64, 48)
    assert s["n_scores"] == 2
    assert s["scores"] == [0.91, 0.42]
    # visibility: frame0 -> 2 present, frame1 -> 1, frame2 -> 1
    assert s["visibility_per_frame"] == [2, 1, 1]
    assert "note" not in s


def test_summarize_none_packed():
    td = {"packed_masks": None, "n_frames": 7, "scores": []}
    s = TA.summarize_track_data(td)
    assert s["is_track_data"] is True
    assert s["n_frames"] == 7
    assert s["n_obj"] == 0
    assert s["packed_masks_shape"] is None
    assert s["visibility_per_frame"] == []
    assert "note" in s and "None" in s["note"]


def test_summarize_non_dict():
    s = TA.summarize_track_data([1, 2, 3])
    assert s["is_track_data"] is False
    assert "note" in s and "list" in s["note"]


def test_summarize_wrong_rank_notes_it():
    td = {"packed_masks": torch.zeros(3, 8, 2, dtype=torch.uint8), "n_frames": 3, "scores": [0.1]}
    s = TA.summarize_track_data(td)
    assert s["packed_masks_shape"] == (3, 8, 2)
    assert "note" in s and "rank" in s["note"]
    assert s["n_obj"] == 0


def test_summarize_missing_orig_size_reflects_mask_res():
    h_m, w_m = 16, 16
    masks = torch.zeros(1, 1, h_m, w_m, dtype=torch.bool)
    masks[0, 0] = _rect_mask(h_m, w_m, 0, 0, 4, 4)
    td = {"packed_masks": _native_pack_masks(masks), "n_frames": 1, "scores": [0.5]}
    s = TA.summarize_track_data(td)
    assert s["orig_size"] == (16, 16)  # falls back to mask resolution
    assert s["mask_resolution"] == (16, 16)


def test_summarize_caps_scores_and_visibility():
    """Long clip / many objects: scores capped at 64, visibility capped at 64."""
    h_m, w_m = 8, 8
    F = 100
    masks = torch.zeros(F, 1, h_m, w_m, dtype=torch.bool)
    for t in range(F):
        masks[t, 0] = _rect_mask(h_m, w_m, 0, 0, 2, 2)
    td = _make_track_data(masks, scores=[0.5] * 200, orig_size=(h_m, w_m))
    s = TA.summarize_track_data(td)
    assert s["n_scores"] == 200
    assert len(s["scores"]) == 64
    assert len(s["visibility_per_frame"]) == 64


# ---------------------------------------------------------------------------
# script runner (works without pytest)
# ---------------------------------------------------------------------------

def _all_tests():
    g = globals()
    return [(n, g[n]) for n in sorted(g) if n.startswith("test_") and callable(g[n])]


def main() -> int:
    passed = 0
    failed = 0
    failures = []
    for name, fn in _all_tests():
        try:
            fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001 — report-and-continue test runner
            failed += 1
            import traceback
            failures.append((name, traceback.format_exc()))
            print(f"FAIL  {name}: {exc!r}")
        else:
            print(f"ok    {name}")

    print("-" * 60)
    print(f"track_adapter: {passed} passed, {failed} failed, {passed + failed} total")
    if failures:
        print("\n=== failure tracebacks ===")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
