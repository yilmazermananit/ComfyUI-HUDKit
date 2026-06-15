"""Unit tests for yea_hud.tracker (brief §10, §11, §25.3).

Covers the tracker-relevant brief §25 cases:

* §25.3 static / slow / fast / crossing / 3-frame occlusion / order-change /
  enter / exit / seed-determinism — all on the GREEDY (IoU+centroid+area) path.
* The PASSTHROUGH path: detections carrying ``source_track_id`` bypass IoU and
  keep native identity (and survive occlusion via index persistence).
* EMA smoothing (geometry.ema_box) on both paths.
* Style assignment: assigned ONCE per track, deterministic from hash(seed, id),
  forced style_mode behaviour, display_probability gate.
* micro_variation determinism and bounds.

Runnable BOTH ways:

* pytest:   python -m pytest tests/test_tracker.py -q
* script:   python tests/test_tracker.py     (runs every test, prints a summary,
            sys.exit(1) on any failure)

No ComfyUI, no network, no model files. Tiny synthetic detections only.
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
import torch  # noqa: E402  (verify 0-d tensor scalars survive into detections)

from yea_hud import geometry as G  # noqa: E402
from yea_hud.tracker import Tracker, micro_variation, _track_id_sort_key  # noqa: E402
from yea_hud.types import (  # noqa: E402
    DEFAULT_PALETTE,
    PRIMITIVE_STYLES,
    Detection,
    Track,
    TrackStyle,
)
from yea_hud.palette import resolve_palette  # noqa: E402


EPS = 1e-6
W, H = 640, 480
PALETTE = resolve_palette(None)  # DEFAULT_PALETTE as RGB triples


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _det(frame, sidx, x1, y1, x2, y2, *, score=None, label=None, sid=None):
    """Build a Detection (xyxy)."""
    return Detection(
        frame_index=frame,
        source_index=sidx,
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        score=score,
        label=label,
        source_track_id=sid,
    )


def _mk(**overrides):
    """Construct a Tracker with sensible defaults; override any kwarg."""
    kw = dict(
        seed=77,
        iou_threshold=0.25,
        max_centroid_ratio=0.35,
        memory_frames=6,
        smoothing=0.0,          # raw by default so position assertions are exact
        palette=PALETTE,
        n_frame_assets=0,
        n_target_assets=0,
        style_mode="mixed",
        display_probability=1.0,  # show all by default (deterministic tests)
        fill_probability=0.25,
        target_probability=0.7,
        W=W,
        H=H,
        adaptive=False,
    )
    kw.update(overrides)
    return Tracker(**kw)


def _approx(a, b, eps=EPS):
    return abs(float(a) - float(b)) <= eps


def _ids(pairs):
    """Sorted list of track_ids from update() output."""
    return [t.track_id for t, _d in pairs]


# ===========================================================================
# §25.3 STATIC — one stationary object keeps a single id across frames
# ===========================================================================

def test_static_single_id():
    tr = _mk()
    box = (100, 100, 200, 260)
    last_id = None
    for f in range(10):
        pairs = tr.update(f, [_det(f, 0, *box)])
        assert len(pairs) == 1
        tid = pairs[0][0].track_id
        if last_id is None:
            last_id = tid
        assert tid == last_id, "static object must keep one id"
    # only one live track ever
    assert len(tr.tracks) == 1


def test_static_smoothed_box_converges_to_box():
    # With heavy smoothing the static object's smoothed box must still equal the
    # true box (EMA of a constant is that constant).
    tr = _mk(smoothing=0.8)
    box = (50, 50, 150, 150)
    for f in range(20):
        pairs = tr.update(f, [_det(f, 0, *box)])
    t = pairs[0][0]
    assert all(_approx(a, b, 1e-3) for a, b in zip(t.smoothed_box, box))


# ===========================================================================
# §25.3 SLOW — slowly drifting object keeps id; smoothing lags raw box
# ===========================================================================

def test_slow_drift_keeps_id():
    tr = _mk()
    first = None
    for f in range(12):
        x = 100 + f * 4  # 4 px/frame drift — well inside any gate
        pairs = tr.update(f, [_det(f, 0, x, 100, x + 100, 260)])
        assert len(pairs) == 1
        if first is None:
            first = pairs[0][0].track_id
        assert pairs[0][0].track_id == first
    assert len(tr.tracks) == 1


def test_slow_smoothing_lags_then_catches_up():
    tr = _mk(smoothing=0.7)  # alpha ~0.3
    raw_x1 = None
    for f in range(6):
        x = 100 + f * 10
        pairs = tr.update(f, [_det(f, 0, x, 100, x + 100, 260)])
        raw_x1 = x
    t = pairs[0][0]
    # smoothed x1 should trail the raw x1 (still catching up under lag)
    assert t.smoothed_box[0] < raw_x1
    assert t.last_box[0] == float(raw_x1)  # last_box is the RAW box, not smoothed


# ===========================================================================
# §25.3 FAST — big jump still re-acquires same id when adaptive / within gate
# ===========================================================================

def test_fast_motion_within_gate_keeps_id():
    # Jump ~120 px/frame. frame diag ~800, max_centroid_ratio 0.35 -> ~280 px gate.
    tr = _mk(smoothing=0.5)
    first = None
    for f in range(5):
        x = 50 + f * 120
        pairs = tr.update(f, [_det(f, 0, x, 100, x + 90, 250)])
        if first is None:
            first = pairs[0][0].track_id
        assert pairs[0][0].track_id == first, "fast-but-within-gate must keep id"


def test_fast_adaptive_snaps_closer_than_constant():
    # Under adaptive smoothing a big motion should snap the smoothed box closer to
    # the raw box than the same constant alpha would.
    seq = [(50, 100), (300, 100)]  # 250 px jump
    box_w, box_h = 90, 150

    def run(adaptive):
        tr = _mk(smoothing=0.7, adaptive=adaptive)
        pairs = None
        for f, (x, y) in enumerate(seq):
            pairs = tr.update(f, [_det(f, 0, x, y, x + box_w, y + box_h)])
        return pairs[0][0].smoothed_box[0]

    const_x1 = run(False)
    adapt_x1 = run(True)
    raw_x1 = 300.0
    # adaptive smoothed x1 must be nearer the raw 300 than the constant one
    assert abs(adapt_x1 - raw_x1) < abs(const_x1 - raw_x1)


def test_teleport_beyond_gate_spawns_new_id():
    # A jump far beyond the centroid gate must NOT be matched -> new id, old dies.
    tr = _mk(memory_frames=0)  # die immediately when unmatched
    p0 = tr.update(0, [_det(0, 0, 0, 0, 40, 40)])
    id0 = p0[0][0].track_id
    # teleport to the opposite corner (well beyond 0.35*diag)
    p1 = tr.update(1, [_det(1, 0, 600, 440, 640, 480)])
    id1 = p1[0][0].track_id
    assert id1 != id0, "teleport beyond gate must be a new identity"
    # old track culled (memory_frames=0)
    assert id0 not in tr.tracks


# ===========================================================================
# §25.3 CROSSING — two objects swap sides; ids stay attached by IoU+centroid
# ===========================================================================

def test_crossing_two_objects_keep_distinct_ids():
    tr = _mk(smoothing=0.0)
    # A moves left->right, B moves right->left; they approach but do not occupy
    # the exact same pixel on any single frame (deterministic greedy match).
    frames = []
    for f in range(7):
        ax = 40 + f * 60      # 40 .. 400
        bx = 460 - f * 60     # 460 .. 100
        frames.append([
            _det(f, 0, ax, 200, ax + 70, 360),
            _det(f, 1, bx, 200, bx + 70, 360),
        ])
    ids_seen = []
    for f, dets in enumerate(frames):
        pairs = tr.update(f, dets)
        assert len(pairs) == 2, "both objects visible every crossing frame"
        ids_seen.append({p[0].track_id for p in pairs})
    # exactly two distinct ids across the whole crossing, both alive throughout
    all_ids = set().union(*ids_seen)
    assert len(all_ids) == 2, "crossing must not invent extra identities"
    for s in ids_seen:
        assert s == all_ids


# ===========================================================================
# §25.3 3-FRAME OCCLUSION — object disappears <= memory_frames, same id resumes
# ===========================================================================

def test_three_frame_occlusion_same_id_resumes():
    tr = _mk(memory_frames=6, smoothing=0.0)
    box = (200, 150, 320, 360)
    p = tr.update(0, [_det(0, 0, *box)])
    orig = p[0][0].track_id
    # frames 1,2,3: occluded (no detections)
    for f in (1, 2, 3):
        out = tr.update(f, [])
        assert out == [], "no visible pairs during occlusion"
        assert orig in tr.tracks, "track must survive within memory_frames"
    # reappears frame 4, slightly moved -> SAME id (still alive, IoU/centroid match)
    p4 = tr.update(4, [_det(4, 0, 210, 150, 330, 360)])
    assert p4[0][0].track_id == orig, "occluded track must keep its id on return"
    assert tr.tracks[orig].missing_frames == 0


def test_occlusion_beyond_memory_is_new_id():
    tr = _mk(memory_frames=2, smoothing=0.0)
    box = (200, 150, 320, 360)
    p = tr.update(0, [_det(0, 0, *box)])
    orig = p[0][0].track_id
    # missing for 3 frames (1,2,3) > memory_frames(2) -> culled after frame 3
    for f in (1, 2, 3):
        tr.update(f, [])
    assert orig not in tr.tracks, "track must be culled past memory_frames TTL"
    p4 = tr.update(4, [_det(4, 0, *box)])
    assert p4[0][0].track_id != orig, "return after TTL is a fresh id"
    # and the fresh id is strictly greater (monotonic, never reused)
    assert p4[0][0].track_id > orig


# ===========================================================================
# §25.3 ORDER-CHANGE — detection list order shuffles; ids stay attached to boxes
# ===========================================================================

def test_detection_order_change_does_not_swap_ids():
    tr = _mk(smoothing=0.0)
    # frame 0: A at left, B at right
    p0 = tr.update(0, [
        _det(0, 0, 40, 200, 110, 360),     # A
        _det(0, 1, 460, 200, 530, 360),    # B
    ])
    id_by_cx = {}
    for t, d in p0:
        id_by_cx[round(d.cx)] = t.track_id
    a_id = id_by_cx[round((40 + 110) / 2)]
    b_id = id_by_cx[round((460 + 530) / 2)]
    assert a_id != b_id
    # frame 1: SAME boxes but listed in REVERSED order (B first, A second)
    p1 = tr.update(1, [
        _det(1, 0, 462, 200, 532, 360),    # B (listed first now)
        _det(1, 1, 42, 200, 112, 360),     # A (listed second now)
    ])
    new_by_cx = {}
    for t, d in p1:
        # bucket by side
        side = "right" if d.cx > W / 2 else "left"
        new_by_cx[side] = t.track_id
    assert new_by_cx["left"] == a_id, "left box must keep A's id despite list order"
    assert new_by_cx["right"] == b_id, "right box must keep B's id despite list order"


def test_update_output_sorted_by_track_id():
    tr = _mk(smoothing=0.0)
    pairs = tr.update(0, [
        _det(0, 0, 400, 50, 470, 200),
        _det(0, 1, 40, 50, 110, 200),
        _det(0, 2, 220, 50, 290, 200),
    ])
    ids = [t.track_id for t, _ in pairs]
    assert ids == sorted(ids), "update() must return pairs sorted by track_id"


# ===========================================================================
# §25.3 ENTER — a new object appears mid-sequence -> new monotonic id
# ===========================================================================

def test_enter_new_object_gets_new_monotonic_id():
    tr = _mk(smoothing=0.0)
    a = (40, 200, 110, 360)
    p0 = tr.update(0, [_det(0, 0, *a)])
    a_id = p0[0][0].track_id
    # frame 1: A persists, B enters
    b = (460, 200, 530, 360)
    p1 = tr.update(1, [_det(1, 0, *a), _det(1, 1, *b)])
    ids1 = sorted(t.track_id for t, _ in p1)
    assert a_id in ids1
    new_ids = [i for i in ids1 if i != a_id]
    assert len(new_ids) == 1
    assert new_ids[0] > a_id, "entering object gets a fresh, larger id"


# ===========================================================================
# §25.3 EXIT — an object leaves -> aged then culled; survivor keeps id
# ===========================================================================

def test_exit_object_culled_survivor_keeps_id():
    tr = _mk(memory_frames=2, smoothing=0.0)
    a = (40, 200, 110, 360)
    b = (460, 200, 530, 360)
    p0 = tr.update(0, [_det(0, 0, *a), _det(0, 1, *b)])
    ids0 = sorted(t.track_id for t, _ in p0)
    a_id, b_id = ids0[0], ids0[1]
    # B exits from frame 1 onward; A stays
    for f in range(1, 5):
        tr.update(f, [_det(f, 0, *a)])
    assert a_id in tr.tracks, "surviving object keeps its id"
    assert b_id not in tr.tracks, "exited object culled past TTL"


# ===========================================================================
# §25.3 SEED-DETERMINISM — same seed => identical styles & ids; diff seed differs
# ===========================================================================

def _run_scene(seed, style_mode="mixed", n_frame_assets=0):
    tr = _mk(seed=seed, style_mode=style_mode, n_frame_assets=n_frame_assets,
             display_probability=0.85)
    # three objects, static, several frames
    boxes = [(40, 60, 140, 300), (260, 60, 360, 300), (480, 60, 580, 300)]
    out = []
    for f in range(4):
        dets = [_det(f, i, *b) for i, b in enumerate(boxes)]
        pairs = tr.update(f, dets)
        out.append([(t.track_id, t.color_id, t.style.primitive_style,
                     t.style.show, t.style.use_fill, t.style.use_target,
                     t.style.caption_side, round(t.style.glitch_phase, 6),
                     t.style.frame_asset_id) for t, _ in pairs])
    return out


def test_seed_determinism_identical_runs():
    a = _run_scene(77)
    b = _run_scene(77)
    assert a == b, "same seed must produce byte-identical styles & ids"


def test_different_seed_changes_styling():
    a = _run_scene(77)
    b = _run_scene(78)
    # ids are positional/monotonic so they match; the STYLE tuple must differ
    # somewhere across the three objects.
    flat_a = [row for fr in a for row in fr]
    flat_b = [row for fr in b for row in fr]
    assert flat_a != flat_b, "a different seed must change the HUD styling"


def test_style_assigned_once_stable_across_frames():
    tr = _mk(seed=5, display_probability=0.85)
    box = (100, 100, 220, 320)
    styles = []
    for f in range(8):
        pairs = tr.update(f, [_det(f, 0, *box)])
        styles.append(pairs[0][0].style)
    first = styles[0]
    for s in styles[1:]:
        assert s is first, "TrackStyle object must be the SAME (assigned once)"


def test_assign_style_is_cached_and_pure():
    tr = _mk(seed=9)
    s1 = tr.assign_style(3)
    s2 = tr.assign_style(3)
    assert s1 is s2, "assign_style must cache per track_id"
    # color matches palette.color_for_track contract
    from yea_hud.palette import color_for_track
    idx, _rgb = color_for_track(PALETTE, 9, 3)
    assert s1.color_id == idx


# ===========================================================================
# STYLE-MODE forcing
# ===========================================================================

def test_style_mode_corners_forces_corners_no_fill_no_target():
    tr = _mk(style_mode="corners")
    for tid in range(12):
        s = tr.assign_style(tid)
        assert s.primitive_style == "corners"
        assert s.use_fill is False
        assert s.use_target is False
        assert s.frame_asset_id == -1


def test_style_mode_solid_forces_fill():
    tr = _mk(style_mode="solid")
    for tid in range(12):
        s = tr.assign_style(tid)
        assert s.primitive_style == "solid"
        assert s.use_fill is True
        assert s.use_target is False


def test_style_mode_target_only_forces_target():
    tr = _mk(style_mode="target_only")
    for tid in range(12):
        s = tr.assign_style(tid)
        assert s.primitive_style == "target_only"
        assert s.use_target is True
        assert s.use_fill is False


def test_style_mode_labels_only_forces_caption():
    tr = _mk(style_mode="labels_only")
    for tid in range(12):
        s = tr.assign_style(tid)
        assert s.use_caption is True
        assert s.use_fill is False
        assert s.use_target is False


def test_style_mode_outline_forces_outline():
    tr = _mk(style_mode="outline")
    for tid in range(12):
        assert tr.assign_style(tid).primitive_style == "outline"


def test_assets_only_uses_frame_asset_when_available():
    tr = _mk(style_mode="assets_only", n_frame_assets=3)
    seen_frame_asset = False
    for tid in range(12):
        s = tr.assign_style(tid)
        assert s.primitive_style == "frame_asset"
        assert 0 <= s.frame_asset_id < 3
        seen_frame_asset = True
    assert seen_frame_asset


def test_assets_only_degrades_without_assets():
    # No eligible frame assets -> 'assets_only' must still draw something (vector).
    tr = _mk(style_mode="assets_only", n_frame_assets=0)
    for tid in range(12):
        s = tr.assign_style(tid)
        assert s.primitive_style in PRIMITIVE_STYLES
        assert s.primitive_style != "frame_asset"
        assert s.frame_asset_id == -1


def test_procedural_only_never_uses_frame_asset():
    tr = _mk(style_mode="procedural_only", n_frame_assets=4)
    for tid in range(16):
        s = tr.assign_style(tid)
        assert s.frame_asset_id == -1
        assert s.primitive_style != "frame_asset"


def test_mixed_primitive_within_vocabulary():
    tr = _mk(style_mode="mixed")
    for tid in range(40):
        s = tr.assign_style(tid)
        assert s.primitive_style in PRIMITIVE_STYLES


def test_frame_asset_id_within_count_or_minus_one():
    tr = _mk(style_mode="mixed", n_frame_assets=2, n_target_assets=3)
    for tid in range(40):
        s = tr.assign_style(tid)
        assert -1 <= s.frame_asset_id < 2
        assert -1 <= s.target_asset_id < 3


def test_display_probability_zero_hides_all():
    tr = _mk(display_probability=0.0)
    for tid in range(30):
        assert tr.assign_style(tid).show is False


def test_display_probability_one_shows_all():
    tr = _mk(display_probability=1.0)
    for tid in range(30):
        assert tr.assign_style(tid).show is True


def test_display_probability_partial_is_mixed_but_deterministic():
    a = [_mk(seed=11, display_probability=0.5).assign_style(t).show for t in range(50)]
    b = [_mk(seed=11, display_probability=0.5).assign_style(t).show for t in range(50)]
    assert a == b, "show gate must be deterministic for a fixed seed"
    # genuinely partial (not all-true / all-false) for a reasonable sample
    assert any(a) and not all(a)


# ===========================================================================
# PASSTHROUGH PATH — source_track_id bypasses IoU
# ===========================================================================

def test_passthrough_identity_from_source_track_id():
    tr = _mk(smoothing=0.0)
    # Two native objects with explicit source_track_ids 5 and 9.
    for f in range(6):
        pairs = tr.update(f, [
            _det(f, 0, 40, 60, 140, 300, sid=5),
            _det(f, 1, 400, 60, 500, 300, sid=9),
        ])
        ids = sorted(t.track_id for t, _ in pairs)
        assert ids == [5, 9], "passthrough must use source_track_id verbatim"
    assert set(tr.tracks.keys()) == {5, 9}


def test_passthrough_ignores_iou_even_on_teleport():
    # Native id is sticky regardless of position: a huge jump keeps the same id
    # (no centroid gate on the passthrough path).
    tr = _mk(smoothing=0.0, max_centroid_ratio=0.01)  # tiny gate would block greedy
    p0 = tr.update(0, [_det(0, 0, 0, 0, 40, 40, sid=3)])
    assert p0[0][0].track_id == 3
    # teleport to far corner; still id 3 because identity is native, not IoU.
    p1 = tr.update(1, [_det(1, 0, 600, 440, 640, 480, sid=3)])
    assert p1[0][0].track_id == 3
    # and the box followed (EMA raw because smoothing=0)
    assert _approx(p1[0][0].smoothed_box[0], 600)


def test_passthrough_occlusion_identity_survives_by_index():
    # Native data: object 7 missing for a few frames, then returns with same sid.
    tr = _mk(memory_frames=6, smoothing=0.0)
    tr.update(0, [_det(0, 0, 100, 100, 200, 260, sid=7)])
    for f in (1, 2, 3):
        out = tr.update(f, [])  # occluded
        assert out == []
        assert 7 in tr.tracks
    p4 = tr.update(4, [_det(4, 0, 120, 100, 220, 260, sid=7)])
    assert p4[0][0].track_id == 7


def test_passthrough_string_source_track_id():
    tr = _mk(smoothing=0.0)
    pairs = tr.update(0, [
        _det(0, 0, 40, 60, 140, 300, sid="objA"),
        _det(0, 1, 400, 60, 500, 300, sid="objB"),
    ])
    ids = {t.track_id for t, _ in pairs}
    assert ids == {"objA", "objB"}
    # styles still deterministic for string ids
    sa = tr.assign_style("objA")
    sb = tr.assign_style("objB")
    assert isinstance(sa, TrackStyle) and isinstance(sb, TrackStyle)


def test_passthrough_style_keyed_by_source_id_not_order():
    # The same native id must get the same style regardless of which order it was
    # first seen in (style keyed on the id, via hash(seed, id)).
    trA = _mk(seed=42, smoothing=0.0)
    trA.update(0, [_det(0, 0, 40, 60, 140, 300, sid=2),
                   _det(0, 1, 400, 60, 500, 300, sid=8)])
    style2_A = trA.tracks[2].style

    trB = _mk(seed=42, smoothing=0.0)
    # reversed listing / different source_index, same ids
    trB.update(0, [_det(0, 9, 400, 60, 500, 300, sid=8),
                   _det(0, 3, 40, 60, 140, 300, sid=2)])
    style2_B = trB.tracks[2].style

    assert (style2_A.color_id, style2_A.primitive_style, style2_A.glitch_phase) == \
           (style2_B.color_id, style2_B.primitive_style, style2_B.glitch_phase)


def test_mixed_frame_native_and_anon():
    # A frame where one det has a source_track_id and one does not -> the frame is
    # treated as native; the anon det still gets tracked (deterministic synthetic id).
    tr = _mk(smoothing=0.0)
    pairs = tr.update(0, [
        _det(0, 0, 40, 60, 140, 300, sid=4),
        _det(0, 1, 400, 60, 500, 300, sid=None),
    ])
    ids = {t.track_id for t, _ in pairs}
    assert 4 in ids
    assert len(ids) == 2, "anon detection in a native frame must still be tracked"


# ===========================================================================
# EMA smoothing wiring (geometry.ema_box) — sanity beyond static/slow
# ===========================================================================

def test_smoothing_seeds_on_first_frame_raw():
    tr = _mk(smoothing=0.9)  # very smooth
    box = (100, 100, 200, 300)
    p = tr.update(0, [_det(0, 0, *box)])
    # first sight: smoothed == raw (ema seeds on previous=None)
    assert all(_approx(a, b) for a, b in zip(p[0][0].smoothed_box, box))


def test_smoothing_zero_equals_raw_every_frame():
    tr = _mk(smoothing=0.0)
    for f in range(5):
        x = 100 + f * 15
        box = (x, 100, x + 80, 240)
        p = tr.update(f, [_det(f, 0, *box)])
        assert all(_approx(a, b) for a, b in zip(p[0][0].smoothed_box, box))


# ===========================================================================
# Degenerate / defensive inputs
# ===========================================================================

def test_empty_and_none_detections():
    tr = _mk()
    assert tr.update(0, []) == []
    assert tr.update(1, None) == []
    assert len(tr.tracks) == 0


def test_degenerate_box_dropped():
    tr = _mk()
    # zero-width box must be dropped (no track created)
    p = tr.update(0, [_det(0, 0, 100, 100, 100, 260)])
    assert p == []
    assert len(tr.tracks) == 0


def test_reversed_box_is_normalized_and_tracked():
    tr = _mk(smoothing=0.0)
    # x2<x1, y2<y1 -> normalized internally, still a valid track
    p = tr.update(0, [_det(0, 0, 200, 300, 100, 100)])
    assert len(p) == 1
    sm = p[0][0].smoothed_box
    assert sm[0] <= sm[2] and sm[1] <= sm[3]


def test_torch_scalar_coords_survive():
    # Detection coords supplied as 0-d torch tensors must flow through cleanly.
    tr = _mk(smoothing=0.0)
    d = Detection(
        frame_index=0, source_index=0,
        x1=torch.tensor(10.0), y1=torch.tensor(20.0),
        x2=torch.tensor(110.0), y2=torch.tensor(220.0),
    )
    p = tr.update(0, [d])
    assert len(p) == 1
    assert _approx(p[0][0].smoothed_box[0], 10.0)


def test_numpy_scalar_coords_survive():
    tr = _mk(smoothing=0.0)
    d = Detection(
        frame_index=0, source_index=0,
        x1=np.float32(10), y1=np.float32(20),
        x2=np.float32(110), y2=np.float32(220),
    )
    p = tr.update(0, [d])
    assert len(p) == 1
    assert _approx(p[0][0].smoothed_box[2], 110.0)


def test_empty_palette_falls_back():
    tr = _mk(palette=[])
    s = tr.assign_style(0)
    assert isinstance(s.color_id, int) and s.color_id == 0


def test_zero_sized_frame_disables_centroid_gate():
    # With W=H=0 the diag is 0; greedy must still run (gate disabled) without error.
    tr = Tracker(seed=1, iou_threshold=0.25, max_centroid_ratio=0.35,
                 memory_frames=6, smoothing=0.0, palette=PALETTE,
                 n_frame_assets=0, n_target_assets=0, style_mode="mixed",
                 display_probability=1.0, W=0, H=0, adaptive=False)
    p0 = tr.update(0, [_det(0, 0, 0, 0, 10, 10)])
    assert len(p0) == 1
    # overlapping next frame still matches (IoU floor)
    p1 = tr.update(1, [_det(1, 0, 0, 0, 10, 10)])
    assert p1[0][0].track_id == p0[0][0].track_id


# ===========================================================================
# Monotonic, never-reused ids
# ===========================================================================

def test_ids_never_reused_after_cull():
    tr = _mk(memory_frames=0, smoothing=0.0)
    seen_ids = []
    # each frame a brand new object at a non-overlapping place, previous one gone
    places = [(0, 0, 40, 40), (600, 0, 640, 40), (0, 440, 40, 480), (600, 440, 640, 480)]
    for f, b in enumerate(places):
        p = tr.update(f, [_det(f, 0, *b)])
        seen_ids.append(p[0][0].track_id)
    # all distinct and strictly increasing (never reused)
    assert seen_ids == sorted(seen_ids)
    assert len(set(seen_ids)) == len(seen_ids)


# ===========================================================================
# label carry
# ===========================================================================

def test_label_carried_onto_track():
    tr = _mk(smoothing=0.0)
    p = tr.update(0, [_det(0, 0, 40, 60, 140, 300, label="PERSON")])
    assert p[0][0].label == "PERSON"


# ===========================================================================
# micro_variation
# ===========================================================================

def test_micro_variation_keys_and_types():
    mv = micro_variation(77, 1, 0, flicker_p=0.5, style_flash_p=0.5,
                         jitter_px=3, opacity_jitter=0.1)
    assert set(mv.keys()) == {
        "opacity_mul", "line_dropout", "target_flash", "fill_flash", "jitter"
    }
    assert isinstance(mv["opacity_mul"], float)
    assert isinstance(mv["line_dropout"], bool)
    assert isinstance(mv["target_flash"], bool)
    assert isinstance(mv["fill_flash"], bool)
    assert isinstance(mv["jitter"], tuple) and len(mv["jitter"]) == 2
    assert all(isinstance(v, int) for v in mv["jitter"])


def test_micro_variation_deterministic():
    a = micro_variation(77, 3, 12, flicker_p=0.3, style_flash_p=0.2,
                        jitter_px=2, opacity_jitter=0.08)
    b = micro_variation(77, 3, 12, flicker_p=0.3, style_flash_p=0.2,
                        jitter_px=2, opacity_jitter=0.08)
    assert a == b


def test_micro_variation_varies_by_frame():
    seen = set()
    for f in range(50):
        mv = micro_variation(77, 1, f, flicker_p=0.5, style_flash_p=0.5,
                             jitter_px=4, opacity_jitter=0.2)
        seen.add((mv["opacity_mul"], mv["line_dropout"], mv["target_flash"],
                  mv["fill_flash"], mv["jitter"]))
    assert len(seen) > 1, "micro variation must change across frames"


def test_micro_variation_jitter_within_bounds():
    for f in range(200):
        mv = micro_variation(7, 2, f, flicker_p=0.0, style_flash_p=0.0,
                             jitter_px=3, opacity_jitter=0.0)
        dx, dy = mv["jitter"]
        assert -3 <= dx <= 3 and -3 <= dy <= 3
        assert mv["opacity_mul"] == 1.0  # opacity_jitter=0 -> exactly 1.0
        assert mv["line_dropout"] is False  # flicker_p=0 -> never
        assert mv["target_flash"] is False and mv["fill_flash"] is False


def test_micro_variation_jitter_zero_is_origin():
    for f in range(20):
        mv = micro_variation(7, 2, f, flicker_p=1.0, style_flash_p=1.0,
                             jitter_px=0, opacity_jitter=0.0)
        assert mv["jitter"] == (0, 0)
        # flicker_p/style_flash_p == 1.0 -> always fire
        assert mv["line_dropout"] is True
        assert mv["target_flash"] is True and mv["fill_flash"] is True


def test_micro_variation_opacity_within_bounds_and_centered():
    vals = []
    for f in range(500):
        mv = micro_variation(7, 2, f, flicker_p=0.0, style_flash_p=0.0,
                             jitter_px=0, opacity_jitter=0.2)
        om = mv["opacity_mul"]
        assert 0.05 <= om <= 2.0
        # within the requested band [1-0.2, 1+0.2] (clamp floor 0.05 < 0.8 so no clip)
        assert 0.8 - EPS <= om <= 1.2 + EPS
        vals.append(om)
    mean = sum(vals) / len(vals)
    assert abs(mean - 1.0) < 0.05, "opacity wobble should center on 1.0"


def test_micro_variation_target_and_fill_independent():
    # With a moderate flash probability, target_flash and fill_flash must not be
    # perfectly correlated across frames (independent sub-streams).
    diffs = 0
    for f in range(200):
        mv = micro_variation(77, 5, f, flicker_p=0.0, style_flash_p=0.5,
                             jitter_px=0, opacity_jitter=0.0)
        if mv["target_flash"] != mv["fill_flash"]:
            diffs += 1
    assert diffs > 0, "target and fill flashes must be independent"


def test_micro_variation_string_track_id():
    # micro_variation must accept a string track id (passthrough native string ids).
    mv = micro_variation(77, "objA", 3, flicker_p=0.5, style_flash_p=0.5,
                         jitter_px=2, opacity_jitter=0.1)
    assert set(mv.keys()) == {
        "opacity_mul", "line_dropout", "target_flash", "fill_flash", "jitter"
    }


def test_micro_variation_clamps_bad_inputs():
    # Out-of-range probabilities / negative jitter must be clamped, not crash.
    mv = micro_variation(1, 1, 1, flicker_p=5.0, style_flash_p=-1.0,
                         jitter_px=-3, opacity_jitter=9.0)
    assert mv["jitter"] == (0, 0)
    assert mv["line_dropout"] is True  # flicker_p clamped to 1.0 -> always
    assert mv["target_flash"] is False  # style_flash_p clamped to 0.0 -> never


# ===========================================================================
# id sort key (total order across int/str)
# ===========================================================================

def test_track_id_sort_key_total_order():
    ids = [3, "b", 1, "a", 2]
    ordered = sorted(ids, key=_track_id_sort_key)
    assert ordered == [1, 2, 3, "a", "b"], "ints before strings, each ascending"


# ---------------------------------------------------------------------------
# Script runner (also runnable via pytest)
# ---------------------------------------------------------------------------

def _run_all():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    passed = 0
    failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            failures.append((name, "AssertionError: %s" % e))
        except Exception as e:  # noqa: BLE001 - surface any unexpected error
            failed += 1
            failures.append((name, "%s: %s" % (type(e).__name__, e)))
        else:
            passed += 1
    print("=" * 70)
    print("test_tracker.py — %d passed, %d failed (of %d)" % (passed, failed, len(tests)))
    if failures:
        print("-" * 70)
        for name, msg in failures:
            print("FAIL  %s\n      %s" % (name, msg))
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
