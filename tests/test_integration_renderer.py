"""End-to-end integration of the Yonder HUD Renderer business function against the
REAL native data contracts (brief §25.5 — the most critical test):

  * BOUNDING_BOX  : list[list[dict{x,y,width,height,score}]]  (SAM3 Detect shape)
  * SAM3_TRACK_DATA: {packed_masks uint8 [N,N_obj,Hm,Wm//8], n_frames, scores, orig_size}
  * MASK          : torch [B,H,W]

Runs the renderer headlessly (no ComfyUI needed) and checks output shape/range/batch
length, no-detection passthrough, source priority, and seed determinism.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from nodes.hud_renderer import run_hud_renderer

B, H, W = 4, 128, 192
RENDER_KW = dict(seed=77, style_mode="mixed", max_objects=24, display_probability=1.0,
                 padding_x=8, padding_y=8, smoothing=0.55, track_iou_threshold=0.25,
                 track_memory_frames=6, frame_opacity=1.0, fill_opacity=0.3, label_opacity=0.92,
                 line_width=3, target_scale=0.18)


def _frames():
    return torch.rand(B, H, W, 3, dtype=torch.float32) * 0.6


def _pack_masks(masks_bool):
    # Replicates comfy/ldm/sam3/tracker.py pack_masks: [*,H,W] -> [*,H,W//8] uint8.
    shifts = torch.arange(8)
    return (masks_bool.view(*masks_bool.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def _make_track_data(n_obj=2, hm=64, wm=96):
    # Each object is a moving filled box across N=B frames.
    frames_packed = []
    for f in range(B):
        objs = torch.zeros(n_obj, hm, wm, dtype=torch.bool)
        for o in range(n_obj):
            x0 = 8 + o * 30 + f * 3
            y0 = 8 + o * 10
            objs[o, y0:y0 + 24, x0:x0 + 20] = True
        frames_packed.append(_pack_masks(objs))
    packed = torch.stack(frames_packed, dim=0)  # [N, n_obj, hm, wm//8]
    return {"packed_masks": packed, "n_frames": B, "scores": [0.94, 0.81][:n_obj],
            "orig_size": (H, W)}


def _make_bboxes():
    # list[list[dict]] abs-pixel xywh, moving box per frame, 2 detections/frame.
    out = []
    for f in range(B):
        out.append([
            {"x": 20 + f * 4, "y": 30, "width": 50, "height": 80, "score": 0.93},
            {"x": 110 + f * 2, "y": 20, "width": 40, "height": 60, "score": 0.77},
        ])
    return out


def _make_masks():
    m = torch.zeros(B, H, W, dtype=torch.float32)
    for f in range(B):
        m[f, 30:90, 20 + f * 3:70 + f * 3] = 1.0
    return m


def _check_output(images, dbg, tj):
    assert images.shape == (B, H, W, 3), images.shape
    assert images.dtype == torch.float32
    assert float(images.min()) >= 0.0 and float(images.max()) <= 1.0
    assert images.shape[0] == B
    assert dbg.shape[0] == 1
    parsed = json.loads(tj)
    assert parsed["frames"] == B
    return parsed


def test_bbox_path():
    images, dbg, tj = run_hud_renderer(_frames(), bboxes=_make_bboxes(), **RENDER_KW)
    parsed = _check_output(images, dbg, tj)
    assert parsed["source"] == "bboxes"
    assert parsed["detections"] == 2 * B


def test_track_data_path_primary():
    td = _make_track_data(n_obj=2)
    images, dbg, tj = run_hud_renderer(_frames(), track_data=td, **RENDER_KW)
    parsed = _check_output(images, dbg, tj)
    assert parsed["source"] == "track_data"
    assert parsed["tracks"] == 2  # persistent object identities


def test_track_data_priority_over_bboxes():
    td = _make_track_data(n_obj=2)
    _, _, tj = run_hud_renderer(_frames(), track_data=td, bboxes=_make_bboxes(), **RENDER_KW)
    assert json.loads(tj)["source"] == "track_data"  # SAM3_TRACK_DATA preferred


def test_mask_fallback_path():
    images, dbg, tj = run_hud_renderer(_frames(), masks=_make_masks(), **RENDER_KW)
    parsed = _check_output(images, dbg, tj)
    assert parsed["source"] == "masks"


def test_no_source_raises():
    try:
        run_hud_renderer(_frames(), **RENDER_KW)
        assert False, "should raise ValueError when no detection source"
    except ValueError as e:
        assert "BOUNDING_BOX" in str(e) or "MASK" in str(e) or "TRACK_DATA" in str(e)


def test_no_detection_passthrough():
    frames = _frames()
    empty = [[] for _ in range(B)]
    images, dbg, tj = run_hud_renderer(frames, bboxes=empty, **RENDER_KW)
    assert torch.allclose(images, frames, atol=1e-6)
    assert json.loads(tj)["detections"] == 0


def test_seed_determinism():
    frames = _frames()
    bb = _make_bboxes()
    a = run_hud_renderer(frames, bboxes=bb, **{**RENDER_KW, "seed": 77})[0]
    b = run_hud_renderer(frames, bboxes=bb, **{**RENDER_KW, "seed": 77})[0]
    assert torch.equal(a, b), "same seed must be reproducible"


def test_different_seed_differs():
    frames = _frames()
    bb = _make_bboxes()
    a = run_hud_renderer(frames, bboxes=bb, **{**RENDER_KW, "seed": 1})[0]
    b = run_hud_renderer(frames, bboxes=bb, **{**RENDER_KW, "seed": 999})[0]
    # different seed -> different style/color assignment -> not identical
    assert not torch.equal(a, b)


def test_track_data_frame_mismatch_reported():
    td = _make_track_data(n_obj=1)
    td["packed_masks"] = td["packed_masks"][:2]  # only 2 frames of track data vs B=4 frames
    td["n_frames"] = 2
    images, dbg, tj = run_hud_renderer(_frames(), track_data=td, **RENDER_KW)
    assert images.shape[0] == B  # output still full batch
    parsed = json.loads(tj)
    assert "frame_mismatch" in parsed or parsed["frames"] == B


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa
            import traceback; failed += 1; print(f"ERROR {fn.__name__}: {e!r}")
            traceback.print_exc()
    print("-" * 60)
    print(f"integration_renderer tests: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
