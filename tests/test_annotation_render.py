"""End-to-end annotation rendering tests (the dense semantic-label feature).

Drives run_hud_renderer with native SAM3_TRACK_DATA + YONDER_ANNOTATIONS and checks
that stable detailed labels, child annotations, frame annotations, multi-group merge,
dense labels and the no-annotation fallback all render without error and behave per
the brief (stable text, hierarchy, collision-safe, generic fallback).
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
from yonder_hud.annotations import AnnotationSet, TrackAnnotation

KW = dict(seed=77, style_mode="mixed", max_objects=64, display_probability=1.0, padding_x=6,
          padding_y=6, smoothing=0.5, track_iou_threshold=0.25, track_memory_frames=6,
          frame_opacity=1.0, fill_opacity=0.3, label_opacity=0.92, line_width=2, target_scale=0.18)


def _pack(masks_bool):
    shifts = torch.arange(8)
    return (masks_bool.view(*masks_bool.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def _track_data(n_obj, B, H, W, hm=64, wm=96):
    fp = []
    for f in range(B):
        objs = torch.zeros(n_obj, hm, wm, dtype=torch.bool)
        for o in range(n_obj):
            x0 = (8 + o * 9 + f * 2) % (wm - 16)
            y0 = (6 + (o % 5) * 11) % (hm - 16)
            objs[o, y0:y0 + 14, x0:x0 + 12] = True
        fp.append(_pack(objs))
    return {"packed_masks": torch.stack(fp), "n_frames": B, "scores": [0.9] * n_obj,
            "orig_size": (H, W)}


def _frames(B, H, W):
    return torch.rand(B, H, W, 3, dtype=torch.float32) * 0.5


def test_annotation_display_label_changes_output():
    B, H, W = 3, 200, 320
    frames = _frames(B, H, W)
    td = _track_data(1, B, H, W)
    anns = {"version": 1, "tracks": [{"track_id": "person:0", "category": "person",
            "short_label": "PERSON", "detailed_label": "SHIRTLESS MAN WITH TATTOOS",
            "display_label": "SHIRTLESS MAN WITH TATTOOS KICKING BALL"}]}
    out_a, _, tj = run_hud_renderer(frames, track_data=td, annotations_json=json.dumps(anns),
                                    caption_mode="detailed", **KW)
    parsed = json.loads(tj)
    assert parsed["annotated_tracks"] == 1
    out_b, _, _ = run_hud_renderer(frames, track_data=td, **KW)  # no annotations -> generic
    assert not torch.equal(out_a, out_b)  # detailed label changes pixels vs generic


def test_child_annotations_render():
    B, H, W = 2, 240, 360
    frames = _frames(B, H, W)
    td = _track_data(1, B, H, W)
    anns = {"version": 1, "tracks": [{"track_id": "person:0", "category": "person",
            "display_label": "PERSON",
            "children": [
                {"annotation_id": "person:0:face", "category": "human_face",
                 "short_label": "HUMAN FACE", "bbox_by_frame": {"0": [40, 20, 90, 70], "1": [42, 22, 92, 72]}},
                {"annotation_id": "person:0:foot", "category": "footwear",
                 "short_label": "FOOTWEAR", "bbox_by_frame": {"0": [30, 150, 80, 200]}},
            ]}]}
    out_on, _, _ = run_hud_renderer(frames, track_data=td, annotations_json=json.dumps(anns),
                                    show_child_annotations=True, **KW)
    out_off, _, _ = run_hud_renderer(frames, track_data=td, annotations_json=json.dumps(anns),
                                     show_child_annotations=False, **KW)
    assert out_on.shape == frames.shape
    assert not torch.equal(out_on, out_off)  # children add graphics


def test_frame_annotations_render():
    B, H, W = 2, 200, 300
    frames = _frames(B, H, W)
    td = _track_data(1, B, H, W)
    anns = {"version": 1, "tracks": [{"track_id": "person:0", "category": "person", "display_label": "PERSON"}],
            "frame_annotations": {"0": [{"annotation_id": "ball:0", "category": "soccer_ball",
                                         "short_label": "SOCCER BALL", "bbox": [100, 120, 140, 160],
                                         "priority": 70}]}}
    out, _, tj = run_hud_renderer(frames, track_data=td, annotations_json=json.dumps(anns), **KW)
    assert out.shape == frames.shape


def test_multi_group_namespacing():
    B, H, W = 2, 220, 340
    frames = _frames(B, H, W)
    person = _track_data(2, B, H, W)
    face = _track_data(2, B, H, W)
    out, _, tj = run_hud_renderer(frames, track_data=person, secondary_track_data=[face, None, None],
                                  group_labels=["FACE", "", ""], **KW)
    parsed = json.loads(tj)
    assert parsed["source"] == "track_data"
    assert parsed["groups"] == 2
    assert parsed["tracks"] == 4  # person:0, person:1, face:0, face:1


def test_no_annotation_generic_fallback():
    B, H, W = 2, 200, 300
    frames = _frames(B, H, W)
    td = _track_data(2, B, H, W)
    out, _, tj = run_hud_renderer(frames, track_data=td, primary_group_label="PERSON", **KW)
    parsed = json.loads(tj)
    assert parsed["annotated_tracks"] == 0
    assert out.shape == frames.shape  # still renders frames + generic labels


def test_annotationset_object_passthrough():
    B, H, W = 2, 200, 300
    frames = _frames(B, H, W)
    td = _track_data(1, B, H, W)
    aset = AnnotationSet(tracks=[TrackAnnotation(track_id="person:0", category="person",
                                                 display_label="SHIRTLESS MAN")])
    out, _, tj = run_hud_renderer(frames, track_data=td, annotations=aset, **KW)
    assert json.loads(tj)["annotated_tracks"] == 1


def test_invalid_annotations_json_falls_back():
    B, H, W = 2, 180, 280
    frames = _frames(B, H, W)
    td = _track_data(1, B, H, W)
    out, _, tj = run_hud_renderer(frames, track_data=td, annotations_json="{bad json", **KW)
    assert out.shape == frames.shape  # no crash; generic fallback


def test_caption_mode_detailed_vs_short_differ():
    B, H, W = 2, 240, 360
    frames = _frames(B, H, W)
    td = _track_data(1, B, H, W)
    anns = {"version": 1, "tracks": [{"track_id": "person:0", "category": "person",
            "short_label": "PERSON", "detailed_label": "SHIRTLESS MAN WITH TATTOOS WEARING SWEATPANTS"}]}
    aj = json.dumps(anns)
    short = run_hud_renderer(frames, track_data=td, annotations_json=aj, caption_mode="short", label_length_mode="single_line", **KW)[0]
    detailed = run_hud_renderer(frames, track_data=td, annotations_json=aj, caption_mode="detailed", label_length_mode="single_line", **KW)[0]
    assert not torch.equal(short, detailed)


def test_dense_50_labels_completes():
    B, H, W = 2, 720, 1280
    frames = _frames(B, H, W)
    td = _track_data(50, B, H, W, hm=360, wm=640)
    tracks = [{"track_id": f"person:{i}", "category": "person", "short_label": "PERSON",
               "detailed_label": f"SUBJECT {i} WITH GEAR"} for i in range(50)]
    out, _, tj = run_hud_renderer(frames, track_data=td, annotations_json=json.dumps({"version": 1, "tracks": tracks}),
                                  label_density=1.0, max_labels_per_frame=50, **KW)
    parsed = json.loads(tj)
    assert out.shape == (B, H, W, 3)
    assert parsed["tracks"] == 50
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_label_density_low_hides_more_than_high():
    B, H, W = 1, 480, 720
    frames = torch.full((B, H, W, 3), 0.5, dtype=torch.float32)
    td = _track_data(20, B, H, W, hm=240, wm=360)
    tracks = [{"track_id": f"person:{i}", "category": "person", "detailed_label": f"SUBJECT {i}"} for i in range(20)]
    aj = json.dumps({"version": 1, "tracks": tracks})
    base = frames
    hi = run_hud_renderer(frames, track_data=td, annotations_json=aj, label_density=1.0, **KW)[0]
    lo = run_hud_renderer(frames, track_data=td, annotations_json=aj, label_density=0.2, **KW)[0]
    # fewer labels at low density -> fewer pixels changed vs base
    hi_changed = (hi - base).abs().sum().item()
    lo_changed = (lo - base).abs().sum().item()
    assert lo_changed <= hi_changed


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa
            import traceback; failed += 1; print(f"ERROR {fn.__name__}: {e!r}"); traceback.print_exc()
    print("-" * 60)
    print(f"annotation_render tests: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
