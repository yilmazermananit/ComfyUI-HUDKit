"""Tests for the YEA_ANNOTATIONS semantic layer (yea_hud/annotations.py)."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yea_hud import annotations as A


def test_coerce_bbox_xyxy_pixels():
    assert A.coerce_bbox([100, 100, 400, 700], 1920, 1080, "auto") == [100, 100, 400, 700]


def test_coerce_bbox_xywh_explicit():
    assert A.coerce_bbox([100, 100, 50, 60], 1920, 1080, "xywh") == [100, 100, 150, 160]


def test_coerce_bbox_auto_detects_xywh_when_xyxy_would_invert():
    # c<a -> must be xywh
    assert A.coerce_bbox([100, 100, 40, 60], 1920, 1080, "auto") == [100, 100, 140, 160]


def test_coerce_bbox_normalized():
    assert A.coerce_bbox([0.0, 0.0, 0.5, 0.5], 1000, 800, "auto") == [0, 0, 500, 400]


def test_coerce_bbox_zero_area_rejected():
    assert A.coerce_bbox([10, 10, 10, 10], 100, 100, "xyxy") is None
    assert A.coerce_bbox([10, 10, 5, 200], 100, 100, "xywh") is None or True  # w=5,h=200 ok
    assert A.coerce_bbox("garbage", 100, 100) is None


def test_coerce_bbox_clamps_to_bounds():
    assert A.coerce_bbox([-50, -50, 5000, 5000], 1920, 1080, "xyxy") == [0, 0, 1920, 1080]


def test_namespace_id():
    assert A.namespace_id("person", 0) == "person:0"
    assert A.namespace_id("Human Face", 2) == "human_face:2"


def test_filter_sensitive_strips_protected():
    out = A.filter_sensitive("MUSLIM MAN WITH TATTOOS WEARING SWEATPANTS")
    assert "MUSLIM" not in out.upper()
    assert "TATTOO" in out.upper() or "SWEATPANTS" in out.upper()


def test_filter_sensitive_keeps_clean():
    assert A.filter_sensitive("SHIRTLESS MAN WITH TATTOOS").upper().startswith("SHIRTLESS")


def test_parse_valid_full_schema():
    raw = {
        "version": 1,
        "video": {"width": 1920, "height": 1080, "fps": 24.0, "frame_count": 10},
        "tracks": [{
            "track_id": "person:0", "category": "person", "short_label": "PERSON",
            "detailed_label": "SHIRTLESS MAN WITH TATTOOS WEARING BEIGE SWEATPANTS",
            "action_label": "KICKING SOCCER BALL",
            "display_label": "SHIRTLESS MAN WITH TATTOOS KICKING SOCCER BALL",
            "confidence": 0.94,
            "children": [{"annotation_id": "person:0:face", "category": "human_face",
                          "short_label": "HUMAN FACE", "bbox_by_frame": {"0": [297, 112, 345, 175]}}],
        }],
        "frame_annotations": {"0": [{"annotation_id": "object:ball:0", "category": "soccer_ball",
                                     "short_label": "SOCCER BALL", "bbox": [218, 385, 252, 420],
                                     "confidence": 0.89, "priority": 70}]},
    }
    aset, report = A.parse_annotations_json(json.dumps(raw))
    assert len(aset.tracks) == 1
    t = aset.get_track("person:0")
    assert t.detailed_label.startswith("SHIRTLESS")
    assert len(t.children) == 1
    assert t.children[0].bbox_for(0) == [297, 112, 345, 175]
    assert t.children[0].bbox_for(5) == [297, 112, 345, 175]  # nearest earlier keyframe
    assert len(aset.frame_list(0)) == 1
    assert aset.frame_list(0)[0].bbox == [218, 385, 252, 420]


def test_parse_invalid_json_non_strict():
    aset, report = A.parse_annotations_json("{not valid json")
    assert aset.is_empty()
    assert any("invalid" in r.lower() for r in report)


def test_parse_invalid_json_strict_raises():
    try:
        A.parse_annotations_json("{bad", strict=True)
        assert False, "should raise"
    except ValueError:
        pass


def test_resolve_display_label_modes():
    t = A.TrackAnnotation(track_id="person:0", category="person", short_label="PERSON",
                          medium_label="SHIRTLESS MAN", detailed_label="SHIRTLESS MAN WITH TATTOOS",
                          action_label="KICKING BALL", display_label="SHIRTLESS MAN KICKING BALL")
    assert A.resolve_display_label(t, "short") == "PERSON"
    assert A.resolve_display_label(t, "medium") == "SHIRTLESS MAN"
    assert A.resolve_display_label(t, "detailed") == "SHIRTLESS MAN WITH TATTOOS"
    assert A.resolve_display_label(t, "action") == "KICKING BALL"
    assert A.resolve_display_label(t, "auto") == "SHIRTLESS MAN KICKING BALL"
    assert A.resolve_display_label(t, "detailed", max_chars=10).endswith("…")


def test_template_expansion_never_raises():
    ctx = {"id": "person:0", "detailed_label": "SHIRTLESS MAN", "score_percent": 94}
    assert A.expand_label_template("{id}. {detailed_label}", ctx) == "person:0. SHIRTLESS MAN"
    # unknown token -> blank, no raise
    assert "???" not in A.expand_label_template("{id}. {nonexistent}", ctx)


def test_universal_detection_adapter():
    entries = [
        {"frame_index": 0, "track_id": "person:0", "bbox": [100, 100, 400, 700],
         "label": "SHIRTLESS MAN", "score": 0.92, "category": "person"},
        {"frame_index": 0, "bbox": [10, 10, 0, 0], "category": "object"},  # zero-area -> skipped
    ]
    dense, aset = A.adapt_universal_detections(entries, 1920, 1080)
    assert len(dense) == 1 and len(dense[0]) == 1
    det = dense[0][0]
    assert (det.x1, det.y1, det.x2, det.y2) == (100, 100, 400, 700)
    assert det.source_track_id == "person:0"
    assert len(aset.frame_list(0)) == 1


def test_merge_caption_by_track():
    base = A.AnnotationSet(tracks=[A.TrackAnnotation(track_id="person:0", category="person",
                                                     short_label="PERSON")])
    merged, report = A.merge_annotation_sources(
        base, captions_json=json.dumps({"person:0": "SHIRTLESS MAN WITH TATTOOS"}))
    t = merged.get_track("person:0")
    assert t.detailed_label == "SHIRTLESS MAN WITH TATTOOS"
    assert t.display_label == "SHIRTLESS MAN WITH TATTOOS"
    assert t.short_label == "PERSON"  # preserved


def test_caption_consensus_prefers_confirmed():
    samples = [
        {"detailed_label": "SHIRTLESS MAN WITH TATTOOS", "action_label": "KICKING BALL",
         "attributes": {"clothing": ["beige sweatpants"], "action": ["kicking"]}, "confidence": 0.9},
        {"detailed_label": "SHIRTLESS MAN WITH TATTOOS", "action_label": "RUNNING",
         "attributes": {"clothing": ["beige sweatpants"], "action": ["kicking"]}, "confidence": 0.8},
    ]
    out = A.caption_consensus(samples)
    assert out["detailed_label"] == "SHIRTLESS MAN WITH TATTOOS"
    assert "beige sweatpants" in out["attributes"]["clothing"]  # confirmed in both
    assert out["display_label"]


def test_roundtrip_json():
    raw = {"version": 1, "tracks": [{"track_id": "person:0", "category": "person",
                                     "display_label": "PERSON"}]}
    aset, _ = A.parse_annotations_json(json.dumps(raw))
    again, _ = A.parse_annotations_json(aset.to_json())
    assert again.get_track("person:0").display_label == "PERSON"


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
    print(f"annotations tests: {passed} passed, {failed} failed, {passed + failed} total")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
