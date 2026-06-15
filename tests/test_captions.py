"""Unit tests for yonder_hud.captions (brief §25 caption cases).

Runnable two ways:

    /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_captions.py -q
    /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_captions.py

The second form (``__main__`` block) runs every ``test_*`` function, prints a
summary, and ``sys.exit(1)`` on the first failure so it works without pytest
installed (the target venv has no pytest).

No ComfyUI, no network, no model files. Fixtures are tiny synthetic
``Detection`` objects (numpy/torch are imported only to prove the module needs
neither and to keep parity with sibling test files).
"""

from __future__ import annotations

import os
import sys

# Make the package importable when run as a bare script from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402  (proves no torch/PIL dependency leaks in)

from yonder_hud.captions import (  # noqa: E402
    build_template_context,
    expand_template,
    format_order_label,
    parse_captions_json,
    resolve_caption,
)
from yonder_hud.types import CAPTION_SOURCES, Detection  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _det(
    *,
    frame_index=0,
    source_index=0,
    x1=10.0,
    y1=20.0,
    x2=110.0,
    y2=220.0,
    score=None,
    label=None,
    source_track_id=None,
    mask_index=None,
):
    """Build a synthetic Detection (xyxy)."""
    return Detection(
        frame_index=frame_index,
        source_index=source_index,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=score,
        label=label,
        source_track_id=source_track_id,
        mask_index=mask_index,
    )


# ---------------------------------------------------------------------------
# parse_captions_json — tolerant parsing (§25)
# ---------------------------------------------------------------------------

def test_parse_none_returns_empty():
    assert parse_captions_json(None) == {}


def test_parse_empty_and_whitespace_returns_empty():
    assert parse_captions_json("") == {}
    assert parse_captions_json("   \n\t ") == {}


def test_parse_invalid_json_returns_empty():
    assert parse_captions_json("{not valid json") == {}
    assert parse_captions_json("}{") == {}
    assert parse_captions_json("nope") == {}


def test_parse_non_dict_toplevel_returns_empty():
    # Valid JSON but not an object -> {}.
    assert parse_captions_json("[1, 2, 3]") == {}
    assert parse_captions_json('"a string"') == {}
    assert parse_captions_json("42") == {}
    assert parse_captions_json("true") == {}
    assert parse_captions_json("null") == {}


def test_parse_non_str_input_returns_empty():
    # Already-parsed / wrong-typed inputs must not crash.
    assert parse_captions_json(123) == {}  # type: ignore[arg-type]
    assert parse_captions_json(["a"]) == {}  # type: ignore[arg-type]
    assert parse_captions_json({"1": "X"}) == {}  # type: ignore[arg-type]


def test_parse_by_track_dict():
    d = parse_captions_json('{"1": "ALPHA", "2": "BRAVO"}')
    assert d == {"1": "ALPHA", "2": "BRAVO"}


def test_parse_by_frame_dict():
    d = parse_captions_json('{"frames": {"0": ["A", "B"], "12": "SOLO"}}')
    assert d["frames"]["0"] == ["A", "B"]
    assert d["frames"]["12"] == "SOLO"


def test_parse_mixed_layout_preserved():
    d = parse_captions_json('{"1": "TRK", "frames": {"3": "FR"}}')
    assert d["1"] == "TRK"
    assert d["frames"]["3"] == "FR"


# ---------------------------------------------------------------------------
# resolve_caption — source = fixed
# ---------------------------------------------------------------------------

def test_source_fixed_returns_label():
    out = resolve_caption(
        source="fixed", det=_det(), track_id=1, order_index=1,
        fixed_label="PERSON", captions={}, template=None, frame_index=0,
    )
    assert out == "PERSON"


def test_source_fixed_none_label_is_blank():
    out = resolve_caption(
        source="fixed", det=_det(), track_id=1, order_index=1,
        fixed_label=None, captions={}, template=None, frame_index=0,
    )
    assert out == ""


def test_source_fixed_does_not_uppercase():
    # Contract: returns RAW string, not uppercased.
    out = resolve_caption(
        source="fixed", det=_det(), track_id=1, order_index=1,
        fixed_label="person", captions={}, template=None, frame_index=0,
    )
    assert out == "person"


# ---------------------------------------------------------------------------
# resolve_caption — source = bbox_label (SAM3 has none)
# ---------------------------------------------------------------------------

def test_source_bbox_label_present():
    out = resolve_caption(
        source="bbox_label", det=_det(label="face"), track_id=1, order_index=1,
        fixed_label="X", captions={}, template=None, frame_index=0,
    )
    assert out == "face"


def test_source_bbox_label_missing_is_blank():
    # SAM3 emits no label -> '' (expected, findings §3).
    out = resolve_caption(
        source="bbox_label", det=_det(label=None), track_id=1, order_index=1,
        fixed_label="X", captions={}, template=None, frame_index=0,
    )
    assert out == ""


def test_source_bbox_label_none_det_is_blank():
    out = resolve_caption(
        source="bbox_label", det=None, track_id=1, order_index=1,
        fixed_label="X", captions={}, template=None, frame_index=0,
    )
    assert out == ""


def test_source_bbox_label_empty_string_is_blank():
    out = resolve_caption(
        source="bbox_label", det=_det(label=""), track_id=1, order_index=1,
        fixed_label="X", captions={}, template=None, frame_index=0,
    )
    assert out == ""


# ---------------------------------------------------------------------------
# resolve_caption — source = json_by_track
# ---------------------------------------------------------------------------

def test_source_json_by_track_hit():
    caps = parse_captions_json('{"7": "HERO", "8": "SIDEKICK"}')
    out = resolve_caption(
        source="json_by_track", det=_det(), track_id=7, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == "HERO"


def test_source_json_by_track_string_id():
    caps = {"abc": "NAMED"}
    out = resolve_caption(
        source="json_by_track", det=_det(), track_id="abc", order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == "NAMED"


def test_source_json_by_track_miss_is_blank():
    caps = {"1": "A"}
    out = resolve_caption(
        source="json_by_track", det=_det(), track_id=99, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == ""


def test_source_json_by_track_empty_dict_is_blank():
    out = resolve_caption(
        source="json_by_track", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions={}, template=None, frame_index=0,
    )
    assert out == ""


def test_source_json_by_track_list_value_indexed_by_source_index():
    # A list value -> pick the detection's source_index slot.
    caps = {"3": ["FIRST", "SECOND", "THIRD"]}
    out = resolve_caption(
        source="json_by_track", det=_det(source_index=1), track_id=3, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == "SECOND"


def test_source_json_by_track_numeric_value_stringified():
    caps = {"1": 42}
    out = resolve_caption(
        source="json_by_track", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == "42"


def test_source_json_by_track_none_value_is_blank():
    caps = {"1": None}
    out = resolve_caption(
        source="json_by_track", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == ""


# ---------------------------------------------------------------------------
# resolve_caption — source = json_by_frame
# ---------------------------------------------------------------------------

def test_source_json_by_frame_string_value():
    caps = parse_captions_json('{"frames": {"5": "SOLO"}}')
    out = resolve_caption(
        source="json_by_frame", det=_det(frame_index=5), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=5,
    )
    assert out == "SOLO"


def test_source_json_by_frame_list_indexed_by_source_index():
    caps = parse_captions_json('{"frames": {"0": ["A", "B", "C"]}}')
    out_a = resolve_caption(
        source="json_by_frame", det=_det(source_index=0), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    out_b = resolve_caption(
        source="json_by_frame", det=_det(source_index=2), track_id=1, order_index=3,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out_a == "A"
    assert out_b == "C"


def test_source_json_by_frame_list_out_of_range_falls_back_to_first():
    caps = parse_captions_json('{"frames": {"0": ["ONLY"]}}')
    out = resolve_caption(
        source="json_by_frame", det=_det(source_index=9), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == "ONLY"


def test_source_json_by_frame_missing_frame_is_blank():
    caps = parse_captions_json('{"frames": {"0": "A"}}')
    out = resolve_caption(
        source="json_by_frame", det=_det(frame_index=7), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=7,
    )
    assert out == ""


def test_source_json_by_frame_no_frames_key_is_blank():
    caps = {"1": "TRACK_CAP"}  # only by-track data present
    out = resolve_caption(
        source="json_by_frame", det=_det(frame_index=0), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == ""


def test_source_json_by_frame_frames_not_a_dict_is_blank():
    caps = {"frames": ["not", "a", "dict"]}
    out = resolve_caption(
        source="json_by_frame", det=_det(frame_index=0), track_id=1, order_index=1,
        fixed_label="X", captions=caps, template=None, frame_index=0,
    )
    assert out == ""


# ---------------------------------------------------------------------------
# resolve_caption — source = template (must NEVER raise on bad tokens)
# ---------------------------------------------------------------------------

def test_template_all_known_tokens():
    det = _det(frame_index=4, source_index=2, x1=10, y1=20, x2=110, y2=220,
               score=0.9375, label="cat")
    out = resolve_caption(
        source="template", det=det, track_id=3, order_index=2,
        fixed_label="X", captions={},
        template="T{track} F{frame} I{index} L={label} S={score} "
                 "[{x},{y},{w},{h}]",
        frame_index=4,
    )
    # x=10,y=20,w=100,h=200 ; score rounded to 2dp
    assert out == "T3 F4 I2 L=cat S=0.94 [10,20,100,200]"


def test_template_unknown_token_left_literal():
    out = resolve_caption(
        source="template", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions={},
        template="{track} {bogus} {label}", frame_index=0,
    )
    # {bogus} unknown -> stays literal; label empty.
    assert out == "1 {bogus} "


def test_template_unbalanced_brace_does_not_raise():
    # str.format() would raise ValueError on these; we must not.
    for bad in ["{", "}", "{track", "track}", "}{track}{", "{ {track} }"]:
        out = resolve_caption(
            source="template", det=_det(), track_id=5, order_index=1,
            fixed_label="X", captions={}, template=bad, frame_index=0,
        )
        assert isinstance(out, str)  # no exception, always a string


def test_template_escaped_braces_literal():
    out = resolve_caption(
        source="template", det=_det(), track_id=9, order_index=1,
        fixed_label="X", captions={},
        template="{{literal}} {track}", frame_index=0,
    )
    assert out == "{literal} 9"


def test_template_empty_and_none():
    assert resolve_caption(
        source="template", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions={}, template="", frame_index=0,
    ) == ""
    assert resolve_caption(
        source="template", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions={}, template=None, frame_index=0,
    ) == ""


def test_template_score_none_is_blank_token():
    out = resolve_caption(
        source="template", det=_det(score=None), track_id=1, order_index=1,
        fixed_label="X", captions={}, template="s=[{score}]", frame_index=0,
    )
    assert out == "s=[]"


def test_template_label_none_is_blank_token():
    out = resolve_caption(
        source="template", det=_det(label=None), track_id=1, order_index=1,
        fixed_label="X", captions={}, template="l=[{label}]", frame_index=0,
    )
    assert out == "l=[]"


def test_template_none_det_geometry_defaults_to_zero():
    out = resolve_caption(
        source="template", det=None, track_id=2, order_index=1,
        fixed_label="X", captions={},
        template="{x}-{y}-{w}-{h}-{label}-{score}", frame_index=0,
    )
    assert out == "0-0-0-0--"


def test_template_format_spec_on_token_ignored_not_raised():
    # A format spec must not crash; the pre-formatted context value is used.
    out = resolve_caption(
        source="template", det=_det(score=0.5), track_id=1, order_index=1,
        fixed_label="X", captions={}, template="{score:.0f}", frame_index=0,
    )
    # spec is ignored; pre-formatted "0.50" used. Crucially: no exception.
    assert out == "0.50"


def test_template_negative_coordinates_rounded():
    det = _det(x1=-3.4, y1=-0.6, x2=6.6, y2=9.4)  # w=10, h=10
    out = resolve_caption(
        source="template", det=det, track_id=1, order_index=1,
        fixed_label="X", captions={}, template="{x},{y},{w},{h}", frame_index=0,
    )
    # round(-3.4)=-3, round(-0.6)=-1, w=round(10.0)=10, h=round(10.0)=10
    assert out == "-3,-1,10,10"


# ---------------------------------------------------------------------------
# resolve_caption — source = none + unknown source
# ---------------------------------------------------------------------------

def test_source_none_is_blank():
    out = resolve_caption(
        source="none", det=_det(label="x"), track_id=1, order_index=1,
        fixed_label="SHOULD_NOT_SHOW", captions={"1": "NOPE"},
        template="{track}", frame_index=0,
    )
    assert out == ""


def test_unknown_source_treated_as_none():
    out = resolve_caption(
        source="totally_unknown", det=_det(), track_id=1, order_index=1,
        fixed_label="X", captions={"1": "Y"}, template="{track}", frame_index=0,
    )
    assert out == ""


def test_non_string_source_treated_as_none():
    out = resolve_caption(
        source=None, det=_det(), track_id=1, order_index=1,  # type: ignore[arg-type]
        fixed_label="X", captions={}, template=None, frame_index=0,
    )
    assert out == ""


def test_all_caption_sources_return_str_and_never_raise():
    # Exhaustive: every declared source must return a str for a normal det.
    det = _det(label="L", score=0.7)
    caps = parse_captions_json('{"1": "BYTRACK", "frames": {"0": "BYFRAME"}}')
    for src in CAPTION_SOURCES:
        out = resolve_caption(
            source=src, det=det, track_id=1, order_index=1,
            fixed_label="FIX", captions=caps, template="{track}/{frame}",
            frame_index=0,
        )
        assert isinstance(out, str), f"source {src!r} returned non-str {out!r}"


# ---------------------------------------------------------------------------
# build_template_context / expand_template (the engine, used directly)
# ---------------------------------------------------------------------------

def test_build_context_keys_present():
    ctx = build_template_context(
        det=_det(score=0.5, label="x"), track_id=1, order_index=2, frame_index=3,
    )
    for k in ("track", "frame", "index", "label", "score", "x", "y", "w", "h"):
        assert k in ctx
        assert isinstance(ctx[k], str)


def test_build_context_none_det():
    ctx = build_template_context(det=None, track_id="t", order_index=5, frame_index=8)
    assert ctx["track"] == "t"
    assert ctx["frame"] == "8"
    assert ctx["index"] == "5"
    assert ctx["label"] == ""
    assert ctx["score"] == ""
    assert ctx["x"] == ctx["y"] == ctx["w"] == ctx["h"] == "0"


def test_expand_template_direct_unknown_literal():
    ctx = {"track": "9"}
    assert expand_template("{track}-{nope}", ctx) == "9-{nope}"


def test_expand_template_attribute_access_left_literal():
    # Nested / attribute fields must NOT reach into anything; stay literal.
    ctx = {"track": "9"}
    assert expand_template("{track.real}", ctx) == "{track.real}"
    assert expand_template("{track[0]}", ctx) == "{track[0]}"


def test_expand_template_empty_returns_empty():
    assert expand_template("", {"track": "1"}) == ""
    assert expand_template(None, {"track": "1"}) == ""


# ---------------------------------------------------------------------------
# format_order_label
# ---------------------------------------------------------------------------

def test_format_order_label_basic():
    assert format_order_label(3, "PERSON") == "03. PERSON"


def test_format_order_label_two_digit():
    assert format_order_label(10, "CAR") == "10. CAR"


def test_format_order_label_three_digit():
    assert format_order_label(100, "X") == "100. X"


def test_format_order_label_one():
    assert format_order_label(1, "ALPHA") == "01. ALPHA"


def test_format_order_label_empty_text_drops_space():
    assert format_order_label(3, "") == "03."


def test_format_order_label_none_text():
    assert format_order_label(7, None) == "07."  # type: ignore[arg-type]


def test_format_order_label_does_not_uppercase():
    # Caller uppercases; this preserves case.
    assert format_order_label(2, "person") == "02. person"


def test_format_order_label_non_int_index_stringified():
    out = format_order_label("A", "X")  # type: ignore[arg-type]
    assert out == "A. X"


# ---------------------------------------------------------------------------
# Determinism: same inputs -> same output (no hidden state)
# ---------------------------------------------------------------------------

def test_resolve_caption_deterministic():
    args = dict(
        source="template", det=_det(score=0.3, label="z"), track_id=4,
        order_index=2, fixed_label="X", captions={}, template="{track}:{label}:{score}",
        frame_index=1,
    )
    a = resolve_caption(**args)
    b = resolve_caption(**args)
    assert a == b == "4:z:0.30"


# ---------------------------------------------------------------------------
# Script runner (works without pytest)
# ---------------------------------------------------------------------------

def _run_all() -> int:
    """Run every test_* in this module; print summary; return failure count."""
    g = dict(globals())
    tests = sorted(
        (name, fn) for name, fn in g.items()
        if name.startswith("test_") and callable(fn)
    )
    passed = 0
    failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            failures.append((name, "AssertionError: " + str(exc)))
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface any unexpected error
            failed += 1
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            passed += 1
    total = passed + failed
    print("-" * 60)
    print(f"captions tests: {passed}/{total} passed, {failed} failed")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
