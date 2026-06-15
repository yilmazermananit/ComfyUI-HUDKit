"""Unit tests for ``yea_hud.palette`` (color resolution, brief §14).

Runnable two ways:

* as pytest::

      /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_palette.py -q

* as a plain script (no pytest required)::

      /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_palette.py

The script runner executes every ``test_*`` function, prints a summary and exits
non-zero on any failure. No ComfyUI, no network, no model files, no file writes.

Coverage maps to the contract in ``docs/INTERFACES.md`` ("## palette.py") and the
brief §25 cases relevant to color:

* hex parsing (with/without '#', case, whitespace, bad lengths, non-hex, None);
* ``resolve_palette`` — drop invalid, fall back to DEFAULT_PALETTE, ALWAYS >= 1
  color, broadcast-safe;
* deterministic ``color_for_track`` via ``stable_hash`` (incl. a fresh-subprocess
  check with a different ``PYTHONHASHSEED`` — the load-bearing determinism case);
* ``contrast_text_color`` (auto Rec.709 luma threshold @0.55, white/black force);
* ``luminance`` (Rec.709 coefficients, clamping, ordering).
"""

from __future__ import annotations

import os
import subprocess
import sys

# Make the package importable when run as a bare script from anywhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json  # noqa: E402

from yea_hud.types import DEFAULT_PALETTE  # noqa: E402
from yea_hud.utils import stable_hash  # noqa: E402
from yea_hud.palette import (  # noqa: E402
    color_for_track,
    contrast_text_color,
    luminance,
    parse_hex,
    resolve_palette,
)


# Small float tolerance for channel comparisons.
_EPS = 1e-9


def _approx(a, b, eps=_EPS):
    return abs(float(a) - float(b)) <= eps


def _rgb_approx(a, b, eps=_EPS):
    return all(_approx(x, y, eps) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# parse_hex
# ---------------------------------------------------------------------------


def test_parse_hex_with_hash():
    assert _rgb_approx(parse_hex("#FFFFFF"), (1.0, 1.0, 1.0))
    assert _rgb_approx(parse_hex("#000000"), (0.0, 0.0, 0.0))


def test_parse_hex_without_hash():
    assert _rgb_approx(parse_hex("ffffff"), (1.0, 1.0, 1.0))
    assert _rgb_approx(parse_hex("000000"), (0.0, 0.0, 0.0))


def test_parse_hex_case_insensitive():
    assert parse_hex("#ff2a1a") == parse_hex("#FF2A1A")
    assert parse_hex("Ff2A1a") == parse_hex("FF2A1A")


def test_parse_hex_channel_values():
    # #FF2A1A -> R=255, G=42, B=26.
    r, g, b = parse_hex("#FF2A1A")
    assert _approx(r, 255 / 255.0)
    assert _approx(g, 42 / 255.0)
    assert _approx(b, 26 / 255.0)


def test_parse_hex_pure_channels():
    assert _rgb_approx(parse_hex("FF0000"), (1.0, 0.0, 0.0))
    assert _rgb_approx(parse_hex("00FF00"), (0.0, 1.0, 0.0))
    assert _rgb_approx(parse_hex("0000FF"), (0.0, 0.0, 1.0))


def test_parse_hex_strips_whitespace():
    assert parse_hex("  #225CFF  ") == parse_hex("#225CFF")
    assert parse_hex("\t225CFF\n") == parse_hex("225CFF")


def test_parse_hex_rejects_bad_length():
    assert parse_hex("#FFF") is None          # 3-digit short form not supported
    assert parse_hex("#FFFFFFFF") is None      # 8-digit RRGGBBAA not supported
    assert parse_hex("FFFFF") is None          # 5 digits
    assert parse_hex("FFFFFFF") is None        # 7 digits
    assert parse_hex("") is None
    assert parse_hex("#") is None
    assert parse_hex("   ") is None


def test_parse_hex_rejects_non_hex_chars():
    assert parse_hex("#GGGGGG") is None
    assert parse_hex("#12345Z") is None
    assert parse_hex("red") is None
    assert parse_hex("0x1234") is None         # '0x' prefix is not 6 hex digits


def test_parse_hex_rejects_non_string():
    assert parse_hex(None) is None
    assert parse_hex(123456) is None
    assert parse_hex(["#FFFFFF"]) is None
    assert parse_hex(0xFFFFFF) is None


def test_parse_hex_range_is_unit_interval():
    for s in ("#000000", "#7F7F7F", "#FFFFFF", "#123456", "#abcdef"):
        rgb = parse_hex(s)
        assert rgb is not None
        for c in rgb:
            assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# resolve_palette
# ---------------------------------------------------------------------------


def _default_rgb():
    return [parse_hex(h) for h in DEFAULT_PALETTE]


def test_resolve_palette_none_returns_default():
    pal = resolve_palette(None)
    assert pal == _default_rgb()
    assert len(pal) == len(DEFAULT_PALETTE) >= 1


def test_resolve_palette_empty_string_returns_default():
    assert resolve_palette("") == _default_rgb()
    assert resolve_palette("   ") == _default_rgb()


def test_resolve_palette_invalid_json_returns_default():
    assert resolve_palette("not json") == _default_rgb()
    assert resolve_palette("[oops") == _default_rgb()
    assert resolve_palette("{bad}") == _default_rgb()


def test_resolve_palette_non_list_json_returns_default():
    assert resolve_palette("123") == _default_rgb()          # number
    assert resolve_palette("true") == _default_rgb()         # bool
    assert resolve_palette("null") == _default_rgb()         # null
    assert resolve_palette('{"a": "#FFFFFF"}') == _default_rgb()  # object


def test_resolve_palette_valid_list():
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF"]')
    assert len(pal) == 3
    assert _rgb_approx(pal[0], (1.0, 0.0, 0.0))
    assert _rgb_approx(pal[1], (0.0, 1.0, 0.0))
    assert _rgb_approx(pal[2], (0.0, 0.0, 1.0))


def test_resolve_palette_drops_invalid_entries():
    # Two valid, the rest invalid (bad hex, wrong type, short form) -> kept = 2.
    pal = resolve_palette('["#FF0000", "nope", "#0000FF", "#FFF", 123, null]')
    assert len(pal) == 2
    assert _rgb_approx(pal[0], (1.0, 0.0, 0.0))
    assert _rgb_approx(pal[1], (0.0, 0.0, 1.0))


def test_resolve_palette_all_invalid_falls_back_to_default():
    assert resolve_palette('["nope", "zzz", 5, null]') == _default_rgb()


def test_resolve_palette_empty_list_falls_back_to_default():
    assert resolve_palette("[]") == _default_rgb()


def test_resolve_palette_always_at_least_one_color():
    # Whatever we throw at it, the contract guarantees >= 1 color.
    for arg in (None, "", "   ", "garbage", "[]", "{}", "123", '["bad"]', "null"):
        pal = resolve_palette(arg)
        assert isinstance(pal, list)
        assert len(pal) >= 1, f"empty palette for {arg!r}"
        for c in pal:
            assert len(c) == 3
            for ch in c:
                assert 0.0 <= ch <= 1.0


def test_resolve_palette_bare_json_string_is_single_color():
    # A bare JSON string that is valid hex -> one-element palette (convenience).
    pal = resolve_palette(json.dumps("#225CFF"))
    assert len(pal) == 1
    assert _rgb_approx(pal[0], parse_hex("#225CFF"))


def test_resolve_palette_bare_json_string_invalid_falls_back():
    assert resolve_palette(json.dumps("not-a-color")) == _default_rgb()


def test_resolve_palette_accepts_no_hash_entries():
    pal = resolve_palette('["FF0000", "00FF00"]')
    assert len(pal) == 2
    assert _rgb_approx(pal[0], (1.0, 0.0, 0.0))


def test_resolve_palette_returns_tuples_of_floats():
    pal = resolve_palette('["#123456"]')
    assert isinstance(pal, list)
    assert isinstance(pal[0], tuple)
    assert all(isinstance(c, float) for c in pal[0])


# ---------------------------------------------------------------------------
# color_for_track
# ---------------------------------------------------------------------------


def test_color_for_track_index_formula():
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"]')
    for seed in (0, 77, 12345):
        for tid in (0, 1, 2, 7, 99):
            idx, rgb = color_for_track(pal, seed, tid)
            assert idx == stable_hash(seed, tid) % len(pal)
            assert _rgb_approx(rgb, pal[idx])


def test_color_for_track_index_in_range():
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF"]')
    for tid in range(200):
        idx, rgb = color_for_track(pal, 77, tid)
        assert 0 <= idx < len(pal)
        assert rgb in [tuple(c) for c in pal]


def test_color_for_track_deterministic_same_inputs():
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF", "#FFFFFF"]')
    a = color_for_track(pal, 77, 5)
    b = color_for_track(pal, 77, 5)
    assert a == b


def test_color_for_track_varies_with_seed_and_id():
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000"]')
    # Different track ids should (across a spread) produce more than one index.
    idxs = {color_for_track(pal, 77, t)[0] for t in range(40)}
    assert len(idxs) > 1
    # Different seeds should generally change the chosen index for a fixed id.
    seeds_idxs = {color_for_track(pal, s, 3)[0] for s in range(40)}
    assert len(seeds_idxs) > 1


def test_color_for_track_string_track_id():
    # source_track_id may be a str (types.Detection). Must work and be stable.
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF"]')
    a = color_for_track(pal, 77, "obj_a")
    b = color_for_track(pal, 77, "obj_a")
    assert a == b
    assert 0 <= a[0] < len(pal)
    # int 1 and str "1" are hashed type-safely -> may differ; both valid indices.
    c_int = color_for_track(pal, 77, 1)
    c_str = color_for_track(pal, 77, "1")
    assert 0 <= c_int[0] < len(pal)
    assert 0 <= c_str[0] < len(pal)


def test_color_for_track_single_color_palette():
    pal = resolve_palette('["#225CFF"]')
    for tid in range(20):
        idx, rgb = color_for_track(pal, 77, tid)
        assert idx == 0
        assert _rgb_approx(rgb, parse_hex("#225CFF"))


def test_color_for_track_empty_palette_raises():
    try:
        color_for_track([], 77, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty palette")


def test_color_for_track_returns_palette_color_object():
    # The returned rgb equals palette[idx] (channel-wise).
    pal = resolve_palette('["#102030", "#405060"]')
    idx, rgb = color_for_track(pal, 1, 1)
    assert _rgb_approx(rgb, pal[idx])
    assert len(rgb) == 3


# ---------------------------------------------------------------------------
# luminance — Rec.709
# ---------------------------------------------------------------------------


def test_luminance_black_and_white():
    assert _approx(luminance((0.0, 0.0, 0.0)), 0.0)
    assert _approx(luminance((1.0, 1.0, 1.0)), 1.0)


def test_luminance_rec709_coefficients():
    assert _approx(luminance((1.0, 0.0, 0.0)), 0.2126)
    assert _approx(luminance((0.0, 1.0, 0.0)), 0.7152)
    assert _approx(luminance((0.0, 0.0, 1.0)), 0.0722)


def test_luminance_weighted_sum():
    r, g, b = 0.4, 0.6, 0.2
    expected = 0.2126 * r + 0.7152 * g + 0.0722 * b
    assert _approx(luminance((r, g, b)), expected)


def test_luminance_green_brighter_than_blue():
    # Perceptual ordering: pure green reads much brighter than pure blue.
    assert luminance((0.0, 1.0, 0.0)) > luminance((0.0, 0.0, 1.0))
    assert luminance((1.0, 0.0, 0.0)) > luminance((0.0, 0.0, 1.0))


def test_luminance_in_unit_interval():
    for rgb in [(0, 0, 0), (1, 1, 1), (0.3, 0.7, 0.9), (1, 0, 0), (0.5, 0.5, 0.5)]:
        y = luminance(rgb)
        assert 0.0 <= y <= 1.0


def test_luminance_clamps_out_of_range_and_nonfinite():
    # Out-of-range channels are clamped before weighting -> still in [0,1].
    assert _approx(luminance((2.0, 2.0, 2.0)), 1.0)
    assert _approx(luminance((-1.0, -1.0, -1.0)), 0.0)
    # NaN channel folds to 0 (does not propagate NaN into the result).
    y = luminance((float("nan"), 1.0, 0.0))
    assert _approx(y, 0.7152)


# ---------------------------------------------------------------------------
# contrast_text_color
# ---------------------------------------------------------------------------


def test_contrast_text_color_force_white():
    assert contrast_text_color((1.0, 1.0, 1.0), "white") == (1.0, 1.0, 1.0)
    # Forced even over a bright background where auto would pick black.
    assert contrast_text_color((0.0, 0.0, 0.0), "white") == (1.0, 1.0, 1.0)


def test_contrast_text_color_force_black():
    assert contrast_text_color((0.0, 0.0, 0.0), "black") == (0.0, 0.0, 0.0)
    # Forced even over a dark background where auto would pick white.
    assert contrast_text_color((1.0, 1.0, 1.0), "black") == (0.0, 0.0, 0.0)


def test_contrast_text_color_auto_bright_bg_gets_black():
    # White / near-white backgrounds (luma > 0.55) -> black text.
    assert contrast_text_color((1.0, 1.0, 1.0), "auto") == (0.0, 0.0, 0.0)
    assert contrast_text_color((0.0, 1.0, 0.0), "auto") == (0.0, 0.0, 0.0)  # luma .7152
    # Default mode is 'auto'.
    assert contrast_text_color((1.0, 1.0, 1.0)) == (0.0, 0.0, 0.0)


def test_contrast_text_color_auto_dark_bg_gets_white():
    # Black / dark / pure-blue backgrounds (luma <= 0.55) -> white text.
    assert contrast_text_color((0.0, 0.0, 0.0), "auto") == (1.0, 1.0, 1.0)
    assert contrast_text_color((0.0, 0.0, 1.0), "auto") == (1.0, 1.0, 1.0)  # luma .0722
    assert contrast_text_color((1.0, 0.0, 0.0), "auto") == (1.0, 1.0, 1.0)  # luma .2126


def test_contrast_text_color_auto_threshold_strict_at_055():
    # Construct a gray whose luma is exactly 0.55: r=g=b=0.55 -> luma 0.55.
    gray = (0.55, 0.55, 0.55)
    assert _approx(luminance(gray), 0.55)
    # Strict '>' => exactly-at-threshold is NOT "bright" -> white text.
    assert contrast_text_color(gray, "auto") == (1.0, 1.0, 1.0)
    # Just above threshold -> black.
    assert contrast_text_color((0.56, 0.56, 0.56), "auto") == (0.0, 0.0, 0.0)
    # Just below threshold -> white.
    assert contrast_text_color((0.54, 0.54, 0.54), "auto") == (1.0, 1.0, 1.0)


def test_contrast_text_color_unknown_mode_defaults_to_auto():
    # Any unrecognized mode is treated as 'auto' (defensive).
    assert contrast_text_color((1.0, 1.0, 1.0), "purple") == (0.0, 0.0, 0.0)
    assert contrast_text_color((0.0, 0.0, 0.0), "") == (1.0, 1.0, 1.0)


def test_contrast_text_color_returns_pure_bw():
    for bg in [(0.1, 0.2, 0.3), (0.9, 0.9, 0.9), (0.5, 0.5, 0.5)]:
        out = contrast_text_color(bg, "auto")
        assert out in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_contrast_text_color_integrates_with_palette():
    # End-to-end: default palette colors get a legible text color each.
    pal = resolve_palette(None)
    for c in pal:
        txt = contrast_text_color(c, "auto")
        assert txt in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Cross-process determinism for color_for_track (the load-bearing property).
# Re-run the index pick in a fresh interpreter with a *different* PYTHONHASHSEED
# and assert it is byte-for-byte identical to the in-process value (the builtin
# randomised hash is forbidden by the contract).
# ---------------------------------------------------------------------------


def test_color_for_track_independent_of_pythonhashseed():
    pal = resolve_palette('["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000"]')
    n = len(pal)
    # Mix of int and str track ids; the index is hash(seed, tid) % n.
    expected = {
        ("int", 0): stable_hash(77, 0) % n,
        ("int", 13): stable_hash(77, 13) % n,
        ("str", "obj_a"): stable_hash(77, "obj_a") % n,
    }
    # Sanity: matches color_for_track in-process.
    assert color_for_track(pal, 77, 0)[0] == expected[("int", 0)]
    assert color_for_track(pal, 77, "obj_a")[0] == expected[("str", "obj_a")]

    snippet = (
        "from yea_hud.palette import resolve_palette, color_for_track;"
        "pal=resolve_palette('[\"#FF0000\",\"#00FF00\",\"#0000FF\",\"#FFFFFF\",\"#000000\"]');"
        "print(color_for_track(pal,77,0)[0]);"
        "print(color_for_track(pal,77,13)[0]);"
        "print(color_for_track(pal,77,'obj_a')[0])"
    )
    for seed in ("0", "1", "987654321", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 0, f"subprocess failed (seed={seed}): {out.stderr}"
        lines = out.stdout.strip().splitlines()
        assert len(lines) == 3, f"unexpected output: {out.stdout!r}"
        assert int(lines[0]) == expected[("int", 0)], f"drift idx0 @seed={seed}"
        assert int(lines[1]) == expected[("int", 13)], f"drift idx13 @seed={seed}"
        assert int(lines[2]) == expected[("str", "obj_a")], f"drift str @seed={seed}"


# ---------------------------------------------------------------------------
# Script runner (no pytest required)
# ---------------------------------------------------------------------------


def _run_all() -> int:
    """Execute every ``test_*`` in this module; print a summary; return exit code."""
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = 0
    failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - we want to report everything
            failed += 1
            failures.append((name, repr(exc)))
            print(f"FAIL  {name}: {exc!r}")
        else:
            passed += 1
            print(f"ok    {name}")
    print("-" * 60)
    print(f"{passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
