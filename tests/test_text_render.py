"""Tests for ``yea_hud.text_render`` (brief §25 — text/label cases).

Runnable two ways:

    /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_text_render.py -q
    /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_text_render.py

The ``__main__`` block runs every ``test_*`` function, prints a summary, and
``sys.exit(1)`` on any failure. No ComfyUI, no network, no model files — only
PIL's built-in default font and small synthetic numpy fixtures.
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

from PIL import ImageFont  # noqa: E402

from yea_hud import text_render as tr  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _font(size=16):
    """A font from the real loader (whatever the environment provides)."""
    return tr.get_font(size)


def _default_only_font():
    """Force PIL's embedded default font, independent of any system font.

    Used where we want measurement behavior that does not depend on which
    monospace happens to be installed on the test host.
    """
    try:
        return ImageFont.load_default(size=16)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# get_font — caching + graceful fallback (never requires a user font)
# ---------------------------------------------------------------------------

def test_get_font_returns_usable_font():
    """get_font always returns a font object exposing a measure API."""
    f = tr.get_font(18)
    assert f is not None
    # Either getbbox (modern) or getsize (legacy) must exist for measuring.
    assert hasattr(f, "getbbox") or hasattr(f, "getsize")


def test_get_font_cached_by_path_and_size():
    """Same (path, size) returns the *same* cached object; different size differs."""
    tr._FONT_CACHE.clear()
    a = tr.get_font(20)
    b = tr.get_font(20)
    assert a is b  # identity: served from cache, not rebuilt
    c = tr.get_font(21)
    assert c is not a  # different size -> different cache entry


def test_get_font_empty_path_shares_default_entry():
    """font_path='' is normalized to None (shares the default-chain entry)."""
    tr._FONT_CACHE.clear()
    a = tr.get_font(16, font_path=None)
    b = tr.get_font(16, font_path="")
    assert a is b


def test_get_font_bad_path_falls_back_no_raise():
    """A non-existent / non-font path must NOT raise; it falls through."""
    tr._FONT_CACHE.clear()
    f = tr.get_font(16, font_path="/no/such/font/definitely_missing.ttf")
    assert f is not None
    # And it is still a usable font.
    assert hasattr(f, "getbbox") or hasattr(f, "getsize")


def test_get_font_non_font_path_falls_back():
    """Pointing at a real file that is NOT a font also degrades gracefully."""
    tr._FONT_CACHE.clear()
    not_a_font = os.path.abspath(__file__)  # this .py file exists but isn't a font
    f = tr.get_font(16, font_path=not_a_font)
    assert f is not None
    assert hasattr(f, "getbbox") or hasattr(f, "getsize")


def test_get_font_zero_or_negative_size_floored():
    """Non-positive sizes are floored to >=1 so PIL never rejects them."""
    tr._FONT_CACHE.clear()
    f0 = tr.get_font(0)
    fneg = tr.get_font(-5)
    assert f0 is not None
    assert fneg is not None
    # Both clamp to size 1 -> same cache key -> same object.
    assert f0 is fneg


def test_get_font_non_int_size_does_not_raise():
    """A float-ish size is coerced; junk size degrades to size 1, no raise."""
    tr._FONT_CACHE.clear()
    f = tr.get_font(14.0)  # int(14.0) == 14
    assert f is not None
    g = tr.get_font(14)
    assert f is g  # 14.0 and 14 collapse to the same int key


def test_get_font_cache_eviction_bounded():
    """The module cache evicts beyond its max so it cannot grow unbounded."""
    tr._FONT_CACHE.clear()
    cap = tr._FONT_CACHE.max_items
    for s in range(1, cap + 25):
        tr.get_font(s)
    assert len(tr._FONT_CACHE) <= cap


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

def test_measure_basic_positive():
    """A non-empty string measures to positive width and height."""
    f = _font(20)
    w, h = tr.measure("HELLO", f)
    assert isinstance(w, int) and isinstance(h, int)
    assert w > 0 and h > 0


def test_measure_empty_string_zero_width_real_height():
    """Empty string -> width 0 but a real (non-zero) line height."""
    f = _font(20)
    w, h = tr.measure("", f)
    assert w == 0
    assert h > 0


def test_measure_none_treated_as_empty():
    """None text is treated as '' (no crash)."""
    f = _font(20)
    w, h = tr.measure(None, f)
    assert w == 0 and h > 0


def test_measure_longer_text_is_wider():
    """More characters -> strictly wider (monotonic for the same font)."""
    f = _font(18)
    w_short, _ = tr.measure("A", f)
    w_long, _ = tr.measure("AAAAAAAA", f)
    assert w_long > w_short


def test_measure_non_negative_ints():
    """Width/height are always non-negative ints across a range of strings."""
    f = _font(16)
    for s in ["", " ", "x", "Tracking 03", "—", "…", "WWWWWWWWWW"]:
        w, h = tr.measure(s, f)
        assert isinstance(w, int) and isinstance(h, int)
        assert w >= 0 and h >= 0


# ---------------------------------------------------------------------------
# truncate_ellipsis
# ---------------------------------------------------------------------------

def test_truncate_fits_returned_unchanged():
    """Text that already fits is returned verbatim (no ellipsis)."""
    f = _font(16)
    text = "PERSON"
    full_w, _ = tr.measure(text, f)
    out = tr.truncate_ellipsis(text, f, full_w + 50)
    assert out == text


def test_truncate_adds_ellipsis_and_fits():
    """Overflowing text is shortened to fit and ends with an ellipsis."""
    f = _font(20)
    text = "A VERY LONG CAPTION THAT WILL NOT FIT IN A NARROW STRIP"
    full_w, _ = tr.measure(text, f)
    max_w = full_w // 3
    out = tr.truncate_ellipsis(text, f, max_w)
    assert out != text
    assert out.endswith(tr._ELLIPSIS)
    # The truncated string must actually fit the budget.
    out_w, _ = tr.measure(out, f)
    assert out_w <= max_w


def test_truncate_zero_width_returns_empty():
    """Non-positive max width -> empty string (no room)."""
    f = _font(16)
    assert tr.truncate_ellipsis("ANYTHING", f, 0) == ""
    assert tr.truncate_ellipsis("ANYTHING", f, -10) == ""


def test_truncate_tiny_width_returns_ellipsis_floor():
    """When not even the ellipsis fits, the ellipsis itself is the floor."""
    f = _font(40)
    ell_w, _ = tr.measure(tr._ELLIPSIS, f)
    # A budget smaller than a single ellipsis glyph.
    out = tr.truncate_ellipsis("HELLO WORLD", f, max(1, ell_w - 1))
    assert out == tr._ELLIPSIS


def test_truncate_empty_input_stays_empty():
    """Empty input with positive budget stays empty (nothing to truncate)."""
    f = _font(16)
    assert tr.truncate_ellipsis("", f, 100) == ""
    assert tr.truncate_ellipsis(None, f, 100) == ""


def test_truncate_is_deterministic():
    """Same inputs -> same output (pure function)."""
    f = _font(18)
    text = "DETERMINISTIC TRUNCATION CHECK"
    full_w, _ = tr.measure(text, f)
    a = tr.truncate_ellipsis(text, f, full_w // 2)
    b = tr.truncate_ellipsis(text, f, full_w // 2)
    assert a == b


# ---------------------------------------------------------------------------
# render_label_strip — straight-alpha RGBA float32 [h,w,4]
# ---------------------------------------------------------------------------

def test_strip_shape_dtype_range():
    """Returns float32 [h,w,4] in 0..1 with sensible (text+2*pad) size."""
    f = _font(20)
    pad = (6, 3)
    text = "PERSON"
    tw, th = tr.measure(text, f)
    strip = tr.render_label_strip(
        None, text=text, font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=0.9, pad=pad,
    )
    assert isinstance(strip, np.ndarray)
    assert strip.dtype == np.float32
    assert strip.ndim == 3 and strip.shape[2] == 4
    assert strip.min() >= 0.0 and strip.max() <= 1.0
    # Exact size contract: text box + symmetric padding.
    assert strip.shape[0] == max(1, th + 2 * pad[1])
    assert strip.shape[1] == max(1, tw + 2 * pad[0])


def test_strip_background_alpha_equals_opacity_in_empty_region():
    """A padding/background pixel has alpha == bg_opacity (straight alpha)."""
    f = _font(20)
    op = 0.5
    strip = tr.render_label_strip(
        None, text="X", font=f, fg_rgb=(1, 1, 1), bg_rgb=(0.1, 0.2, 0.3),
        bg_opacity=op, pad=(8, 8),
    )
    # Top-left corner is inside the padding -> pure background, no glyph ink.
    a_corner = strip[0, 0, 3]
    assert abs(float(a_corner) - op) <= 1.0 / 255.0 + 1e-4
    # Background color is preserved there (within 8-bit rounding).
    assert abs(float(strip[0, 0, 0]) - 0.1) <= 2.0 / 255.0
    assert abs(float(strip[0, 0, 1]) - 0.2) <= 2.0 / 255.0
    assert abs(float(strip[0, 0, 2]) - 0.3) <= 2.0 / 255.0


def test_strip_text_raises_alpha_above_background():
    """Somewhere under the glyph, alpha exceeds the background opacity."""
    f = _font(28)
    op = 0.4
    strip = tr.render_label_strip(
        None, text="HELLO", font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=op, pad=(6, 3),
    )
    # At least one pixel has alpha pushed toward 1 by opaque text ink.
    assert float(strip[..., 3].max()) > op + 0.2
    # And the max alpha reaches (near) full opacity where glyphs are solid.
    assert float(strip[..., 3].max()) > 0.95


def test_strip_zero_opacity_background_transparent():
    """bg_opacity=0 -> background fully transparent; only text adds alpha."""
    f = _font(24)
    strip = tr.render_label_strip(
        None, text="HI", font=f, fg_rgb=(1, 1, 1), bg_rgb=(1, 0, 0),
        bg_opacity=0.0, pad=(10, 10),
    )
    # A corner padding pixel is fully transparent.
    assert float(strip[0, 0, 3]) == 0.0
    # But the text still renders some opaque pixels.
    assert float(strip[..., 3].max()) > 0.5


def test_strip_full_opacity_background_solid_corner():
    """bg_opacity=1 -> background corner alpha is 1.0 and is the bg color."""
    f = _font(20)
    strip = tr.render_label_strip(
        None, text="A", font=f, fg_rgb=(0, 1, 0), bg_rgb=(0, 0, 1),
        bg_opacity=1.0, pad=(6, 6),
    )
    assert abs(float(strip[0, 0, 3]) - 1.0) <= 1e-6
    # corner is background blue, not text green
    assert float(strip[0, 0, 2]) > 0.9
    assert float(strip[0, 0, 1]) < 0.1


def test_strip_empty_text_min_size_strip():
    """Empty text yields a valid >=1x1 strip filled with the background."""
    f = _font(18)
    pad = (4, 2)
    strip = tr.render_label_strip(
        None, text="", font=f, fg_rgb=(1, 1, 1), bg_rgb=(0.5, 0.5, 0.5),
        bg_opacity=1.0, pad=pad,
    )
    # width = 0 + 2*pad_x ; height = lineheight + 2*pad_y
    assert strip.shape[1] == max(1, 2 * pad[0])
    assert strip.shape[0] >= 1 and strip.shape[2] == 4
    # all background (no text), alpha 1
    assert np.allclose(strip[..., 3], 1.0, atol=1e-6)


def test_strip_none_text_behaves_like_empty():
    """text=None is treated as '' (no crash, valid strip)."""
    f = _font(18)
    strip = tr.render_label_strip(
        None, text=None, font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=0.5, pad=(3, 3),
    )
    assert strip.dtype == np.float32 and strip.shape[2] == 4
    assert strip.shape[0] >= 1 and strip.shape[1] >= 1


def test_strip_scalar_pad_supported():
    """pad may be a single scalar applied to both axes."""
    f = _font(16)
    text = "OK"
    tw, th = tr.measure(text, f)
    strip = tr.render_label_strip(
        None, text=text, font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=1.0, pad=5,
    )
    assert strip.shape[0] == max(1, th + 10)
    assert strip.shape[1] == max(1, tw + 10)


def test_strip_negative_pad_floored():
    """Negative padding is floored to 0 (no negative dimensions)."""
    f = _font(16)
    text = "Z"
    tw, th = tr.measure(text, f)
    strip = tr.render_label_strip(
        None, text=text, font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=1.0, pad=(-4, -4),
    )
    assert strip.shape[0] == max(1, th)
    assert strip.shape[1] == max(1, tw)


def test_strip_color_clamped():
    """Out-of-range colors are clamped into 0..1 in the output."""
    f = _font(16)
    strip = tr.render_label_strip(
        None, text="", font=f, fg_rgb=(2.0, -1.0, 0.5), bg_rgb=(5.0, -2.0, 0.25),
        bg_opacity=1.0, pad=(4, 4),
    )
    corner = strip[0, 0]
    assert corner[0] >= 0.99   # 5.0 -> clamped to 1
    assert corner[1] <= 0.01   # -2.0 -> clamped to 0
    assert abs(float(corner[2]) - 0.25) <= 2.0 / 255.0
    assert strip.min() >= 0.0 and strip.max() <= 1.0


def test_strip_opacity_clamped():
    """Out-of-range bg_opacity is clamped to 0..1."""
    f = _font(16)
    over = tr.render_label_strip(
        None, text="", font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=4.0, pad=(3, 3),
    )
    under = tr.render_label_strip(
        None, text="", font=f, fg_rgb=(1, 1, 1), bg_rgb=(1, 1, 1),
        bg_opacity=-2.0, pad=(3, 3),
    )
    assert np.allclose(over[..., 3], 1.0, atol=1e-6)   # 4.0 -> 1.0
    assert np.allclose(under[..., 3], 0.0, atol=1e-6)  # -2.0 -> 0.0


def test_strip_straight_alpha_not_premultiplied():
    """Background RGB is stored straight (NOT premultiplied by its alpha).

    With a half-opaque red background, the stored R channel must be ~1.0
    (straight), not ~0.5 (which is what premultiplication would produce).
    """
    f = _font(16)
    strip = tr.render_label_strip(
        None, text="", font=f, fg_rgb=(1, 1, 1), bg_rgb=(1.0, 0.0, 0.0),
        bg_opacity=0.5, pad=(4, 4),
    )
    corner = strip[0, 0]
    assert abs(float(corner[0]) - 1.0) <= 2.0 / 255.0   # straight red, not 0.5
    assert abs(float(corner[3]) - 0.5) <= 1.0 / 255.0 + 1e-4


def test_strip_deterministic():
    """Identical inputs produce byte-identical strips (pure render)."""
    f = _font(20)
    kw = dict(text="REPEAT 01", font=f, fg_rgb=(0.9, 0.1, 0.1),
              bg_rgb=(0, 0, 0), bg_opacity=0.8, pad=(6, 3))
    a = tr.render_label_strip(None, **kw)
    b = tr.render_label_strip(None, **kw)
    assert np.array_equal(a, b)


def test_strip_with_default_font_only():
    """Works with PIL's embedded default font (the locked-down Cloud case)."""
    f = _default_only_font()
    strip = tr.render_label_strip(
        None, text="CLOUD", font=f, fg_rgb=(1, 1, 1), bg_rgb=(0, 0, 0),
        bg_opacity=0.9, pad=(6, 3),
    )
    assert strip.dtype == np.float32 and strip.shape[2] == 4
    assert strip.shape[0] >= 1 and strip.shape[1] >= 1


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
    print(f"text_render tests: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFAILURES:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
