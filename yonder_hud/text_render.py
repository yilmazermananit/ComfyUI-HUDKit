"""text_render.py — font loading + caption label strips (brief §13.5).

Schema-agnostic. Depends only on the Python standard library, :mod:`numpy`,
:mod:`PIL` (Pillow) and :mod:`yonder_hud.utils` / :mod:`yonder_hud.types`. It
performs **no** ComfyUI imports, no network, no subprocess, no ``eval``/``exec``,
and no arbitrary-user-path file IO — pure deterministic compute that is safe to
run inside Comfy Cloud.

What this module provides
-------------------------
* :func:`get_font` — an LRU-cached font loader keyed by ``(path, size)``. It tries
  a small, *fixed* allow-list of well-known open monospace fonts (any bundled font
  shipped inside this package, then standard OS monospace locations) and **always**
  falls back to :func:`PIL.ImageFont.load_default`. It never *requires* a user font
  file, and it never reads an arbitrary user-supplied path except the explicit
  ``font_path`` the caller passes in (that one is opened defensively and, on any
  failure, silently degrades to the bundled/default chain).
* :func:`measure` — measure a single line of text ``(w, h)`` in pixels.
* :func:`truncate_ellipsis` — shorten text with a ``'…'`` so it fits a max width.
* :func:`render_label_strip` — render a solid background label strip with text,
  returning a straight-alpha RGBA ``float32 [h, w, 4]`` numpy array.

Font policy / Cloud safety (see ``docs/cloud_compatibility.md`` §5)
-------------------------------------------------------------------
The HUD must look identical whether or not the host machine happens to have a
particular font installed, and it must keep working on a locked-down Cloud
sandbox where *no* font may be readable. The loader therefore degrades
gracefully and is fully deterministic for a given environment: the same
``(path, size)`` always returns the same cached font object.

Color / array conventions (see ``docs/INTERFACES.md``)
------------------------------------------------------
* RGB colors are ``(r, g, b)`` floats in ``0..1``.
* The returned strip is numpy ``float32 [h, w, 4]`` RGBA with **straight**
  (non-premultiplied) alpha, matching :mod:`yonder_hud.alpha` overlays.
* Uppercasing of caption text is the *caller's* responsibility (see
  :mod:`yonder_hud.captions`); this module renders whatever string it is given.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import LRUCache

__all__ = [
    "get_font",
    "measure",
    "truncate_ellipsis",
    "render_label_strip",
]

# ---------------------------------------------------------------------------
# Module-level font cache, keyed by (path, size). ``path`` is the *requested*
# path (or None) so a miss vs. a custom path is distinguishable from the default
# chain. Values are PIL font objects (FreeTypeFont or the bitmap default).
# ---------------------------------------------------------------------------

#: Default LRU capacity for the font cache. Fonts are cheap to keep; a HUD never
#: uses more than a handful of distinct sizes, so this is comfortably large.
_FONT_CACHE_MAX = 64

#: Module-level cache. Exposed (underscore-private) so tests can clear it.
_FONT_CACHE = LRUCache(max_items=_FONT_CACHE_MAX)

#: Ellipsis glyph used by :func:`truncate_ellipsis`. A single Unicode horizontal
#: ellipsis (U+2026); the default bitmap font may render it as a fallback box but
#: the *width arithmetic* stays correct regardless.
_ELLIPSIS = "…"

# A fixed, curated allow-list of open monospace fonts to try, in order. This is a
# CONSTANT list — never built from user input — so the loader cannot be coerced
# into reading an arbitrary path. The first entry that exists and opens wins;
# otherwise we fall through to PIL's bundled default. Covers Linux (DejaVu /
# Liberation, the common Cloud/CI case), macOS, and Windows.
_MONO_FONT_CANDIDATES: Tuple[str, ...] = (
    # Linux — DejaVu (ships with most distros, matplotlib, and many CI images)
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    # Linux — Liberation Mono (RHEL/Fedora/Amazon Linux family)
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
    # Linux — other common monospace
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    # macOS
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Courier.ttc",
    "/Library/Fonts/Courier New.ttf",
    # Windows
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
)


def _bundled_font_dir() -> str:
    """Absolute path of an optional ``fonts/`` directory bundled in this package.

    Reading from *here* is explicitly allowed (it is package-internal, not a
    user path — see ``docs/cloud_compatibility.md`` §5). The directory may not
    exist; callers must handle that.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _iter_bundled_fonts() -> List[str]:
    """Return candidate absolute paths of bundled ``.ttf``/``.otf`` fonts.

    Deterministic order (sorted by filename) so the chosen font is stable across
    runs and operating systems. Returns ``[]`` if no bundled ``fonts/`` directory
    exists. Any error while listing the directory is swallowed (returns ``[]``).
    """
    fonts_dir = _bundled_font_dir()
    try:
        names = sorted(os.listdir(fonts_dir))
    except OSError:
        return []
    out: List[str] = []
    for name in names:
        if name.lower().endswith((".ttf", ".otf", ".ttc")):
            out.append(os.path.join(fonts_dir, name))
    return out


def _try_truetype(path: str, size: int):
    """Try to open ``path`` as a scalable font at ``size``; return it or ``None``.

    Never raises: any :class:`OSError` / :class:`ValueError` (missing file,
    unreadable, not a font) results in ``None`` so the caller can fall through to
    the next candidate. ``path`` must be a real, existing file — we check first to
    avoid surfacing noisy errors for the (common) missing-font case.
    """
    if not path:
        return None
    try:
        if not os.path.isfile(path):
            return None
        return ImageFont.truetype(path, size=size)
    except (OSError, ValueError):
        return None


def _load_default(size: int):
    """Return PIL's built-in default font, scaled to ``size`` when supported.

    Pillow >= 10 ships a scalable Type-1 default and accepts a ``size`` argument
    to :func:`PIL.ImageFont.load_default`; older Pillow returns a fixed-size
    bitmap font and ignores ``size``. We try the sized form first and fall back
    to the no-arg form so this works on every supported Pillow. This call cannot
    fail in practice (the default font is embedded in Pillow), but it is still
    wrapped so the loader is total.
    """
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Old Pillow signature: load_default() takes no argument.
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001 — last-resort safety; default must never throw
        return ImageFont.load_default()


def get_font(size: int, font_path: Optional[str] = None) -> "ImageFont.ImageFont":
    """Return a PIL font for ``size``, cached by ``(font_path, size)``.

    Resolution order:

    1. If ``font_path`` is given and opens as a scalable font, use it. (This is
       the *only* user-supplied path the loader will read, by explicit request;
       on any failure it degrades quietly to the chain below — a bad path NEVER
       raises.)
    2. Any font bundled in this package's ``fonts/`` directory (deterministic,
       sorted order).
    3. A fixed allow-list of well-known open **monospace** system fonts (DejaVu /
       Liberation / Menlo / Courier / Consolas), tried in order.
    4. :func:`PIL.ImageFont.load_default` (always succeeds) — so a user font is
       **never required** and the function always returns a usable font.

    The chosen font for a given ``(font_path, size)`` is cached in a module-level
    :class:`~yonder_hud.utils.LRUCache`, so repeated calls (one per frame, per
    track) are O(1) and return the *same* object.

    Args:
        size: Pixel size for the font. Coerced to ``int`` and floored at ``1``
            (a non-positive size would make PIL raise / produce a zero-height
            glyph).
        font_path: Optional explicit font file path. ``None`` (default) uses the
            bundled/system/default chain. A path that does not exist or is not a
            font is ignored (falls through), never raised.

    Returns:
        A ``PIL.ImageFont`` object (``FreeTypeFont`` when a scalable font was
        found, otherwise the embedded default). Suitable for
        :meth:`PIL.ImageDraw.ImageDraw.text` and for :func:`measure`.
    """
    # Normalise size defensively: PIL rejects size <= 0 for truetype fonts.
    try:
        isize = int(size)
    except (TypeError, ValueError):
        isize = 1
    if isize < 1:
        isize = 1

    # Normalise the path component of the cache key (empty string -> None) so
    # get_font(s) and get_font(s, "") share a cache entry.
    key_path = font_path if font_path else None
    cache_key = (key_path, isize)

    cached = _FONT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    font = None

    # (1) explicit user path — the one allowed user-path read, fully defensive.
    if key_path is not None:
        font = _try_truetype(str(key_path), isize)

    # (2) bundled package fonts.
    if font is None:
        for cand in _iter_bundled_fonts():
            font = _try_truetype(cand, isize)
            if font is not None:
                break

    # (3) curated system monospace allow-list.
    if font is None:
        for cand in _MONO_FONT_CANDIDATES:
            font = _try_truetype(cand, isize)
            if font is not None:
                break

    # (4) PIL embedded default — always succeeds.
    if font is None:
        font = _load_default(isize)

    _FONT_CACHE.put(cache_key, font)
    return font


def _measure_with_draw(text: str, font) -> Tuple[int, int]:
    """Internal: measure ``text`` using the most accurate API the font supports.

    Prefers ``font.getbbox`` (FreeType + modern bitmap fonts), which returns the
    glyph ink box ``(left, top, right, bottom)``; width is ``right - left`` and
    height is ``bottom`` (top-anchored, so the strip reserves room for descenders
    and the natural top sidebearing). Falls back to ``font.getsize`` on very old
    Pillow, and finally to a draw-based ``textbbox`` measurement. All paths return
    non-negative integers.
    """
    # Modern path: getbbox -> (l, t, r, b).
    getbbox = getattr(font, "getbbox", None)
    if callable(getbbox):
        try:
            left, top, right, bottom = getbbox(text)
            w = int(round(max(0.0, right - left)))
            # Use the full descent extent (bottom) as height so descenders ('g',
            # 'p') are not clipped. top is typically small/zero for top anchoring.
            h = int(round(max(0.0, bottom)))
            return w, h
        except Exception:  # noqa: BLE001 — fall through to other measurers
            pass

    # Legacy Pillow: getsize -> (w, h).
    getsize = getattr(font, "getsize", None)
    if callable(getsize):
        try:
            w, h = getsize(text)
            return int(round(max(0.0, w))), int(round(max(0.0, h)))
        except Exception:  # noqa: BLE001
            pass

    # Last resort: measure via a throwaway 1x1 draw context.
    tmp = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(tmp)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return int(round(max(0.0, right - left))), int(round(max(0.0, bottom - top)))


def measure(text: str, font) -> Tuple[int, int]:
    """Measure a single line of ``text`` in pixels for ``font``.

    Returns ``(width, height)`` as non-negative ints. An empty string measures as
    ``(0, line_height)`` — zero width but the font's natural line height — so an
    empty label still has a sensible strip height; ``None`` is treated as ``""``.

    Newlines are **not** interpreted (this is a single-line measurer, matching the
    single-line label strips the HUD draws); any ``'\\n'`` is measured literally.

    Args:
        text: The string to measure (``None`` -> ``""``).
        font: A ``PIL.ImageFont`` (e.g. from :func:`get_font`).

    Returns:
        ``(w, h)`` in integer pixels, both ``>= 0``.
    """
    s = "" if text is None else str(text)
    if s == "":
        # Width 0 but keep a real line height (measure a representative glyph).
        _, h = _measure_with_draw("Ag", font)
        return 0, int(max(0, h))
    w, h = _measure_with_draw(s, font)
    return int(max(0, w)), int(max(0, h))


def _text_width(text: str, font) -> int:
    """Pixel advance width of ``text`` (height ignored). Non-negative int."""
    if text == "":
        return 0
    w, _ = _measure_with_draw(text, font)
    return int(max(0, w))


def truncate_ellipsis(text: str, font, max_w: int) -> str:
    """Truncate ``text`` with a trailing ``'…'`` so it fits within ``max_w`` px.

    If the full text already fits, it is returned unchanged. Otherwise characters
    are dropped from the end (a deterministic linear scan) and a single ellipsis
    is appended; the longest prefix whose ``prefix + '…'`` width is ``<= max_w`` is
    chosen. If not even the ellipsis alone fits (``max_w`` smaller than one
    ellipsis glyph), the ellipsis is still returned (callers clip to the strip
    rect), because returning an empty string would silently drop the label.

    Args:
        text: Source string (``None`` -> ``""``).
        font: A ``PIL.ImageFont`` (e.g. from :func:`get_font`).
        max_w: Maximum allowed width in pixels. Non-positive ``max_w`` yields
            ``""`` (no room for any text at all).

    Returns:
        A string that, when rendered with ``font``, is no wider than ``max_w``
        (except the single-ellipsis floor described above).
    """
    s = "" if text is None else str(text)

    try:
        limit = int(max_w)
    except (TypeError, ValueError):
        limit = 0

    if limit <= 0:
        return ""

    # Fast path: it already fits.
    if _text_width(s, font) <= limit:
        return s

    if s == "":
        return ""

    ell_w = _text_width(_ELLIPSIS, font)

    # If even the ellipsis alone is wider than the budget, there is no honest way
    # to fit anything; return the ellipsis (the compositor clips to the rect) so
    # the label's presence is still indicated rather than vanishing entirely.
    if ell_w > limit:
        return _ELLIPSIS

    # Grow the prefix as long as prefix + ellipsis fits. Linear, deterministic.
    best = ""
    for i in range(1, len(s) + 1):
        prefix = s[:i]
        if _text_width(prefix, font) + ell_w <= limit:
            best = prefix
        else:
            break
    return best + _ELLIPSIS


def _clip01(value: float) -> float:
    """Clamp a scalar to ``[0.0, 1.0]`` as a Python float."""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _rgb_to_u8(rgb) -> Tuple[int, int, int]:
    """Convert an ``(r, g, b)`` 0..1 (clamped) triple to 0..255 ints."""
    r = int(round(_clip01(float(rgb[0])) * 255.0))
    g = int(round(_clip01(float(rgb[1])) * 255.0))
    b = int(round(_clip01(float(rgb[2])) * 255.0))
    return r, g, b


def _normalize_pad(pad) -> Tuple[int, int]:
    """Normalise the ``pad`` argument to a non-negative ``(px, py)`` int pair.

    Accepts a 2-tuple/list ``(px, py)`` or a single scalar (applied to both
    axes). Negative values are floored at ``0``.
    """
    if isinstance(pad, (tuple, list)):
        if len(pad) >= 2:
            px, py = pad[0], pad[1]
        elif len(pad) == 1:
            px = py = pad[0]
        else:
            px = py = 0
    else:
        px = py = pad
    try:
        px = max(0, int(px))
    except (TypeError, ValueError):
        px = 0
    try:
        py = max(0, int(py))
    except (TypeError, ValueError):
        py = 0
    return px, py


def render_label_strip(
    draw_or_size=None,
    *,
    text: str,
    font,
    fg_rgb,
    bg_rgb,
    bg_opacity,
    pad=(6, 3),
) -> np.ndarray:
    """Render a single-line label strip and return straight-alpha RGBA.

    The strip is sized to the measured text plus ``pad`` on each side::

        strip_w = text_w + 2 * pad_x
        strip_h = text_h + 2 * pad_y

    and is composed of:

    * a solid background of ``bg_rgb`` at alpha ``bg_opacity`` (covering the whole
      strip), and
    * the ``text`` drawn in ``fg_rgb`` (fully opaque) inset by ``pad``.

    Straight (non-premultiplied) alpha is used throughout, matching the overlay
    convention in :mod:`yonder_hud.alpha` and :mod:`yonder_hud.compositor`: the
    background fills alpha ``bg_opacity`` and the text pixels raise alpha toward
    ``1.0`` where glyphs are inked (so anti-aliased text edges blend correctly
    over whatever the strip is later composited onto).

    The text is rendered in whatever case it is given — **uppercasing is the
    caller's responsibility** (see :mod:`yonder_hud.captions`). The font should be
    a monospace from :func:`get_font` for the intended HUD look, but any PIL font
    works.

    Args:
        draw_or_size: Ignored / compatibility slot. The contract names this
            parameter ``draw_or_size`` because an earlier design considered
            drawing directly onto a caller-provided ``ImageDraw``; the agreed
            behavior is to return a *standalone* strip array (the compositor
            blits it itself), so this argument is accepted for signature
            compatibility and otherwise unused. May be ``None``.
        text: The label string (``None`` -> ``""`` -> an empty/min-size strip).
        font: A ``PIL.ImageFont`` (e.g. from :func:`get_font`).
        fg_rgb: Foreground/text color ``(r, g, b)`` in ``0..1`` (clamped).
        bg_rgb: Background color ``(r, g, b)`` in ``0..1`` (clamped).
        bg_opacity: Background alpha in ``0..1`` (clamped). ``0`` -> transparent
            background (text only); ``1`` -> solid.
        pad: ``(pad_x, pad_y)`` padding in pixels around the text (or a single
            scalar for both axes). Negative values floored at ``0``.

    Returns:
        ``np.ndarray`` of shape ``[h, w, 4]``, dtype ``float32``, values in
        ``0..1`` (straight alpha). ``h``/``w`` are always ``>= 1``.
    """
    s = "" if text is None else str(text)
    pad_x, pad_y = _normalize_pad(pad)

    text_w, text_h = measure(s, font)

    # Strip dimensions: text box + symmetric padding. Always at least 1x1 so the
    # array is well-formed even for an empty label / zero pad.
    strip_w = max(1, int(text_w) + 2 * pad_x)
    strip_h = max(1, int(text_h) + 2 * pad_y)

    bg_a = _clip01(float(bg_opacity))
    bg_r, bg_g, bg_b = _rgb_to_u8(bg_rgb)
    fg_r, fg_g, fg_b = _rgb_to_u8(fg_rgb)

    # Build the strip in 8-bit RGBA with a transparent canvas, then:
    #  1. paint the solid background rectangle at bg_opacity,
    #  2. draw the text fully opaque on top.
    # PIL's RGBA ImageDraw with an alpha in the fill color composites using
    # straight alpha (source-over), which is exactly the overlay model we want.
    canvas = Image.new("RGBA", (strip_w, strip_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    bg_a8 = int(round(bg_a * 255.0))
    if bg_a8 > 0:
        draw.rectangle(
            [0, 0, strip_w - 1, strip_h - 1],
            fill=(bg_r, bg_g, bg_b, bg_a8),
        )

    if s != "":
        # Anchor the text at the padding origin. Use the glyph's own top offset
        # so the visible ink sits inside the padded box consistently with how it
        # was measured (getbbox top is folded into the y origin).
        top_off = 0
        getbbox = getattr(font, "getbbox", None)
        if callable(getbbox):
            try:
                top_off = int(getbbox(s)[1])
            except Exception:  # noqa: BLE001
                top_off = 0
        draw.text(
            (pad_x, pad_y - top_off),
            s,
            font=font,
            fill=(fg_r, fg_g, fg_b, 255),
        )

    out = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(out)
