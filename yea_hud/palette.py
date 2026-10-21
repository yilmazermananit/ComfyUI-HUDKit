"""Color resolution for ComfyUI-HUDKit (brief §14).

This module is schema-agnostic: it imports only the Python standard library and
sibling :mod:`yea_hud` modules (``types`` for the palette constants, ``utils``
for the deterministic hash). It contains **no** ComfyUI imports and does no I/O,
networking, subprocess, ``eval``/``exec`` or any non-deterministic work, so it is
safe to run inside Comfy Cloud.

It owns four concerns:

* **Hex parsing** (:func:`parse_hex`) — turn a ``'#RRGGBB'`` / ``'RRGGBB'`` string
  into an ``(r, g, b)`` float triple in ``0..1``, or ``None`` when it is not a
  valid 6-digit hex color.
* **Palette resolution** (:func:`resolve_palette`) — parse a JSON list of hex
  strings, drop the invalid entries, and guarantee a non-empty palette by falling
  back to :data:`yea_hud.types.DEFAULT_PALETTE`.
* **Deterministic per-track color** (:func:`color_for_track`) — pick a palette
  index from ``stable_hash(seed, track_id)`` so a given ``(seed, track_id)`` always
  draws the same color on every machine / run (seed-stable HUD identity).
* **Readable label text** (:func:`contrast_text_color` / :func:`luminance`) —
  choose black or white text over a colored strip using a Rec.709 luma threshold.

RGB colors are ``(r, g, b)`` floats in ``0..1`` everywhere, matching the rest of
the pack (see ``docs/INTERFACES.md`` and ``palette.luminance`` referenced by
:mod:`yea_hud.alpha`).
"""

from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple, Union

from .types import DEFAULT_PALETTE
from .utils import stable_hash


__all__ = [
    "parse_hex",
    "resolve_palette",
    "color_for_track",
    "contrast_text_color",
    "luminance",
]

# A 3-tuple of floats in 0..1.
RGB = Tuple[float, float, float]

# Rec.709 luma coefficients. These are the same coefficients used in
# :mod:`yea_hud.alpha` (``_LUMA``) so the whole pack agrees on "brightness".
# Applied directly to the already-display (gamma-encoded) sRGB values — this is
# the standard "luma" (Y') convention, NOT a linear-light luminance, and it is
# what we want for a fast, perceptual-enough black-vs-white text decision.
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722

# Above this luma the background is "bright" -> use black text; at or below it ->
# white text. Specified by the contract (brief §13.5 / INTERFACES palette.py).
_AUTO_TEXT_LUMA_THRESHOLD = 0.55

# Pure black / white text colors.
_BLACK: RGB = (0.0, 0.0, 0.0)
_WHITE: RGB = (1.0, 1.0, 1.0)


def _clip01(value: float) -> float:
    """Clamp a single float to the closed unit interval ``[0.0, 1.0]``.

    Non-finite inputs (``NaN`` / ``inf``) are folded to ``0.0`` so that a
    degenerate color component can never poison a downstream luma computation or
    escape the ``0..1`` contract.
    """
    # NaN compares False to everything, so this also catches NaN -> 0.0.
    if not (value == value):  # NaN
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def parse_hex(s: str) -> Optional[RGB]:
    """Parse a 6-digit hex color string into an ``(r, g, b)`` triple in ``0..1``.

    Accepts ``'#RRGGBB'`` or ``'RRGGBB'`` (case-insensitive). Surrounding
    whitespace is ignored. A single leading ``'#'`` is optional. Any input that is
    not exactly six hexadecimal digits (after stripping an optional ``'#'``) is
    rejected and returns ``None`` — this includes empty strings, ``None``, the 3-
    digit short form, 8-digit ``#RRGGBBAA``, names like ``'red'`` and any
    non-string type. Rejecting (rather than raising) lets :func:`resolve_palette`
    silently drop bad entries.

    Args:
        s: The candidate hex string (e.g. ``'#FF2A1A'`` or ``'ff2a1a'``).

    Returns:
        The ``(r, g, b)`` floats in ``0..1`` (each channel = byte / 255), or
        ``None`` if ``s`` is not a valid 6-digit hex color.
    """
    if not isinstance(s, str):
        return None
    text = s.strip()
    if not text:
        return None
    if text[0] == "#":
        text = text[1:]
    # Exactly six hex digits; ``int(..., 16)`` plus an explicit length/charset
    # guard. We avoid ``int`` accepting things like ``'0x..'`` or a leading sign by
    # validating the character set ourselves.
    if len(text) != 6:
        return None
    # ``str.isascii`` + per-char hexdigit check: reject any non-ASCII or non-hex.
    lower = text.lower()
    for ch in lower:
        if ch not in "0123456789abcdef":
            return None
    r = int(lower[0:2], 16)
    g = int(lower[2:4], 16)
    b = int(lower[4:6], 16)
    return (r / 255.0, g / 255.0, b / 255.0)


def resolve_palette(palette_json: Optional[str]) -> list:
    """Resolve a JSON palette string into a non-empty list of RGB triples.

    The expected input is a JSON array of hex strings, e.g.
    ``'["#FF2A1A", "#225CFF"]'``. Parsing is intentionally forgiving so a user
    typo never breaks rendering:

    * ``None``, empty / whitespace-only string, or text that is not valid JSON
      -> :data:`yea_hud.types.DEFAULT_PALETTE`.
    * Valid JSON that is not a list (object / number / bare string) -> default.
    * A list whose entries are filtered through :func:`parse_hex`; non-string and
      invalid-hex entries are dropped. (A bare JSON string that happens to be a
      single hex value is also accepted as a one-element palette, for
      convenience.)
    * If *every* entry was dropped (or the list was empty) -> default.

    The result is **always** at least one color (the contract guarantees
    ``len >= 1``), so callers can index/modulo into it unconditionally.

    Args:
        palette_json: JSON array of hex strings, or ``None``.

    Returns:
        list[tuple[float, float, float]]: One or more RGB triples in ``0..1``.
    """
    raw_entries: Optional[Sequence] = None

    if isinstance(palette_json, str):
        text = palette_json.strip()
        if text:
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, str):
                # Convenience: a bare JSON string ``"\"#FF2A1A\""`` is treated as a
                # single-element palette rather than rejected outright.
                raw_entries = [parsed]
            # Any other JSON type (dict/number/bool/None) -> fall through to default.

    colors: list = []
    if raw_entries is not None:
        for entry in raw_entries:
            rgb = parse_hex(entry) if isinstance(entry, str) else None
            if rgb is not None:
                colors.append(rgb)

    if not colors:
        # Fall back to the bundled default palette. DEFAULT_PALETTE is a tuple of
        # hex strings; convert each to an RGB triple. These are known-valid, but
        # we still go through parse_hex so there is a single parsing path.
        colors = []
        for hex_str in DEFAULT_PALETTE:
            rgb = parse_hex(hex_str)
            if rgb is not None:
                colors.append(rgb)
        # Last-ditch guarantee: even if DEFAULT_PALETTE were somehow empty/bad,
        # never return an empty list (the contract promises >= 1 color).
        if not colors:
            colors = [_WHITE]

    return colors


def color_for_track(
    palette: Sequence[RGB],
    seed: int,
    track_id: Union[int, str],
) -> Tuple[int, RGB]:
    """Deterministically pick a palette color for a track.

    The index is ``stable_hash(seed, track_id) % len(palette)`` using
    :func:`yea_hud.utils.stable_hash`, which is independent of
    ``PYTHONHASHSEED`` and stable across runs / processes / operating systems.
    The same ``(seed, track_id)`` therefore always maps to the same color — this
    is the seam that gives every tracked object a stable HUD identity (brief
    §10.5 / §11.1). ``track_id`` may be an ``int`` or a ``str`` (see
    :class:`yea_hud.types.Detection.source_track_id`); both are hashed
    type-safely.

    Args:
        palette: A non-empty sequence of RGB triples (typically the output of
            :func:`resolve_palette`).
        seed: The user-facing HUD seed.
        track_id: The track identity (``int`` or ``str``).

    Returns:
        ``(idx, rgb)`` where ``idx`` is the chosen palette index and ``rgb`` is
        ``palette[idx]``.

    Raises:
        ValueError: If ``palette`` is empty. Callers should pass the result of
            :func:`resolve_palette`, which is guaranteed non-empty.
    """
    n = len(palette)
    if n <= 0:
        raise ValueError("color_for_track requires a non-empty palette")
    idx = stable_hash(seed, track_id) % n
    return idx, tuple(palette[idx])  # type: ignore[return-value]


def luminance(rgb: Sequence[float]) -> float:
    """Return the Rec.709 luma (Y') of an ``(r, g, b)`` color in ``0..1``.

    Computed as ``0.2126*r + 0.7152*g + 0.0722*b`` directly on the supplied
    (display-encoded sRGB) values — the standard luma convention shared with
    :mod:`yea_hud.alpha`. Each channel is clamped to ``0..1`` first, so the
    returned value is always in ``0..1`` even if a caller passes a slightly
    out-of-range or non-finite component.

    Args:
        rgb: An ``(r, g, b)`` triple (any indexable of length >= 3) in ``0..1``.

    Returns:
        float: The luma in ``0..1``.
    """
    r = _clip01(float(rgb[0]))
    g = _clip01(float(rgb[1]))
    b = _clip01(float(rgb[2]))
    return _LUMA_R * r + _LUMA_G * g + _LUMA_B * b


def contrast_text_color(bg_rgb: Sequence[float], mode: str = "auto") -> RGB:
    """Pick a readable text color (black or white) for a colored background.

    Args:
        bg_rgb: The background ``(r, g, b)`` triple in ``0..1`` (e.g. the track's
            label-strip color).
        mode: One of :data:`yea_hud.types.LABEL_TEXT_COLOR_MODES`:

            * ``'white'`` — always return white.
            * ``'black'`` — always return black.
            * ``'auto'`` (default, and the fallback for any unrecognized value) —
              return **black** when ``luminance(bg_rgb) > 0.55`` (a bright
              background), otherwise **white**. The ``>`` (strict) comparison
              means a background exactly at the 0.55 threshold yields white.

    Returns:
        Either ``(0.0, 0.0, 0.0)`` (black) or ``(1.0, 1.0, 1.0)`` (white).
    """
    if mode == "white":
        return _WHITE
    if mode == "black":
        return _BLACK
    # 'auto' (and any unknown mode, defensively): luma-based decision.
    return _BLACK if luminance(bg_rgb) > _AUTO_TEXT_LUMA_THRESHOLD else _WHITE
