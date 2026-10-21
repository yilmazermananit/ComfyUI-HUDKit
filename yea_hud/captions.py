"""Caption text resolution for ComfyUI-HUDKit (brief §13).

This module turns a per-detection rendering request into the *raw* caption
string that the HUD label strip will display. It is intentionally
schema-agnostic (no ComfyUI imports) and depends only on the Python standard
library plus :mod:`yea_hud.types`.

Design contract (see ``docs/INTERFACES.md`` → "## captions.py"):

* :func:`parse_captions_json` is *tolerant*: any malformed / ``None`` / non-dict
  JSON yields ``{}`` instead of raising. It recognises two layouts that may be
  mixed in the same document:

  - by-track   ``{"1": "ALPHA", "2": "BRAVO"}`` (top-level string keys)
  - by-frame   ``{"frames": {"0": ["A", "B"], "12": "SOLO"}}``

* :func:`resolve_caption` maps one of :data:`yea_hud.types.CAPTION_SOURCES`
  onto an actual string. It must **never raise** on bad input — most notably
  on a malformed ``template`` (an unknown ``{token}`` is left literal, an
  unbalanced brace degrades gracefully). It returns the *raw* string; the
  caller (compositor / renderer) is responsible for uppercasing + truncation.

* :func:`format_order_label` builds the ``"03. PERSON"`` prefix used when the
  renderer shows display order. The caller uppercases the final result.

Coordinate convention: detections carry absolute-pixel ``xyxy`` (see
:class:`yea_hud.types.Detection`). Template ``{x}{y}{w}{h}`` therefore expand
to the box top-left corner and its width/height in integer pixels.
"""

from __future__ import annotations

import json
import string
from typing import Any, Optional

from .types import CAPTION_SOURCES, Detection

__all__ = [
    "parse_captions_json",
    "resolve_caption",
    "format_order_label",
    "expand_template",
    "build_template_context",
]


# ---------------------------------------------------------------------------
# JSON parsing (tolerant)
# ---------------------------------------------------------------------------

def parse_captions_json(s: Optional[str]) -> dict:
    """Parse a captions JSON document into a plain ``dict``.

    The function is deliberately forgiving so a typo in a text widget can never
    crash a render. It returns ``{}`` for any of the following:

    * ``s`` is ``None`` or not a ``str`` (e.g. already a list / number).
    * ``s`` is empty or whitespace-only.
    * ``s`` is not valid JSON.
    * ``s`` is valid JSON but the top-level value is not an object/dict
      (a bare list, string, number, ``true``/``false``/``null``).

    On success the parsed mapping is returned **as-is** (string keys preserved).
    Two layouts are understood downstream by :func:`resolve_caption`:

    * by-track: top-level ``{"<track_id>": "<text>"}`` entries.
    * by-frame: a nested ``{"frames": {"<frame_index>": <text or list>}}`` map.

    The two may coexist in one document; this parser does not validate or
    reshape them, it only guarantees a ``dict`` comes back.

    Parameters
    ----------
    s:
        Raw JSON text from a node string widget, or ``None``.

    Returns
    -------
    dict
        The parsed mapping, or ``{}`` when the input is missing / invalid.
    """
    if s is None:
        return {}
    # Accept only genuine strings; a non-str (already-parsed structure, bytes,
    # an int, ...) is treated as "no usable JSON text" per the contract.
    if not isinstance(s, str):
        return {}
    text = s.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # ValueError covers json.JSONDecodeError; TypeError guards exotic inputs.
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _coerce_caption_value(value: Any, index: Optional[int]) -> str:
    """Coerce one resolved caption *value* into a display string.

    Captions in JSON may legitimately be:

    * a plain string -> returned verbatim.
    * a list/tuple   -> indexed by ``index`` (the detection's per-frame slot);
      out-of-range / ``None`` index falls back to the first element, then ``""``.
    * a number / bool -> ``str()`` of it (lets ``{"1": 42}`` work).
    * ``None``        -> ``""`` (explicit blank).
    * a dict / other  -> ``""`` (no sensible single-line rendering).

    Never raises.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        # bool is an int subclass; handle before the (int/float) branch so we
        # render "True"/"False" rather than "1"/"0".
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        # Choose the element for this detection's slot when available.
        if index is not None and isinstance(index, int) and 0 <= index < len(value):
            return _coerce_caption_value(value[index], None)
        # Fall back to the first element for a single shared caption.
        return _coerce_caption_value(value[0], None)
    # dicts / objects / anything else: no single-line representation.
    return ""


# ---------------------------------------------------------------------------
# Template expansion (never raises)
# ---------------------------------------------------------------------------

#: The tokens the brief guarantees for the ``template`` caption source. Any
#: other ``{token}`` is left literal (kept verbatim, braces included).
TEMPLATE_TOKENS: tuple[str, ...] = (
    "track", "frame", "index", "label", "score", "x", "y", "w", "h",
)


def _fmt_score(score: Optional[float]) -> str:
    """Format a detection score for a template, defensively.

    ``None`` / non-numeric -> ``""``. A finite number renders with 2 decimals
    (e.g. ``0.94``). Non-finite (``nan``/``inf``) renders as ``str(float(...))``
    so it never raises and stays debuggable.
    """
    if score is None:
        return ""
    try:
        f = float(score)
    except (TypeError, ValueError):
        return ""
    # math.isfinite without importing math for one call site.
    if f != f or f in (float("inf"), float("-inf")):
        return str(f)
    return f"{f:.2f}"


def build_template_context(
    *,
    det: Optional[Detection],
    track_id: Any,
    order_index: Any,
    frame_index: Any,
) -> dict:
    """Build the mapping of ``{token}`` -> replacement string for a template.

    All values are pre-stringified so substitution can never raise on a type
    mismatch. Geometry tokens come from ``det`` (absolute-pixel xyxy); when
    ``det`` is ``None`` they resolve to ``"0"`` (top-left) / ``""``.

    Tokens:

    ====== ===============================================================
    token  value
    ====== ===============================================================
    track  ``track_id`` as-is (str)
    frame  ``frame_index`` as int when possible, else str
    index  ``order_index`` as int when possible, else str
    label  ``det.label`` or ``""``
    score  ``det.score`` formatted to 2dp, or ``""``
    x      ``round(det.x1)`` (left edge), or ``"0"``
    y      ``round(det.y1)`` (top edge), or ``"0"``
    w      ``round(det.width)``, or ``"0"``
    h      ``round(det.height)``, or ``"0"``
    ====== ===============================================================
    """
    def _as_int_str(v: Any) -> str:
        try:
            return str(int(v))
        except (TypeError, ValueError):
            return "" if v is None else str(v)

    if det is not None:
        try:
            x = str(int(round(float(det.x1))))
            y = str(int(round(float(det.y1))))
            w = str(int(round(float(det.width))))
            h = str(int(round(float(det.height))))
        except (TypeError, ValueError, AttributeError):
            x = y = w = h = "0"
        label = det.label if getattr(det, "label", None) else ""
        score = _fmt_score(getattr(det, "score", None))
    else:
        x = y = w = h = "0"
        label = ""
        score = ""

    return {
        "track": "" if track_id is None else str(track_id),
        "frame": _as_int_str(frame_index),
        "index": _as_int_str(order_index),
        "label": label,
        "score": score,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    }


def expand_template(template: Optional[str], context: dict) -> str:
    """Expand ``{token}`` placeholders in ``template`` using ``context``.

    This is the hardened substitution engine behind the ``template`` caption
    source. It **never raises**:

    * ``template`` is ``None`` / empty -> ``""``.
    * A known token in ``context`` is replaced by its value.
    * An *unknown* ``{token}`` is left literal (braces preserved) so the user
      can see their typo instead of getting a crash or a blank.
    * Unbalanced / stray braces (``"{"``, ``"}"``, ``"{a"``) are emitted
      verbatim rather than throwing ``ValueError`` the way ``str.format`` would.
    * ``{{`` / ``}}`` are treated as literal escaped braces (``{`` / ``}``),
      matching :meth:`str.format` semantics.

    The implementation uses :class:`string.Formatter` field parsing (which
    already understands ``{{``/``}}`` escaping) and resolves each field by hand
    so missing keys degrade to literal text. Format specs / conversions on a
    token (``{score:.0f}``) are ignored — the pre-formatted context value is
    used as-is — to keep behaviour predictable and exception-free.
    """
    if not template:
        return ""
    formatter = string.Formatter()
    out_parts: list[str] = []
    try:
        for literal_text, field_name, format_spec, conversion in formatter.parse(template):
            if literal_text:
                out_parts.append(literal_text)
            if field_name is None:
                # Pure literal chunk (or trailing text); nothing more to add.
                continue
            # Reconstruct the original ``{...}`` text in case we must keep it
            # literal (unknown token / unsupported nested field).
            original = "{" + field_name
            if conversion:
                original += "!" + conversion
            if format_spec:
                original += ":" + format_spec
            original += "}"

            # Only flat, simple field names are treated as tokens; anything
            # with indexing/attribute access (e.g. "a.b", "a[0]") is left
            # literal so we never reach into objects unexpectedly.
            key = field_name.strip()
            if key and key.isidentifier() and key in context:
                out_parts.append(str(context[key]))
            else:
                out_parts.append(original)
    except (ValueError, IndexError, KeyError):
        # Truly malformed brace soup that even Formatter.parse() chokes on:
        # fall back to returning the template verbatim. Never propagate.
        return template
    return "".join(out_parts)


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

def _frame_caption(captions: dict, frame_index: Any, source_index: Optional[int]) -> str:
    """Look up a by-frame caption (``captions['frames'][str(frame)]``).

    Returns ``""`` if ``frames`` is missing/not-a-dict, the frame key is absent,
    or the value cannot be coerced. A list value is indexed by ``source_index``
    (the detection's slot within that frame). Never raises.
    """
    frames = captions.get("frames")
    if not isinstance(frames, dict):
        return ""
    # Frame keys are JSON object keys -> strings. Accept both the canonical
    # str key and a stray int key for robustness.
    value = None
    if frame_index is not None:
        skey = str(frame_index)
        if skey in frames:
            value = frames[skey]
        elif frame_index in frames:  # tolerate non-string keys
            value = frames[frame_index]
    if value is None:
        return ""
    return _coerce_caption_value(value, source_index)


def _track_caption(captions: dict, track_id: Any, source_index: Optional[int]) -> str:
    """Look up a by-track caption (``captions[str(track_id)]``).

    Tries the canonical string key first, then a raw-typed key. Returns ``""``
    when absent / uncoercible. Never raises.
    """
    if track_id is None:
        return ""
    value = None
    skey = str(track_id)
    if skey in captions:
        value = captions[skey]
    elif track_id in captions:  # tolerate int keys in a programmatically built dict
        value = captions[track_id]
    if value is None:
        return ""
    return _coerce_caption_value(value, source_index)


def resolve_caption(
    *,
    source: str,
    det: Optional[Detection],
    track_id: Any,
    order_index: int,
    fixed_label: str,
    captions: dict,
    template: Optional[str],
    frame_index: int,
) -> str:
    """Resolve the raw caption string for one detection.

    The ``source`` selects one of :data:`yea_hud.types.CAPTION_SOURCES`:

    ================= =========================================================
    source            behaviour
    ================= =========================================================
    ``"fixed"``       returns ``fixed_label`` (or ``""`` if it is ``None``).
    ``"bbox_label"``  returns ``det.label`` or ``""`` (SAM3 emits no label, so
                      this is blank for SAM3 — expected, findings §3).
    ``"json_by_track"`` ``captions[str(track_id)]`` (list values indexed by the
                      detection's ``source_index``); ``""`` if missing.
    ``"json_by_frame"`` ``captions['frames'][str(frame_index)]``; if that value
                      is a list it is indexed by ``det.source_index``; ``""`` if
                      missing.
    ``"template"``    expands ``{track}{frame}{index}{label}{score}{x}{y}{w}{h}``
                      via :func:`expand_template`; unknown tokens stay literal
                      and nothing raises.
    ``"none"``        always returns ``""``.
    ================= =========================================================

    An unrecognised ``source`` is treated as ``"none"`` (returns ``""``) so a
    bad widget value can never crash a render.

    The returned string is **raw** — not uppercased and not truncated. The
    compositor applies casing / ellipsis when it draws the label strip.

    Parameters
    ----------
    source:
        One of :data:`CAPTION_SOURCES`.
    det:
        The :class:`Detection` being labelled. May be ``None`` for the
        ``fixed`` / ``none`` / ``template`` (geometry-less) paths; geometry and
        label tokens then resolve to their defaults.
    track_id:
        The unified track id (int or str). Used by ``json_by_track`` and the
        ``{track}`` template token.
    order_index:
        1-based display order within the frame. Used by the ``{index}`` token.
    fixed_label:
        The literal text for the ``fixed`` source.
    captions:
        The dict from :func:`parse_captions_json` (may be ``{}``).
    template:
        The template string for the ``template`` source (may be ``None``).
    frame_index:
        The current frame index. Used by ``json_by_frame`` and ``{frame}``.

    Returns
    -------
    str
        The resolved raw caption (possibly empty). Never ``None``.
    """
    # Normalise source defensively; unknown -> "none".
    src = source if isinstance(source, str) else ""
    if src not in CAPTION_SOURCES:
        src = "none"

    # The slot used to index list-valued JSON captions.
    source_index = getattr(det, "source_index", None) if det is not None else None

    if src == "none":
        return ""

    if src == "fixed":
        return "" if fixed_label is None else str(fixed_label)

    if src == "bbox_label":
        label = getattr(det, "label", None) if det is not None else None
        return label if label else ""

    if src == "json_by_track":
        if not isinstance(captions, dict):
            return ""
        return _track_caption(captions, track_id, source_index)

    if src == "json_by_frame":
        if not isinstance(captions, dict):
            return ""
        return _frame_caption(captions, frame_index, source_index)

    if src == "template":
        context = build_template_context(
            det=det,
            track_id=track_id,
            order_index=order_index,
            frame_index=frame_index,
        )
        return expand_template(template, context)

    # Unreachable (all CAPTION_SOURCES handled), but keep a safe default.
    return ""


# ---------------------------------------------------------------------------
# Order label
# ---------------------------------------------------------------------------

def format_order_label(order_index: int, text: str) -> str:
    """Prefix ``text`` with a zero-padded display order, e.g. ``"03. PERSON"``.

    The numeric prefix is always at least two digits (``1`` -> ``"01"``,
    ``10`` -> ``"10"``, ``100`` -> ``"100"``). The caller is responsible for
    uppercasing the final string (the contract returns the prefix in whatever
    case ``text`` already carries).

    Defensive behaviour (never raises):

    * A non-int ``order_index`` is coerced via ``int(...)``; if that fails it is
      stringified and used verbatim as the prefix.
    * ``text`` is ``None`` -> rendered as the empty string, so the result is
      just the numbered prefix plus ``". "`` trimmed to ``"03."``.
    * When ``text`` is empty the trailing ``" "`` is dropped so we return
      ``"03."`` rather than ``"03. "``.

    Parameters
    ----------
    order_index:
        1-based display order within the frame.
    text:
        The (already resolved) caption text.

    Returns
    -------
    str
        ``"<NN>. <text>"`` (or ``"<NN>."`` when ``text`` is empty).
    """
    try:
        n = int(order_index)
        prefix = f"{n:02d}"
    except (TypeError, ValueError):
        prefix = str(order_index)

    body = "" if text is None else str(text)
    if body == "":
        return f"{prefix}."
    return f"{prefix}. {body}"
