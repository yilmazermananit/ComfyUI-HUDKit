"""bbox_adapter.py — native ``BOUNDING_BOX`` -> per-frame detections.

Schema-agnostic. Pure deterministic compute only: no ComfyUI imports, no
network, no subprocess, no file I/O, no eval/exec.

This module is the boundary between ComfyUI's native ``BOUNDING_BOX`` value and
the renderer's internal :class:`~yonder_hud.types.Detection` model. It turns the
loose, multi-shaped native value into a clean per-frame
``list[list[Detection]]`` of absolute-pixel **xyxy** boxes, with all the
defensive normalization the brief (§3, §4.3) and ``docs/runtime_findings.md``
(§3) require.

Native ``BOUNDING_BOX`` ground truth (verified against
``comfy_extras/nodes_sam3.py`` ``SAM3_Detect.execute`` and
``comfy_api/latest/_io.py`` ``BoundingBox``):

* The atomic element is an **xywh dict** ``{"x","y","width","height"}`` where
  ``x,y`` is the *top-left* corner (NOT cxcywh, NOT xyxy). SAM3 also stores a
  ``"score"``; ``nodes_mediapipe`` additionally stores a ``"label"``.
* The runtime value comes in three real shapes (this is exactly what
  ``SAM3_Detect._boxes_to_tensor`` / its caller accept):

    1. a single ``dict``                 -> the same box for ALL frames
    2. a ``list[dict]``                  -> the same boxes for ALL frames
    3. a ``list[list[dict]]``            -> per-frame boxes (SAM3's own output)

* Coordinates are absolute pixels (SAM3 runs the model with ``orig_size=(H,W)``;
  its *input* path normalizes via ``d["x"]/W`` which proves the input is pixels).

Defensive extras this adapter accepts (the native nodes do not, but real graphs
mix tools — brief §4.3):

* ``xyxy`` dicts ``{"x1","y1","x2","y2"}`` (also ``"xmin"/"ymin"/"xmax"/"ymax"``).
* 4-element tuples/lists/ndarrays. xywh vs xyxy is auto-detected per box:
  if the 3rd value > 1st AND 4th value > 2nd (i.e. they look like a bottom-right
  corner that is to the lower-right of the top-left), it is treated as xyxy;
  otherwise as xywh.
* Normalized boxes (all four coords ``<= 1.0``) are scaled by ``W``/``H``.
  Detection is *per box* so a graph that mixes normalized and pixel boxes still
  resolves correctly.

Output guarantees (every emitted :class:`Detection`):

* ``xyxy`` absolute pixels, swapped/clamped to ``[0,W] x [0,H]``,
  ``x2 >= x1`` and ``y2 >= y1`` (via :mod:`yonder_hud.geometry`).
* degenerate boxes (width/height ``< 1px`` after clamping) are dropped.
* ``frame_index`` set to the output-frame slot; ``source_index`` set to the
  box's position within its source frame list (BEFORE degenerate filtering, so
  it stays aligned with the source — and with an ``individual_masks`` mask
  batch, findings §4).
* ``score`` / ``label`` carried through when present (``score`` coerced to a
  finite float, ``label`` to a str). ``source_track_id`` / ``mask_index`` stay
  ``None`` on this (non-tracking) path so the renderer's IoU tracker engages.

Broadcasting (brief §7 — never auto-repeat a sequence):

* single ``dict`` / ``list[dict]``  => identical detections for ALL ``B`` frames.
* ``list[list]`` shorter than ``B`` => the remaining frames get ``[]`` (empty),
  NEVER a repeat of the last frame.
* ``list[list]`` longer than ``B``  => the extra frames are ignored.

See ``docs/INTERFACES.md`` ("## bbox_adapter.py") for the binding signatures.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

from . import geometry
from .types import Detection

__all__ = ["adapt_bboxes", "summarize_bboxes"]


# ---------------------------------------------------------------------------
# Scalar / coordinate coercion helpers (private)
# ---------------------------------------------------------------------------

def _finite_float(v: Any) -> Optional[float]:
    """Coerce ``v`` to a finite ``float`` or return ``None``.

    Accepts python numbers, numeric strings, and 0-d numpy / torch scalars
    (anything implementing ``float()``). Bools are treated as numbers
    (``True``->1.0). Non-finite (NaN/Inf) and non-numeric values yield ``None``.
    """
    if v is None:
        return None
    # bool is an int subclass; float() handles it. Reject containers explicitly
    # so e.g. a 1-element list does not silently coerce.
    if isinstance(v, (list, tuple, dict, set)):
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _coerce_score(v: Any) -> Optional[float]:
    """Coerce a detection ``score`` to a finite float, else ``None``."""
    return _finite_float(v)


def _coerce_label(v: Any) -> Optional[str]:
    """Coerce a detection ``label`` to a non-empty ``str``, else ``None``.

    SAM3 never emits a label (findings §3); MediaPipe does. An empty / whitespace
    label is treated as absent so the caption layer's ``bbox_label`` source can
    fall back cleanly.
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v
    else:
        try:
            s = str(v)
        except Exception:  # noqa: BLE001 — never let a weird object raise here
            return None
    s = s.strip()
    return s if s else None


def _is_normalized_quad(a: float, b: float, c: float, d: float) -> bool:
    """True if all four coords look normalized (in ``[0, 1]``).

    Used to decide whether to scale a box by ``W``/``H``. We require *all four*
    values to be ``<= 1.0`` (and ``>= 0`` is not required — a box can sit at a
    slightly negative normalized edge). A box like ``(0,0,1,1)`` is treated as
    normalized full-frame, matching the brief's ``all<=1.0`` rule.
    """
    return (a <= 1.0) and (b <= 1.0) and (c <= 1.0) and (d <= 1.0)


# ---------------------------------------------------------------------------
# Single-box extraction
# ---------------------------------------------------------------------------

# Key alias groups for dict-shaped boxes. First present key in each group wins.
_X_KEYS = ("x", "x1", "xmin", "left", "X")
_Y_KEYS = ("y", "y1", "ymin", "top", "Y")
_W_KEYS = ("width", "w", "W")
_H_KEYS = ("height", "h", "H")
_X2_KEYS = ("x2", "xmax", "right")
_Y2_KEYS = ("y2", "ymax", "bottom")


def _first_key(d: dict, keys: Tuple[str, ...]) -> Optional[float]:
    """Return the finite float value of the first present key in ``keys``."""
    for k in keys:
        if k in d:
            f = _finite_float(d[k])
            if f is not None:
                return f
    return None


def _quad_to_xyxy_box(a: float, b: float, c: float, d: float, is_xyxy: bool,
                      W: int, H: int) -> Optional[geometry.Box]:
    """Turn four raw coords into a normalized/clamped absolute-pixel xyxy box.

    ``is_xyxy`` selects the interpretation (xyxy vs xywh). Normalized inputs
    (all four ``<= 1.0``) are scaled by ``W``/``H`` *before* xywh->xyxy
    conversion. Returns ``None`` if the box is degenerate (< 1px) after clamp.
    """
    if _is_normalized_quad(a, b, c, d):
        # Scale to pixels. For xyxy all four are coords; for xywh, (a,b) are the
        # corner and (c,d) are width/height — both scale the same way (x by W,
        # y by H, w by W, h by H).
        a *= W
        c *= W
        b *= H
        d *= H

    if is_xyxy:
        x1, y1, x2, y2 = a, b, c, d
    else:  # xywh
        x1, y1, x2, y2 = geometry.xywh_to_xyxy(a, b, c, d)

    box = geometry.clamp_box(geometry.normalize_box(x1, y1, x2, y2), int(W), int(H))
    if geometry.is_degenerate(box, min_size=1.0):
        return None
    return box


def _box_from_dict(d: dict, W: int, H: int) -> Tuple[Optional[geometry.Box], Optional[float], Optional[str]]:
    """Extract ``(xyxy_box | None, score, label)`` from a single box dict.

    Recognizes the native xywh form ``{"x","y","width","height"}`` first
    (findings §3), then defensively an xyxy form ``{"x1","y1","x2","y2"}`` /
    ``{"xmin","ymin","xmax","ymax"}``. ``score`` and ``label`` are read from any
    of the usual keys when present. A box that cannot be parsed yields
    ``(None, score, label)`` so the caller can still account for the slot.
    """
    score = _coerce_score(d.get("score", d.get("confidence", d.get("conf"))))
    label = _coerce_label(d.get("label", d.get("class", d.get("name"))))

    x = _first_key(d, _X_KEYS)
    y = _first_key(d, _Y_KEYS)

    # Prefer explicit width/height (native xywh) over xmax/ymax (xyxy).
    w = _first_key(d, _W_KEYS)
    h = _first_key(d, _H_KEYS)
    x2 = _first_key(d, _X2_KEYS)
    y2 = _first_key(d, _Y2_KEYS)

    if x is None or y is None:
        return None, score, label

    if w is not None and h is not None:
        box = _quad_to_xyxy_box(x, y, w, h, is_xyxy=False, W=W, H=H)
    elif x2 is not None and y2 is not None:
        box = _quad_to_xyxy_box(x, y, x2, y2, is_xyxy=True, W=W, H=H)
    else:
        # Only a corner, no extent -> nothing to draw.
        return None, score, label

    return box, score, label


def _looks_like_xyxy(third: float, fourth: float, x: float, y: float) -> bool:
    """Heuristic for an ambiguous 4-tuple: is it xyxy (vs xywh)?

    Implements the brief's (§4.3) literal disambiguation rule for a bare
    4-element sequence (a dict is never ambiguous — its keys name the format):

        "if 3rd/4th look like absolute >x and >y treat as xyxy else xywh"

    i.e. the sequence is read as ``xyxy`` when the 3rd value lies to the right
    of the 1st AND the 4th lies below the 2nd (``x2 > x1`` and ``y2 > y1``) —
    the geometric signature of a valid bottom-right corner. Otherwise (3rd/4th
    not strictly past the top-left — including zero/negative which can only be a
    width/height pair) it is read as ``xywh``.

    Note this is genuinely heuristic: a tuple like ``(10, 20, 30, 40)`` is
    treated as ``xyxy`` under this rule. Callers that KNOW the format should use
    the dict form (``{"x","y","width","height"}``) which is unambiguous and is
    the native SAM3 shape. The two interpretations only diverge in geometry and
    both are normalized + clamped downstream; a wrong guess on a near-square box
    is small, and a degenerate result is dropped.
    """
    return (third > x) and (fourth > y)


def _box_from_sequence(seq: Any, W: int, H: int) -> Tuple[Optional[geometry.Box], Optional[float], Optional[str]]:
    """Extract ``(xyxy_box | None, None, None)`` from a 4-element sequence.

    Accepts lists / tuples / 1-D numpy arrays / 1-D torch tensors of length 4.
    The xywh-vs-xyxy interpretation is auto-detected per box (see
    :func:`_looks_like_xyxy`). Sequences carry no score/label.
    """
    vals = _sequence_to_floats(seq)
    if vals is None or len(vals) != 4:
        return None, None, None
    a, b, c, d = vals
    is_xyxy = _looks_like_xyxy(c, d, a, b)
    box = _quad_to_xyxy_box(a, b, c, d, is_xyxy=is_xyxy, W=W, H=H)
    return box, None, None


def _sequence_to_floats(seq: Any) -> Optional[List[float]]:
    """Convert a 1-D list/tuple/ndarray/tensor to a list of finite floats.

    Returns ``None`` for anything that is not a flat numeric sequence of finite
    values. Used only for the defensive 4-tuple path; dicts go through
    :func:`_box_from_dict`.
    """
    # torch / numpy: use ``.tolist()`` when available and 1-D.
    tolist = getattr(seq, "tolist", None)
    if tolist is not None and not isinstance(seq, (list, tuple)):
        try:
            shape = getattr(seq, "shape", None)
            if shape is not None and len(shape) != 1:
                return None
            seq = tolist()
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(seq, (list, tuple)):
        return None
    out: List[float] = []
    for v in seq:
        f = _finite_float(v)
        if f is None:
            return None
        out.append(f)
    return out


def _extract_one(item: Any, W: int, H: int) -> Tuple[Optional[geometry.Box], Optional[float], Optional[str]]:
    """Dispatch a single box-like item to the dict or sequence extractor."""
    if isinstance(item, dict):
        return _box_from_dict(item, W, H)
    if isinstance(item, (list, tuple)) or hasattr(item, "tolist"):
        return _box_from_sequence(item, W, H)
    return None, None, None


# ---------------------------------------------------------------------------
# Frame-list extraction
# ---------------------------------------------------------------------------

def _is_box_dict(obj: Any) -> bool:
    """True if ``obj`` is a dict that carries at least an x/y corner key."""
    if not isinstance(obj, dict):
        return False
    has_x = any(k in obj for k in _X_KEYS)
    has_y = any(k in obj for k in _Y_KEYS)
    return has_x and has_y


def _is_box_sequence(obj: Any) -> bool:
    """True if ``obj`` looks like a single 4-number box sequence (not a list of boxes)."""
    vals = _sequence_to_floats(obj)
    return vals is not None and len(vals) == 4


def _frame_detections(frame_list: Any, frame_index: int, W: int, H: int) -> List[Detection]:
    """Build the ``Detection`` list for one frame from its list of box items.

    ``source_index`` is the box's position in ``frame_list`` (counting from 0,
    BEFORE dropping degenerate boxes) so it stays aligned with the source frame
    and with an ``individual_masks`` mask batch (findings §4). Degenerate /
    unparseable boxes are skipped (they consume a ``source_index`` but emit no
    Detection).
    """
    if frame_list is None:
        return []
    # A single dict / single sequence used where a list-of-boxes is expected:
    # treat it as a one-element frame (defensive; some tools wrap loosely).
    if _is_box_dict(frame_list) or _is_box_sequence(frame_list):
        frame_list = [frame_list]
    if not isinstance(frame_list, (list, tuple)):
        return []

    dets: List[Detection] = []
    for src_idx, item in enumerate(frame_list):
        box, score, label = _extract_one(item, W, H)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        dets.append(Detection(
            frame_index=frame_index,
            source_index=src_idx,
            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
            score=score,
            label=label,
            source_track_id=None,
            mask_index=None,
        ))
    return dets


# ---------------------------------------------------------------------------
# Shape classification of the whole BOUNDING_BOX value
# ---------------------------------------------------------------------------

def _classify(bboxes: Any) -> str:
    """Classify the top-level native value into a broadcasting category.

    Returns one of:

    * ``'none'``      — ``None`` / empty -> no detections anywhere.
    * ``'dict'``      — a single box dict -> broadcast to all frames.
    * ``'seq4'``      — a single 4-number sequence -> broadcast to all frames.
    * ``'list_dict'`` — a ``list`` whose elements are box dicts/sequences ->
                        one frame's worth of boxes, broadcast to all frames.
    * ``'list_list'`` — a ``list`` whose elements are themselves lists (per-frame).
    * ``'empty_list'``— an empty list -> no detections anywhere.
    * ``'unknown'``   — anything unrecognized -> no detections (defensive).
    """
    if bboxes is None:
        return "none"
    if isinstance(bboxes, dict):
        return "dict"
    if _is_box_sequence(bboxes):
        # A bare 4-number sequence is a single box (NOT a per-frame list of 4 ints).
        return "seq4"
    if not isinstance(bboxes, (list, tuple)):
        return "unknown"
    if len(bboxes) == 0:
        return "empty_list"

    # Decide list[dict] vs list[list]. SAM3 uses ``isinstance(bboxes[0], list)``
    # to detect the per-frame form; we generalize: if ANY element is a frame
    # container (a list/tuple that is NOT itself a 4-number box) treat the whole
    # value as per-frame; otherwise it is a flat list of boxes.
    any_frame_container = False
    for el in bboxes:
        if isinstance(el, (list, tuple)) and not _is_box_sequence(el):
            any_frame_container = True
            break
        if el is None:
            # ``None`` only appears as a per-frame placeholder (an empty frame),
            # never as a single box -> per-frame form.
            any_frame_container = True
            break
    return "list_list" if any_frame_container else "list_dict"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adapt_bboxes(bboxes: Any, B: int, W: int, H: int) -> List[List[Detection]]:
    """Normalize a native ``BOUNDING_BOX`` value into per-frame detections.

    Parameters
    ----------
    bboxes :
        The native ``BOUNDING_BOX`` runtime value. One of (findings §3):
        a single xywh ``dict``; a ``list[dict]``; a ``list[list[dict]]``. Also
        defensively accepts xyxy dicts, 4-element tuples/lists/arrays, and
        normalized (``<=1.0``) coordinates (brief §4.3).
    B : int
        Number of output frames (the IMAGE batch size). The returned list always
        has length ``max(B, 0)``.
    W, H : int
        Frame width / height in pixels, used for clamping and for scaling
        normalized boxes.

    Returns
    -------
    list[list[Detection]]
        Exactly ``B`` frames. Each inner list holds the frame's detections in
        absolute-pixel **xyxy**, normalized + clamped, degenerate boxes dropped,
        ``frame_index`` / ``source_index`` / ``score`` / ``label`` populated.
        ``source_track_id`` and ``mask_index`` are ``None`` (this path feeds the
        IoU tracker).

    Broadcasting (brief §7)
    -----------------------
    * single ``dict`` / ``list[dict]`` -> identical detections for ALL frames.
    * ``list[list]`` shorter than ``B`` -> remaining frames get ``[]`` (never a
      repeat of the last frame).
    * ``list[list]`` longer than ``B``  -> extra frames ignored.

    Defensive behavior
    ------------------
    ``None`` / empty / unrecognized input yields ``B`` empty frames. ``W``/``H``
    ``<= 0`` collapse every box to degenerate -> ``B`` empty frames. The input is
    never mutated.
    """
    # --- sanitize batch / dimensions -------------------------------------
    try:
        B = int(B)
    except (TypeError, ValueError):
        B = 0
    if B < 0:
        B = 0

    Wi = _safe_dim(W)
    Hi = _safe_dim(H)

    empty_frames: List[List[Detection]] = [[] for _ in range(B)]

    if B == 0:
        return []
    if Wi <= 0 or Hi <= 0:
        # Nothing can be placed in a zero-area frame.
        return empty_frames

    kind = _classify(bboxes)

    if kind in ("none", "empty_list", "unknown"):
        return empty_frames

    if kind in ("dict", "seq4"):
        # Single box -> broadcast identical detections to every frame.
        # Build per frame (distinct Detection objects, correct frame_index).
        return [_frame_detections([bboxes], f, Wi, Hi) for f in range(B)]

    if kind == "list_dict":
        # One frame's worth of boxes -> broadcast to all frames.
        return [_frame_detections(bboxes, f, Wi, Hi) for f in range(B)]

    # kind == 'list_list' -> per-frame. Map source frame -> output frame 1:1,
    # truncate if longer than B, pad with EMPTY frames if shorter (NEVER repeat).
    out: List[List[Detection]] = []
    src_frames = list(bboxes)
    for f in range(B):
        if f < len(src_frames):
            out.append(_frame_detections(src_frames[f], f, Wi, Hi))
        else:
            out.append([])
    return out


def _safe_dim(v: Any) -> int:
    """Coerce a width/height to a non-negative int (``0`` on failure)."""
    f = _finite_float(v)
    if f is None:
        return 0
    iv = int(f)
    return iv if iv > 0 else 0


def summarize_bboxes(bboxes: Any, max_frames: int = 3, max_dets: int = 10) -> dict:
    """Produce a small, safe structural summary of a native ``BOUNDING_BOX``.

    For the inspector node and error messages: describes the value's *shape*
    without dumping giant tensors or every coordinate. Never raises on weird
    input.

    Parameters
    ----------
    bboxes :
        Any native ``BOUNDING_BOX`` value (or garbage).
    max_frames : int
        Maximum number of per-frame entries to detail (in the ``list_list``
        case). Extra frames are counted but not detailed.
    max_dets : int
        Maximum number of boxes to scan per frame when sampling keys / coord
        ranges (bounds the work on huge inputs).

    Returns
    -------
    dict
        Keys:

        * ``kind``        — one of the :func:`_classify` categories.
        * ``broadcast``   — ``'all_frames'`` | ``'per_frame'`` | ``'none'``.
        * ``n_frames``    — number of source frames (``list_list``) or ``None``.
        * ``total_boxes`` — total boxes counted across (sampled) frames.
        * ``per_frame_counts`` — list of per-frame box counts (truncated to
          ``max_frames``; ``'+N more'`` style overflow noted in ``frames_omitted``).
        * ``frames_omitted`` — number of frames not detailed.
        * ``keys``        — sorted union of keys seen on sampled box dicts.
        * ``coord_space`` — ``'pixel'`` | ``'normalized'`` | ``'mixed'`` |
          ``'unknown'`` (guessed from sampled coords).
        * ``has_score`` / ``has_label`` — booleans from sampled dicts.
        * ``element_type`` — python type name of the top-level value.
        * ``note``        — human-readable one-liner.
    """
    try:
        mf = int(max_frames)
    except (TypeError, ValueError):
        mf = 3
    if mf < 0:
        mf = 0
    try:
        md = int(max_dets)
    except (TypeError, ValueError):
        md = 10
    if md < 0:
        md = 0

    kind = _classify(bboxes)
    summary: dict = {
        "kind": kind,
        "broadcast": _broadcast_for(kind),
        "n_frames": None,
        "total_boxes": 0,
        "per_frame_counts": [],
        "frames_omitted": 0,
        "keys": [],
        "coord_space": "unknown",
        "has_score": False,
        "has_label": False,
        "element_type": type(bboxes).__name__,
        "note": "",
    }

    # Normalize into a list-of-frames view for uniform scanning.
    if kind == "none":
        summary["note"] = "None — no bounding boxes"
        return summary
    if kind == "empty_list":
        summary["n_frames"] = 0
        summary["note"] = "empty list — no bounding boxes"
        return summary
    if kind == "unknown":
        summary["note"] = (
            "unrecognized BOUNDING_BOX value (expected dict / list[dict] / "
            "list[list[dict]])"
        )
        return summary

    if kind in ("dict", "seq4"):
        frames_view: List[Any] = [[bboxes]]
        summary["n_frames"] = None  # broadcasts; not a per-frame count
    elif kind == "list_dict":
        frames_view = [list(bboxes)]
        summary["n_frames"] = None
    else:  # list_list
        frames_view = [f if isinstance(f, (list, tuple)) else ([] if f is None else [f])
                       for f in bboxes]
        summary["n_frames"] = len(frames_view)

    keys: set = set()
    saw_pixel = False
    saw_norm = False
    has_score = False
    has_label = False
    per_frame_counts: List[int] = []
    total = 0

    for fi, frame in enumerate(frames_view):
        if isinstance(frame, (list, tuple)):
            count = len(frame)
        else:
            count = 1
            frame = [frame]
        total += count
        if fi < mf:
            per_frame_counts.append(count)
        # Sample up to ``md`` boxes for key / coord-space inference.
        for item in list(frame)[:md]:
            if isinstance(item, dict):
                for k in item.keys():
                    keys.add(str(k))
                if any(k in item for k in ("score", "confidence", "conf")):
                    has_score = True
                if any(k in item for k in ("label", "class", "name")):
                    has_label = True
                pix, nrm = _coord_space_of_dict(item)
            elif isinstance(item, (list, tuple)) or hasattr(item, "tolist"):
                pix, nrm = _coord_space_of_sequence(item)
            else:
                pix = nrm = False
            saw_pixel = saw_pixel or pix
            saw_norm = saw_norm or nrm

    summary["total_boxes"] = total
    summary["per_frame_counts"] = per_frame_counts
    summary["frames_omitted"] = max(0, len(frames_view) - mf) if kind == "list_list" else 0
    summary["keys"] = sorted(keys)
    summary["has_score"] = has_score
    summary["has_label"] = has_label
    summary["coord_space"] = _coord_space_label(saw_pixel, saw_norm)
    summary["note"] = _summary_note(summary)
    return summary


# ---------------------------------------------------------------------------
# Summary helpers (private)
# ---------------------------------------------------------------------------

def _broadcast_for(kind: str) -> str:
    """Map a classification to a broadcast description."""
    if kind in ("dict", "seq4", "list_dict"):
        return "all_frames"
    if kind == "list_list":
        return "per_frame"
    return "none"


def _coord_space_of_dict(d: dict) -> Tuple[bool, bool]:
    """Return ``(looks_pixel, looks_normalized)`` for a single box dict.

    A box is "normalized" if every coordinate it exposes is ``<= 1.0``; "pixel"
    if any coordinate exceeds ``1.0``. Width/height are included (a normalized
    box's width is also ``<= 1``).
    """
    vals: List[float] = []
    for grp in (_X_KEYS, _Y_KEYS, _W_KEYS, _H_KEYS, _X2_KEYS, _Y2_KEYS):
        f = _first_key(d, grp)
        if f is not None:
            vals.append(f)
    return _coord_space_of_values(vals)


def _coord_space_of_sequence(seq: Any) -> Tuple[bool, bool]:
    """Return ``(looks_pixel, looks_normalized)`` for a 4-number sequence."""
    vals = _sequence_to_floats(seq)
    if vals is None:
        return False, False
    return _coord_space_of_values(vals)


def _coord_space_of_values(vals: List[float]) -> Tuple[bool, bool]:
    """Infer pixel/normalized from a list of coordinate magnitudes."""
    if not vals:
        return False, False
    if any(abs(v) > 1.0 for v in vals):
        return True, False
    return False, True


def _coord_space_label(saw_pixel: bool, saw_norm: bool) -> str:
    """Combine per-box pixel/normalized observations into one label."""
    if saw_pixel and saw_norm:
        return "mixed"
    if saw_pixel:
        return "pixel"
    if saw_norm:
        return "normalized"
    return "unknown"


def _summary_note(s: dict) -> str:
    """Build a short human-readable note from the assembled summary."""
    kind = s["kind"]
    if kind in ("dict", "seq4"):
        return f"single box -> broadcast to all frames ({s['total_boxes']} box)"
    if kind == "list_dict":
        return f"flat list of {s['total_boxes']} box(es) -> broadcast to all frames"
    if kind == "list_list":
        nf = s["n_frames"]
        return (f"per-frame: {nf} frame(s), {s['total_boxes']} box(es) total, "
                f"coords look {s['coord_space']}")
    return s.get("note", "")
