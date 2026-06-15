"""track_adapter.py — SAM3_TRACK_DATA -> detections (PRIMARY tracking path).

This is the renderer's *primary* tracking input. Priority order (per
``docs/runtime_findings.md`` §3.5): (1) SAM3_TRACK_DATA -> (2) BOUNDING_BOX+MASK
-> (3) custom IoU tracker (fallback). When this adapter runs, the per-detection
``source_track_id`` is filled from the native persistent object index, so the
renderer's own IoU tracker is bypassed entirely — identity comes for free from
SAM3 ("track identity survives short occlusions").

Verified native structure (``comfy/ldm/sam3/tracker.py`` return @ 1791/1802 +
``comfy_extras/nodes_sam3.py`` @ 311 which appends ``orig_size``)::

    SAM3_TRACK_DATA = {                       # plain python dict
      "packed_masks": Tensor | None,          # [N_frames, N_obj, H_m, W_m//8] uint8 (bit-packed)
      "n_frames":     int,                    # N
      "scores":       list[float],            # length N_obj ; [] if no tracks
      "orig_size":    (H, W),                 # original frame size in pixels
    }

Semantics (from source, do NOT guess):

* ``packed_masks`` dim 1 (``N_obj``) is the **persistent object/track identity**.
  Object ``i`` is the same tracked subject across every frame; ``scores[i]`` is
  its score. Per-frame object counts are padded video-wide to ``max_obj``
  (tracker.py:1793-1801): a frame where an object is absent gets an all-zero
  (empty) mask -> occlusion on that frame, but the identity index stays fixed.
* Bit-pack format (tracker.py:194-204): ``pack_masks`` turns ``[*,H,W] bool``
  into ``[*,H,W//8] uint8`` (W divisible by 8); ``unpack_masks`` is the inverse
  ``(packed.unsqueeze(-1) & bits).bool().view(...)`` with
  ``bits=[1,2,4,8,16,32,64,128]``. Reimplemented here (Cloud / version-drift
  safety; this module never imports ``comfy``).
* ``packed_masks is None`` -> no tracks at all:
  ``{"packed_masks": None, "n_frames": N, "scores": []}``.
* Mask resolution ``H_m, W_m`` is SAM3's internal tracker resolution; the bbox
  extent is computed there and then scaled to ``orig_size (H, W)`` by
  ``(H/H_m, W/W_m)``.

Allowed deps only: stdlib, numpy, torch, ``from .types import ...`` and sibling
``geometry``. NO ComfyUI imports. Pure deterministic compute (no network, no
subprocess, no eval, no file I/O).

See ``docs/INTERFACES.md`` ("## track_adapter.py") for the binding signatures.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from . import geometry
from .types import Detection

__all__ = [
    "unpack_packed_masks",
    "is_track_data",
    "adapt_track_data",
    "summarize_track_data",
]


# ---------------------------------------------------------------------------
# Bit-unpack — exact reimplementation of comfy/ldm/sam3/tracker.py:201
# ---------------------------------------------------------------------------

def unpack_packed_masks(packed):
    """Bit-unpack ``[*, H, W//8] uint8`` -> bool ``[*, H, W]``.

    Exact reimplementation of ``comfy.ldm.sam3.tracker.unpack_masks`` (kept
    in-house so this module has no hard dependency on ``comfy.ldm.sam3`` and is
    immune to ComfyUI version drift / Comfy Cloud sandboxing):

        bits = [1, 2, 4, 8, 16, 32, 64, 128]                       # uint8
        out  = (packed.unsqueeze(-1) & bits).bool().view(*packed.shape[:-1], -1)

    The last dim expands by a factor of 8, so an input whose last dim is
    ``W//8`` yields an output last dim of ``W`` (LSB-first per byte, matching
    ``pack_masks``' ``1 << arange(8)`` packing).

    Parameters
    ----------
    packed : torch.Tensor | array-like
        Bit-packed mask(s). Any shape ``[*, K]`` (``K = W//8``) is accepted;
        the leading dims are preserved. A numpy array (or anything
        ``torch.as_tensor`` accepts) is coerced to a tensor. The dtype is cast
        to ``uint8`` before the bitwise-AND so the masking is well defined
        regardless of the incoming integer/bool dtype.

    Returns
    -------
    torch.Tensor
        A ``bool`` tensor of shape ``[*, K*8]`` on the same device as the
        (coerced) input.

    Raises
    ------
    ValueError
        If ``packed`` is ``None`` or has zero dimensions (a scalar cannot carry
        a packed row).
    """
    if packed is None:
        raise ValueError("unpack_packed_masks: 'packed' is None")

    if not isinstance(packed, torch.Tensor):
        # numpy arrays, python lists, etc. -> tensor (no copy when possible).
        packed = torch.as_tensor(packed)

    if packed.dim() == 0:
        raise ValueError(
            "unpack_packed_masks: expected at least a 1-D packed tensor, "
            f"got a 0-d scalar of dtype {packed.dtype}"
        )

    # Bitwise-AND requires an integer dtype; uint8 matches the native packer and
    # makes the 8 bit-planes [1,2,4,8,16,32,64,128] exact for any input dtype
    # (bool/int included).
    packed_u8 = packed.to(torch.uint8)
    bits = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed_u8.device
    )
    return (packed_u8.unsqueeze(-1) & bits).bool().view(*packed_u8.shape[:-1], -1)


# ---------------------------------------------------------------------------
# Type guard
# ---------------------------------------------------------------------------

def is_track_data(obj) -> bool:
    """Return True iff ``obj`` looks like a SAM3_TRACK_DATA dict.

    The minimal, source-verified signature is a mapping that carries BOTH a
    ``'packed_masks'`` key and an ``'n_frames'`` key (``scores`` / ``orig_size``
    may be absent on degenerate producers, so they are not required). This is
    the discriminator the node layer uses to choose the primary tracking path
    over the BOUNDING_BOX path.
    """
    return (
        isinstance(obj, dict)
        and "packed_masks" in obj
        and "n_frames" in obj
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_n_frames(value, fallback: int) -> int:
    """Coerce a reported ``n_frames`` to a non-negative int, else ``fallback``."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return max(0, int(fallback))
    return n if n >= 0 else max(0, int(fallback))


def _orig_size(track_data: dict, h_m: int, w_m: int) -> Tuple[int, int]:
    """Resolve ``orig_size`` -> ``(H, W)`` ints, falling back to mask res.

    ``SAM3_VideoTrack`` always appends ``orig_size=(H, W)`` (nodes_sam3.py:311),
    but a hand-built / older dict may omit it; in that case we render at the
    mask resolution (1:1 scale) so the adapter still produces sane boxes instead
    of dividing by an unknown size.
    """
    osz = track_data.get("orig_size")
    if isinstance(osz, (tuple, list)) and len(osz) >= 2:
        try:
            H = int(osz[0])
            W = int(osz[1])
        except (TypeError, ValueError):
            H, W = int(h_m), int(w_m)
    else:
        H, W = int(h_m), int(w_m)
    # Guard against zero / negative target sizes (would invert scaling).
    if H <= 0:
        H = int(h_m)
    if W <= 0:
        W = int(w_m)
    return H, W


def _mask_extent_xyxy(mask_bool: torch.Tensor) -> Optional[Tuple[float, float, float, float]]:
    """Tight extent of a 2-D bool mask as ``xyxy`` in mask pixels, or None.

    ``x1, y1`` is the top-left of the smallest enclosing box; ``x2, y2`` is one
    past the bottom-right occupied pixel (so a single lit pixel at ``(r, c)``
    yields ``(c, r, c+1, r+1)`` — a unit box, never zero-area). Returns ``None``
    when the mask has no lit pixel (the caller treats that as occlusion).
    """
    # ``mask_bool`` is [H_m, W_m] bool. Find occupied rows/cols.
    if mask_bool.dim() != 2:
        mask_bool = mask_bool.reshape(mask_bool.shape[-2], mask_bool.shape[-1])
    rows = torch.any(mask_bool, dim=1)
    cols = torch.any(mask_bool, dim=0)
    if not bool(torch.any(rows)):
        return None
    row_idx = torch.nonzero(rows, as_tuple=False).flatten()
    col_idx = torch.nonzero(cols, as_tuple=False).flatten()
    y1 = float(row_idx[0].item())
    y2 = float(row_idx[-1].item()) + 1.0
    x1 = float(col_idx[0].item())
    x2 = float(col_idx[-1].item()) + 1.0
    return (x1, y1, x2, y2)


def _score_at(scores, i: int) -> Optional[float]:
    """Return ``float(scores[i])`` if available and finite, else ``None``."""
    if scores is None:
        return None
    try:
        if i < 0 or i >= len(scores):
            return None
        s = float(scores[i])
    except (TypeError, ValueError, IndexError):
        return None
    # NaN / inf scores would poison downstream sorting; drop to None.
    if s != s or s in (float("inf"), float("-inf")):
        return None
    return s


# ---------------------------------------------------------------------------
# Primary adapter
# ---------------------------------------------------------------------------

def adapt_track_data(
    track_data: dict, *, area_threshold: int = 1
) -> Tuple[List[List[Detection]], int]:
    """Convert SAM3_TRACK_DATA into per-frame detections (PRIMARY path).

    For every frame ``t`` and persistent object ``i`` (``i`` IS the track
    identity): unpack ``packed_masks[t, i]`` to a bool mask; if its lit-pixel
    area ``>= area_threshold`` emit::

        Detection(frame_index=t, source_index=i,
                  x1, y1, x2, y2 = mask_extent_scaled_to_orig,
                  score=scores[i] if i < len(scores) else None,
                  source_track_id=i, mask_index=i)

    An empty (all-zero) mask is **skipped** for that frame (the object is
    occluded / absent there) but the identity ``i`` persists across frames —
    nothing is renumbered. The mask extent is computed at the tracker
    resolution ``(H_m, W_m)`` and scaled to ``orig_size (H, W)`` by
    ``(H/H_m, W/W_m)``, then normalized / clamped / degenerate-dropped via
    :mod:`yea_hud.geometry`.

    Parameters
    ----------
    track_data : dict
        A SAM3_TRACK_DATA dict (see module docstring). Defensive against missing
        keys and ``packed_masks is None``.
    area_threshold : int, keyword-only
        Minimum lit-pixel count (in *mask* resolution) for an object to be
        considered visible on a frame. ``1`` (default) means "any lit pixel".
        Coerced to ``>= 1``.

    Returns
    -------
    (per_frame, n_obj)
        ``per_frame`` is a list of length ``n_frames``; each entry is the list
        of :class:`Detection` for that frame (possibly empty). ``n_obj`` is the
        padded object/track count (``packed_masks.shape[1]``), i.e. the number
        of distinct persistent identities. ``packed_masks is None`` ->
        ``([[], ...] * n_frames, 0)``.

    Notes
    -----
    The function never mutates ``track_data`` and never raises on a
    well-formed-but-empty input; a malformed ``packed_masks`` (not a tensor /
    wrong rank) is treated as "no tracks" and yields empty frames rather than
    throwing, so the renderer degrades gracefully to the fallback path.
    """
    if not isinstance(track_data, dict):
        # Not a track-data dict at all: nothing to adapt.
        return [], 0

    area_min = max(1, int(area_threshold)) if _is_intish(area_threshold) else 1

    packed = track_data.get("packed_masks", None)
    scores = track_data.get("scores", None)

    # --- No tracks at all ------------------------------------------------
    if packed is None:
        n_frames = _coerce_n_frames(track_data.get("n_frames"), 0)
        return [[] for _ in range(n_frames)], 0

    # Coerce to a tensor and validate rank. We need [N_frames, N_obj, H_m, W_m//8].
    if not isinstance(packed, torch.Tensor):
        try:
            packed = torch.as_tensor(packed)
        except (TypeError, ValueError, RuntimeError):
            n_frames = _coerce_n_frames(track_data.get("n_frames"), 0)
            return [[] for _ in range(n_frames)], 0

    if packed.dim() != 4:
        # Unexpected shape -> degrade to "no usable tracks" but still report the
        # frame count so the renderer can pad an empty plan per frame.
        n_frames = _coerce_n_frames(track_data.get("n_frames"), 0)
        return [[] for _ in range(n_frames)], 0

    n_frames_packed, n_obj, h_m, w_packed = (int(s) for s in packed.shape)
    w_m = w_packed * 8

    # Reported n_frames wins for the OUTPUT length (the renderer pads its plan to
    # it), but we only have packed data for n_frames_packed frames.
    n_frames = _coerce_n_frames(track_data.get("n_frames"), n_frames_packed)

    H, W = _orig_size(track_data, h_m, w_m)
    # Scale factors from mask resolution to original pixels.
    sx = (W / w_m) if w_m > 0 else 1.0
    sy = (H / h_m) if h_m > 0 else 1.0

    # Work on CPU so .item()/nonzero are cheap and device-independent; bit-unpack
    # the whole batch once (vectorized, identical math to native unpack_masks).
    packed_cpu = packed.detach().to("cpu")
    # masks_bool: [N_frames_packed, N_obj, H_m, W_m] bool
    masks_bool = unpack_packed_masks(packed_cpu)

    per_frame: List[List[Detection]] = []
    for t in range(n_frames):
        frame_dets: List[Detection] = []
        if t < n_frames_packed:
            frame_masks = masks_bool[t]  # [N_obj, H_m, W_m]
            for i in range(n_obj):
                mask_i = frame_masks[i]  # [H_m, W_m] bool
                # Lit-pixel area in mask resolution.
                area = int(mask_i.sum().item())
                if area < area_min:
                    # Occluded / absent on this frame; identity i persists.
                    continue
                extent = _mask_extent_xyxy(mask_i)
                if extent is None:
                    continue
                ex1, ey1, ex2, ey2 = extent
                # Scale mask-resolution extent -> original pixels.
                box = (ex1 * sx, ey1 * sy, ex2 * sx, ey2 * sy)
                box = geometry.clamp_box(geometry.normalize_box(*box), W, H)
                if geometry.is_degenerate(box, min_size=1.0):
                    # Sub-pixel box after scaling/clamping -> not drawable.
                    continue
                x1, y1, x2, y2 = box
                frame_dets.append(
                    Detection(
                        frame_index=t,
                        source_index=i,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=_score_at(scores, i),
                        label=None,
                        source_track_id=i,
                        mask_index=i,
                    )
                )
        per_frame.append(frame_dets)

    return per_frame, n_obj


def _is_intish(v) -> bool:
    """True if ``v`` can be losslessly read as an int (bool excluded as int-ish)."""
    if isinstance(v, bool):
        return True  # bool is a valid int (True->1) for the threshold use
    try:
        int(v)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Inspector summary
# ---------------------------------------------------------------------------

def summarize_track_data(track_data) -> dict:
    """Build a small, safe structural summary of a SAM3_TRACK_DATA dict.

    For the inspector node and error messages — NO giant tensor dumps. Reports
    shapes, frame/object counts, the score list (rounded, capped), the resolved
    original size, and a per-frame visibility count (how many objects are
    present, i.e. non-empty mask, on each frame, capped to a preview length).

    Robust to anything: a non-dict, a missing/None ``packed_masks``, or a
    malformed tensor all produce a descriptive dict rather than raising.

    Returns
    -------
    dict
        Keys: ``is_track_data`` (bool), ``n_frames`` (int|None), ``n_obj`` (int),
        ``packed_masks_shape`` (tuple|None), ``packed_masks_dtype`` (str|None),
        ``mask_resolution`` ((H_m, W_m)|None), ``orig_size`` ((H, W)|None),
        ``scores`` (list, capped), ``n_scores`` (int), ``visibility_per_frame``
        (list[int], capped, count of present objects per frame),
        ``note`` (str, present only when something is degenerate/unexpected).
    """
    out: dict = {
        "is_track_data": is_track_data(track_data),
        "n_frames": None,
        "n_obj": 0,
        "packed_masks_shape": None,
        "packed_masks_dtype": None,
        "mask_resolution": None,
        "orig_size": None,
        "scores": [],
        "n_scores": 0,
        "visibility_per_frame": [],
    }

    if not isinstance(track_data, dict):
        out["note"] = f"not a dict: {type(track_data).__name__}"
        return out

    # n_frames (reported)
    if "n_frames" in track_data:
        try:
            out["n_frames"] = int(track_data["n_frames"])
        except (TypeError, ValueError):
            out["n_frames"] = None

    # scores (cap to a sane preview, round for readability)
    scores = track_data.get("scores", None)
    if scores is not None:
        try:
            seq = list(scores)
        except TypeError:
            seq = []
        out["n_scores"] = len(seq)
        capped = seq[:64]
        rounded = []
        for s in capped:
            try:
                rounded.append(round(float(s), 4))
            except (TypeError, ValueError):
                rounded.append(None)
        out["scores"] = rounded

    # orig_size (reported)
    osz = track_data.get("orig_size")
    if isinstance(osz, (tuple, list)) and len(osz) >= 2:
        try:
            out["orig_size"] = (int(osz[0]), int(osz[1]))
        except (TypeError, ValueError):
            out["orig_size"] = None

    packed = track_data.get("packed_masks", None)
    if packed is None:
        out["note"] = "packed_masks is None (no tracks)"
        return out

    # Try to read shape/dtype without forcing a full tensor materialization.
    if not isinstance(packed, torch.Tensor):
        try:
            packed = torch.as_tensor(packed)
        except (TypeError, ValueError, RuntimeError):
            out["note"] = (
                f"packed_masks not tensor-coercible: {type(packed).__name__}"
            )
            return out

    out["packed_masks_dtype"] = str(packed.dtype)
    out["packed_masks_shape"] = tuple(int(s) for s in packed.shape)

    if packed.dim() != 4:
        out["note"] = (
            f"packed_masks rank {packed.dim()} != 4 "
            "(expected [N_frames, N_obj, H_m, W_m//8])"
        )
        return out

    n_frames_packed, n_obj, h_m, w_packed = (int(s) for s in packed.shape)
    w_m = w_packed * 8
    out["n_obj"] = n_obj
    out["mask_resolution"] = (h_m, w_m)
    if out["orig_size"] is None:
        # Reflect the 1:1 fallback the adapter would use.
        out["orig_size"] = (h_m, w_m)

    # Per-frame visibility = number of objects with any lit pixel. Capped preview
    # so the inspector never explodes on a long clip; compute via cheap reductions.
    preview_frames = min(n_frames_packed, 64)
    vis: List[int] = []
    if preview_frames > 0 and n_obj > 0:
        sub = packed[:preview_frames].detach().to("cpu")
        # A byte != 0 implies at least one lit pixel in that packed row, so an
        # object is "present" on a frame iff ANY of its packed bytes are nonzero.
        present = (sub != 0).any(dim=-1).any(dim=-1)  # [preview_frames, n_obj]
        vis = [int(c) for c in present.sum(dim=1).tolist()]
    out["visibility_per_frame"] = vis

    return out
