"""mask_adapter.py — native MASK handling + connected components.

Schema-agnostic core for ComfyUI-YEAHUD (brief §6.2 node, §8 align). Pure,
deterministic compute only: **no** ComfyUI imports, no network, no subprocess,
no ``eval``/``exec``, no file IO. Allowed dependencies are the standard library,
:mod:`numpy`, :mod:`PIL` (unused here but allowed), :mod:`torch`, the shared
:mod:`yea_hud.types`, and the sibling :mod:`yea_hud.geometry`.

What this module does
---------------------
ComfyUI's native ``MASK`` type is a ``torch.Tensor`` of float values in
``{0.0, 1.0}`` (the ``.float()`` of a boolean mask). Its batch layout depends on
the producer — verified against the reference ComfyUI 0.24.0 source
(``comfy_extras/nodes_sam3.py`` ``SAM3_Detect.execute``, see
``docs/runtime_findings.md`` §4):

* ``individual_masks=False`` (SAM3 default): one **union** mask per frame
  ``[H, W]`` stacked to ``[B, H, W]``.
* ``individual_masks=True``: per-frame ``[N_obj, H, W]`` concatenated to a
  **flattened** ``[ΣN_obj, H, W]`` — frame boundaries are *not* encoded in the
  tensor (they must come from the per-frame bbox counts).
* No detections: union ``[H, W]`` zeros / individual ``[0, H, W]``.

The general MASK convention is ``[B, H, W]``; some upstream nodes emit a single
``[H, W]`` or a channel-first ``[B, 1, H, W]``. This module accepts all of those.

Three public entry points:

* :func:`normalize_mask_batch` — coerce any MASK shape into a list (length ``B``)
  of ``float32 [H, W]`` arrays in ``0..1``.
* :func:`mask_to_detections` — connected-component analysis of one ``[H, W]``
  mask into :class:`~yea_hud.types.Detection` boxes (xyxy), with a
  ``min_area`` drop filter and a ``merge_distance`` near-box merge. Implemented
  in pure numpy (a flood-fill labeller); an optional OpenCV fast path is used
  when ``cv2`` is importable, producing identical boxes.
* :func:`masks_to_native_bboxes` — for the *YEA Mask to Native BBoxes* node:
  returns the **native** ``BOUNDING_BOX`` runtime value
  (``list[list[{"x","y","width","height","score"}]]`` of absolute-pixel xywh),
  with NO custom type, so the output drops straight into any consumer that
  accepts SAM3/Mediapipe-style boxes.

Coordinate standard: internal boxes are absolute-pixel **xyxy**; the native
``BOUNDING_BOX`` dicts are absolute-pixel **xywh** (``x``/``y`` top-left).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import geometry
from .types import Detection

__all__ = [
    "normalize_mask_batch",
    "mask_to_detections",
    "masks_to_native_bboxes",
]


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public contract)
# ---------------------------------------------------------------------------

def _is_torch_tensor(obj) -> bool:
    """True if ``obj`` is a ``torch.Tensor`` without importing torch eagerly.

    Duck-typing by module/class name keeps this import-order independent and
    cheap even though torch is an allowed dependency.
    """
    return (
        type(obj).__module__.split(".")[0] == "torch"
        and type(obj).__name__ == "Tensor"
    )


def _to_numpy(obj) -> np.ndarray:
    """Convert a torch tensor or array-like to a contiguous numpy array.

    Tensors are detached and moved to CPU first. Never mutates the input.
    """
    if _is_torch_tensor(obj):
        return obj.detach().to("cpu").contiguous().numpy()
    return np.asarray(obj)


def _to_float01(arr: np.ndarray) -> np.ndarray:
    """Coerce an array to ``float32`` clipped to ``[0, 1]``.

    Integer arrays are scaled by their dtype max (so ``uint8`` ``0..255`` maps to
    ``0..1``); booleans map ``False/True -> 0.0/1.0``; floats are clipped only.
    Non-finite values (NaN/Inf) are sanitised to ``0.0`` so a single bad pixel
    can never poison thresholding downstream.
    """
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = float(info.max) if info.max > 0 else 1.0
        out = arr.astype(np.float32) / denom
    elif arr.dtype == np.bool_:
        out = arr.astype(np.float32)
    else:
        out = arr.astype(np.float32, copy=False)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def _as_2d_float01(arr: np.ndarray) -> np.ndarray:
    """Best-effort reduce an array to a single ``float32 [H, W]`` mask in ``0..1``.

    Squeezes a trailing singleton channel (``[H, W, 1]``) and collapses a leading
    singleton/batch dim (``[1, H, W]`` / ``[B, H, W]`` -> first frame). Used only
    on inputs that are already known to represent one mask plane.
    """
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"mask_adapter: cannot interpret shape {arr.shape} as a single [H,W] mask"
        )
    return _to_float01(np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# normalize_mask_batch — coerce any MASK layout to a per-frame list
# ---------------------------------------------------------------------------

def normalize_mask_batch(mask, B: int) -> List[np.ndarray]:
    """Coerce any native MASK tensor/array into a list of per-frame masks.

    Returns a Python ``list`` whose elements are ``float32 [H, W]`` numpy arrays
    in ``0..1`` (one per frame, straight binary-ish values). The handled layouts
    (see ``docs/runtime_findings.md`` §4):

    * ``[B, H, W]`` — the canonical MASK batch: split into ``B`` planes.
    * ``[H, W]`` — a single union mask. If ``B == 1`` it is returned as a
      one-element list; if ``B > 1`` it is **broadcast** to all ``B`` frames
      (the only sensible interpretation of a single union mask for a batch).
    * ``[B, 1, H, W]`` — channel-first batch: the singleton channel is squeezed,
      then split into ``B`` planes.
    * ``[1, B, H, W]`` — a leading singleton wrapper around a channel/batch: the
      leading 1 is squeezed and the result re-interpreted.
    * ``[N, H, W]`` flattened individual-instance masks where ``N != B``: the
      frame boundaries are not encoded in the tensor, so this **cannot** be split
      by frame here. The full list of ``N`` planes is returned **as-is** (length
      ``N``, not ``B``); the caller aligns instances to frames using the per-frame
      bbox counts (findings §4).

    Parameters
    ----------
    mask : torch.Tensor | np.ndarray | None
        The MASK value. ``None`` (unconnected socket) yields ``[]``.
    B : int
        The number of image frames the mask is expected to accompany.

    Returns
    -------
    list[np.ndarray]
        Per-frame (or per-instance, in the flattened case) ``float32 [H, W]``
        masks. May be empty (``mask is None`` or a zero-length batch).

    Notes
    -----
    The ambiguous ``[N, H, W]`` (``N != B``) case is intentionally returned with
    its native length so no information is silently fabricated — the contract
    calls this out explicitly ("cannot split w/o counts -> return as-is list").
    """
    if mask is None:
        return []

    try:
        Bi = int(B)
    except (TypeError, ValueError):
        Bi = 1
    if Bi < 1:
        Bi = 1

    arr = _to_numpy(mask)

    # Squeeze a trailing singleton channel everywhere: [..., 1] -> [...].
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # 2-D: a single [H, W] union mask.
    if arr.ndim == 2:
        plane = _to_float01(np.ascontiguousarray(arr))
        # Broadcast a single union mask across all requested frames.
        return [plane.copy() for _ in range(Bi)]

    # 4-D: channel-first [B, 1, H, W] or a leading-singleton wrapper.
    if arr.ndim == 4:
        # [*, 1, H, W] -> squeeze the channel axis (dim 1).
        if arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        # [1, N, H, W] -> squeeze a leading singleton wrapper.
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            raise ValueError(
                f"normalize_mask_batch: unsupported 4-D MASK shape {arr.shape} "
                "(expected [B,1,H,W] or [1,B,H,W])"
            )

    # 3-D: the [N, H, W] case (canonical batch, or flattened individual masks).
    if arr.ndim == 3:
        n = int(arr.shape[0])
        if n == 0:
            # Empty batch (e.g. SAM3 individual_masks with zero detections): no
            # planes to return.
            return []
        planes = [_to_float01(np.ascontiguousarray(arr[i])) for i in range(n)]
        if n == Bi:
            # Canonical [B, H, W].
            return planes
        if n == 1 and Bi > 1:
            # A single plane wrapped as [1, H, W]: broadcast like the 2-D case.
            return [planes[0].copy() for _ in range(Bi)]
        # n != B and n != 1: flattened individual-instance masks. We cannot
        # recover the frame split without per-frame counts, so return as-is.
        return planes

    raise ValueError(
        f"normalize_mask_batch: cannot interpret MASK shape {np.asarray(mask).shape} "
        "(expected [H,W], [B,H,W], [B,1,H,W], [1,B,H,W], or flattened [N,H,W])"
    )


# ---------------------------------------------------------------------------
# Connected-component labelling
# ---------------------------------------------------------------------------

def _label_components_cv2(binary: np.ndarray):
    """4-connected component labels via OpenCV, or ``None`` if unavailable.

    Returns ``(num_labels, labels)`` where ``labels`` is an ``int32 [H, W]`` array
    with ``0`` = background and ``1..num_labels-1`` = components, matching the
    semantics of the numpy labeller below. ``cv2`` is an *optional* fast path;
    any import/runtime failure returns ``None`` so the caller falls back to the
    pure-numpy implementation. No behavioural difference in the resulting boxes.
    """
    try:
        import cv2  # type: ignore  # optional, never required
    except Exception:  # pragma: no cover - depends on environment
        return None
    try:
        mask_u8 = (binary.astype(np.uint8)) * 255
        num, labels = cv2.connectedComponents(mask_u8, connectivity=4)
        return int(num), labels.astype(np.int32, copy=False)
    except Exception:  # pragma: no cover - defensive
        return None


def _label_components_numpy(binary: np.ndarray) -> Tuple[int, np.ndarray]:
    """4-connected component labelling in pure numpy (iterative flood fill).

    No ``cv2``, no ``scipy`` required. Background (``False``) gets label ``0``;
    each connected foreground region gets a distinct label ``1, 2, ...`` assigned
    in raster-scan (row-major) order of first encounter, so component ordering is
    deterministic across runs and platforms.

    Parameters
    ----------
    binary : np.ndarray
        Boolean ``[H, W]`` mask (``True`` = foreground).

    Returns
    -------
    (num_labels, labels)
        ``num_labels`` includes background (so it is ``1 + n_components``);
        ``labels`` is ``int32 [H, W]``.
    """
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    if not binary.any():
        return 1, labels

    fg = binary
    next_label = 1
    # An explicit stack flood-fill; raster order of seeds gives stable labels.
    # Using a Python list as a LIFO stack keeps memory bounded to the largest
    # component and avoids recursion limits on big blobs.
    stack: List[Tuple[int, int]] = []
    for sy in range(h):
        row_fg = fg[sy]
        row_lab = labels[sy]
        for sx in range(w):
            if not row_fg[sx] or row_lab[sx] != 0:
                continue
            lab = next_label
            next_label += 1
            labels[sy, sx] = lab
            stack.append((sy, sx))
            while stack:
                y, x = stack.pop()
                # 4-connectivity neighbours.
                if y > 0 and fg[y - 1, x] and labels[y - 1, x] == 0:
                    labels[y - 1, x] = lab
                    stack.append((y - 1, x))
                if y + 1 < h and fg[y + 1, x] and labels[y + 1, x] == 0:
                    labels[y + 1, x] = lab
                    stack.append((y + 1, x))
                if x > 0 and fg[y, x - 1] and labels[y, x - 1] == 0:
                    labels[y, x - 1] = lab
                    stack.append((y, x - 1))
                if x + 1 < w and fg[y, x + 1] and labels[y, x + 1] == 0:
                    labels[y, x + 1] = lab
                    stack.append((y, x + 1))
    return next_label, labels


def _component_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    """Connected-component bounding boxes of a boolean mask.

    Returns a list of ``(x1, y1, x2, y2, area)`` tuples — **integer**, half-open
    on the high edge (``x2``/``y2`` are exclusive, i.e. one-past the last
    foreground column/row) so width ``= x2 - x1`` and the box matches numpy
    slicing. ``area`` is the true pixel count of the component (not the box
    area). Components are ordered by raster-scan first appearance (deterministic).

    Uses the OpenCV labeller when available, else the pure-numpy flood fill; the
    resulting boxes are identical either way.
    """
    if binary.ndim != 2:
        raise ValueError(f"_component_boxes expects [H,W], got {binary.shape}")
    if not binary.any():
        return []

    result = _label_components_cv2(binary)
    if result is None:
        num_labels, labels = _label_components_numpy(binary)
    else:
        num_labels, labels = result

    boxes: List[Tuple[int, int, int, int, int]] = []
    # Compute extents per non-background label with vectorised reductions.
    ys, xs = np.nonzero(labels)
    if ys.size == 0:
        return []
    lab_vals = labels[ys, xs]
    # First-appearance order in raster scan == ascending (y, x); np.nonzero
    # already returns row-major order, so the first time we see each label is its
    # raster-scan seed. Build an ordered, de-duplicated label sequence.
    order: List[int] = []
    seen = set()
    for lv in lab_vals:
        lvi = int(lv)
        if lvi not in seen:
            seen.add(lvi)
            order.append(lvi)

    for lv in order:
        sel = lab_vals == lv
        cx = xs[sel]
        cy = ys[sel]
        x1 = int(cx.min())
        y1 = int(cy.min())
        x2 = int(cx.max()) + 1  # half-open
        y2 = int(cy.max()) + 1
        area = int(sel.sum())
        boxes.append((x1, y1, x2, y2, area))
    return boxes


def _merge_near_boxes(
    boxes: List[Tuple[int, int, int, int, int]],
    merge_distance: float,
) -> List[Tuple[int, int, int, int, int]]:
    """Union-merge component boxes whose gap is ``<= merge_distance`` pixels.

    Two boxes are merged when the axis-aligned gap between them (the positive
    separation along x AND y, ``0`` when they touch/overlap) is within
    ``merge_distance``. Merging is transitive (a connected cluster collapses to a
    single enclosing box) and the merged ``area`` is the **sum** of the member
    pixel counts (an upper bound; exact recomputation is unnecessary for the HUD
    and keeps this O(n²) merge cheap). Determinism: clusters are emitted in the
    raster-scan order of their first member.

    ``merge_distance <= 0`` is a no-op (returns the input order unchanged).
    """
    n = len(boxes)
    if n <= 1 or merge_distance <= 0:
        return list(boxes)

    md = float(merge_distance)

    def _gap(a, b) -> float:
        ax1, ay1, ax2, ay2, _ = a
        bx1, by1, bx2, by2, _ = b
        # Separation along each axis (0 if overlapping/touching on that axis).
        dx = max(0, max(bx1 - ax2, ax1 - bx2))
        dy = max(0, max(by1 - ay2, ay1 - by2))
        # Boxes are close enough to merge only if BOTH axis gaps are within md.
        # (Using max => the larger of the two gaps must be <= md, i.e. a true
        # bounding-box proximity, not a diagonal corner technicality.)
        return float(max(dx, dy))

    # Union-Find over the box indices.
    parent = list(range(n))

    def _find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        # Path compression.
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            # Keep the smaller index as the root so cluster identity follows the
            # earliest (raster-scan-first) member -> deterministic emit order.
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _gap(boxes[i], boxes[j]) <= md:
                _union(i, j)

    # Collect clusters, preserving first-member order.
    clusters: "dict[int, List[int]]" = {}
    cluster_order: List[int] = []
    for i in range(n):
        r = _find(i)
        if r not in clusters:
            clusters[r] = []
            cluster_order.append(r)
        clusters[r].append(i)

    merged: List[Tuple[int, int, int, int, int]] = []
    for r in cluster_order:
        members = clusters[r]
        xs1 = min(boxes[m][0] for m in members)
        ys1 = min(boxes[m][1] for m in members)
        xs2 = max(boxes[m][2] for m in members)
        ys2 = max(boxes[m][3] for m in members)
        area = sum(boxes[m][4] for m in members)
        merged.append((xs1, ys1, xs2, ys2, area))
    return merged


# ---------------------------------------------------------------------------
# mask_to_detections — connected components -> Detection boxes
# ---------------------------------------------------------------------------

def mask_to_detections(
    mask_hw,
    frame_index: int,
    *,
    threshold: float,
    min_area: int,
    merge_distance: int,
) -> List[Detection]:
    """Connected-component detections from a single ``[H, W]`` mask.

    The mask is binarised at ``mask > threshold``, split into 4-connected
    components, optionally merged when components are within ``merge_distance``
    pixels, filtered by ``min_area`` (true component pixel count), and emitted as
    :class:`~yea_hud.types.Detection` boxes in absolute-pixel **xyxy**.

    Parameters
    ----------
    mask_hw : torch.Tensor | np.ndarray | None
        A single mask plane. ``[H, W]`` is expected; ``[1, H, W]`` / ``[H, W, 1]``
        are accepted and squeezed. ``None`` or an empty array yields ``[]``.
    frame_index : int
        The frame this mask belongs to; written onto every emitted Detection.
    threshold : float
        Foreground cutoff. A pixel is foreground when ``value > threshold``.
        Clamped to a sane range; the default HUD value is ``0.5``.
    min_area : int
        Components with fewer than ``min_area`` foreground pixels are dropped
        (after merging). ``<= 0`` disables the area filter.
    merge_distance : int
        Components whose bounding boxes are within this many pixels (on both
        axes) are union-merged into one detection. ``<= 0`` disables merging.

    Returns
    -------
    list[Detection]
        ``source_index`` is the component's order (0-based, raster-scan-first /
        post-merge cluster order). ``score`` is ``None`` (a mask carries no
        confidence). ``mask_index`` mirrors ``source_index``. Boxes are
        guaranteed non-degenerate (width/height ``>= 1``) and clamped to the mask
        bounds; degenerate components are dropped.
    """
    if mask_hw is None:
        return []

    arr = _to_numpy(mask_hw)
    if arr.size == 0:
        return []

    plane = _as_2d_float01(arr)
    h, w = plane.shape
    if h == 0 or w == 0:
        return []

    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        thr = 0.5
    if not np.isfinite(thr):
        thr = 0.5
    # Foreground uses strict ``>`` semantics (matches the brief). A negative
    # threshold selects every non-zero pixel; a threshold ``>= 1.0`` selects
    # nothing (a clean 0/1 mask has no pixel strictly greater than 1.0). Both are
    # honoured as-is — no silent clamping that would change which pixels qualify.
    binary = plane > thr
    if not binary.any():
        return []

    try:
        min_a = int(min_area)
    except (TypeError, ValueError):
        min_a = 0
    try:
        merge_d = int(merge_distance)
    except (TypeError, ValueError):
        merge_d = 0

    boxes = _component_boxes(binary)
    if merge_d > 0:
        boxes = _merge_near_boxes(boxes, merge_d)

    detections: List[Detection] = []
    src = 0
    for (x1, y1, x2, y2, comp_area) in boxes:
        if min_a > 0 and comp_area < min_a:
            continue
        # Clamp + normalise via geometry (boxes are already canonical, but this
        # enforces the [0,W]/[0,H] bound and the non-degenerate guarantee).
        nb = geometry.clamp_box((float(x1), float(y1), float(x2), float(y2)), w, h)
        if geometry.is_degenerate(nb, min_size=1.0):
            continue
        bx1, by1, bx2, by2 = nb
        detections.append(
            Detection(
                frame_index=int(frame_index),
                source_index=src,
                x1=bx1,
                y1=by1,
                x2=bx2,
                y2=by2,
                score=None,
                label=None,
                source_track_id=None,
                mask_index=src,
            )
        )
        src += 1
    return detections


# ---------------------------------------------------------------------------
# masks_to_native_bboxes — native BOUNDING_BOX value (NO custom type)
# ---------------------------------------------------------------------------

def masks_to_native_bboxes(
    mask,
    threshold: float,
    min_area: int,
    merge_distance: int,
) -> List[List[dict]]:
    """Convert a MASK batch to the native ``BOUNDING_BOX`` runtime value.

    This backs the *YEA Mask to Native BBoxes* node. The return value matches
    exactly what SAM3 / Mediapipe emit (``docs/runtime_findings.md`` §3) so it
    plugs into any consumer of ``BOUNDING_BOX`` with **no custom type**::

        list[ list[ {"x", "y", "width", "height", "score"} ] ]
          outer length = number of mask frames
          inner = detections in that frame (possibly [])

    Coordinates are **absolute-pixel xywh** floats (``x``/``y`` = top-left
    corner). ``score`` is always ``1.0`` (a mask carries no confidence; SAM3's
    own dicts always include ``score``, so we keep the key present and honest).

    Parameters
    ----------
    mask : torch.Tensor | np.ndarray | None
        Native MASK. Handled layouts (via :func:`normalize_mask_batch` with
        ``B`` inferred from the leading dimension): ``[B, H, W]``, ``[H, W]``,
        ``[B, 1, H, W]``, flattened ``[N, H, W]``. ``None`` -> ``[]``.
    threshold : float
        Foreground cutoff for binarisation (``value > threshold``).
    min_area : int
        Drop components below this many foreground pixels.
    merge_distance : int
        Merge components within this many pixels.

    Returns
    -------
    list[list[dict]]
        Per-frame lists of native bbox dicts. A frame with no qualifying
        component yields an empty inner list (never dropped) so the outer length
        always equals the number of input mask planes.
    """
    if mask is None:
        return []

    arr = _to_numpy(mask)
    if arr.size == 0:
        return []

    # Infer the frame count from the leading dimension so a 2-D union mask maps
    # to exactly one frame (not broadcast), while [B,H,W] / [B,1,H,W] / [N,H,W]
    # map to B / N frames. This keeps the native value's outer length == the
    # number of mask planes actually present.
    work = arr
    if work.ndim >= 3 and work.shape[-1] == 1:
        work = work[..., 0]
    if work.ndim == 4:
        if work.shape[1] == 1:
            work = work[:, 0, :, :]
        elif work.shape[0] == 1:
            work = work[0]
    if work.ndim == 2:
        b_infer = 1
    elif work.ndim == 3:
        b_infer = int(work.shape[0])
    else:
        b_infer = 1

    planes = normalize_mask_batch(mask, b_infer)

    out: List[List[dict]] = []
    for i, plane in enumerate(planes):
        dets = mask_to_detections(
            plane,
            frame_index=i,
            threshold=threshold,
            min_area=min_area,
            merge_distance=merge_distance,
        )
        frame_boxes: List[dict] = []
        for d in dets:
            frame_boxes.append(
                {
                    "x": float(d.x1),
                    "y": float(d.y1),
                    "width": float(d.width),
                    "height": float(d.height),
                    "score": 1.0,
                }
            )
        out.append(frame_boxes)
    return out
