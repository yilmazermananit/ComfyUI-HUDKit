"""assets.py — user PNG asset slot management (brief §16).

Schema-agnostic. Imports only the Python standard library, :mod:`numpy`, and the
sibling :mod:`yonder_hud` modules :mod:`alpha`, :mod:`palette`, :mod:`utils`
(plus :mod:`yonder_hud.types` for shared constants). It performs **no** ComfyUI
imports, no network, no subprocess, no ``eval``/``exec``, and no file IO — pure
deterministic compute, safe to run inside Comfy Cloud.

What this module owns
---------------------
The HUD renderer lets the user wire up to *N* transparent "frame" PNGs (decorative
brackets / scan-frames drawn around a tracked box) and up to *N* "target" PNGs
(reticles drawn at a tracked box's center). Each PNG arrives as a ComfyUI
``(IMAGE, MASK)`` pair via a Load Image node. Slots the user did not connect show
up as ``None``.

:class:`AssetBank` turns those raw, sparsely-connected slot lists into a clean,
compact, deterministic asset service for the tracker / compositor:

* It builds **eligible index lists** of the *connected* slots only. The tracker
  picks ``frame_asset_id`` uniformly in ``[0, n_frame_assets)`` (the *eligible*
  index space — see ``yonder_hud.tracker.Tracker.assign_style``), so the eligible
  numbering must be a contiguous ``0..count-1`` over connected slots and never
  expose the gaps left by unconnected sockets.
* :meth:`frame_rgba` / :meth:`target_rgba` recolor a connected asset to a palette
  color and resize it to (approximately) the requested pixel size, returning a
  straight-alpha RGBA ``float32 [h, w, 4]`` overlay — or ``None`` when the slot
  cannot be rendered.
* Results are cached in a :class:`yonder_hud.utils.LRUCache` keyed by
  ``(eligible_id, w_bucket, h_bucket, color_id)`` so the per-frame, per-track
  recolor+resize work (the expensive part) is reused across the many frames where
  a track's box only wobbles by a pixel or two.

Connectivity / render rules (brief §16, verbatim contract)
----------------------------------------------------------
A slot's pair is one of:

* ``None`` (or ``(None, None)``) — **nothing connected** -> not eligible, never
  drawn, does not occupy an eligible index.
* ``(image, mask)`` — **image + mask connected** -> eligible; alpha is recovered
  from the ComfyUI Load Image MASK via :func:`yonder_hud.alpha.recover_alpha`.
* ``(image, None)`` — **image connected, mask not** -> eligible; alpha falls back
  to the image *luminance* (:func:`yonder_hud.alpha.luminance_alpha`).
* ``(None, mask)`` — **mask connected, image not** -> eligible (the socket *is*
  connected, so it occupies an eligible index the tracker may pick), but the asset
  is **NOT rendered**: :meth:`frame_rgba` / :meth:`target_rgba` return ``None``.
  This is the documented brief §16 behaviour ("Mask connected but image not ->
  asset NOT rendered").

Keeping a mask-only slot eligible (rather than dropping it) preserves the
1:1 mapping between what the user wired and the eligible index space, so the
deterministic ``hash(seed, track_id) % n_assets`` selection in the tracker stays
stable when a user connects a mask before its image.

Cache & output-size contract
-----------------------------
The contract (``docs/INTERFACES.md`` assets.py) is "recolor+resize the eligible
frame asset to ``out_w``/``out_h``; cached by ``(id, w-bucket, h-bucket,
color_id)``" with the buckets at 8px granularity. To make the cached value and
the returned value identical (so a cache hit returns exactly what a miss would),
the requested ``(out_w, out_h)`` are **snapped to a multiple of 8** (minimum 8px),
and the asset is produced and cached at that snapped size. A box's sub-8px wobble
from EMA smoothing therefore collapses onto a single cache entry — which is the
whole point of bucketing — and the returned overlay's dimensions are a stable,
deterministic function of the bucket rather than of the exact pixel-jittered box.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import alpha as alpha_mod
from . import palette as palette_mod
from . import utils as utils_mod
from .types import ASSET_FIT_MODES

__all__ = [
    "AssetBank",
    "color_to_cache_id",
    "snap_bucket",
]

# A user (IMAGE, MASK) pair for one slot, or ``None`` for an unconnected slot.
# Either element of the tuple may itself be ``None`` (image-only / mask-only).
Pair = Optional[Tuple[object, object]]

# Asset overlays are quantised to an 8px grid for caching (brief §16 / INTERFACES).
_BUCKET = 8


def snap_bucket(value: int, bucket: int = _BUCKET) -> int:
    """Snap a pixel length down to the nearest positive multiple of ``bucket``.

    Used to derive a stable cache key (and the produced overlay size) from a
    pixel-jittered box dimension. The result is always ``>= bucket`` so an asset
    is never produced at a zero / degenerate size.

    Args:
        value: A pixel length (may be float-ish, negative, or zero).
        bucket: The grid size (default 8).

    Returns:
        ``max(bucket, (int(value) // bucket) * bucket)`` — i.e. floored to the
        bucket grid with a one-bucket floor.

    Examples:
        ``snap_bucket(1) == 8``, ``snap_bucket(8) == 8``, ``snap_bucket(15) == 8``,
        ``snap_bucket(16) == 16``, ``snap_bucket(0) == 8``, ``snap_bucket(-4) == 8``.
    """
    b = int(bucket)
    if b <= 0:
        b = _BUCKET
    v = int(value)
    snapped = (v // b) * b
    if snapped < b:
        snapped = b
    return snapped


def color_to_cache_id(color_rgb: Sequence[float]) -> int:
    """Quantise an ``(r, g, b)`` float color into a small stable integer cache id.

    :meth:`AssetBank.frame_rgba` receives a *color* (the resolved palette RGB), not
    a palette index, so the cache key's ``color_id`` is derived here by clamping
    each channel to ``0..1`` and quantising to 8 bits. Two calls with the same
    color therefore hit the same cache entry, and two visually-distinct colors
    (>= 1/255 apart on any channel) never collide. Out-of-range or non-finite
    components are folded to the nearest valid byte (``NaN`` -> 0) so a degenerate
    color can never raise here.

    Args:
        color_rgb: An ``(r, g, b)`` triple (indexable of length >= 3) in ``0..1``.

    Returns:
        int: A value in ``[0, 256**3)`` uniquely identifying the quantised color.
    """
    def _q(x: float) -> int:
        xf = float(x)
        if not (xf == xf):  # NaN
            return 0
        if xf <= 0.0:
            return 0
        if xf >= 1.0:
            return 255
        return int(xf * 255.0 + 0.5)

    r = _q(color_rgb[0])
    g = _q(color_rgb[1])
    b = _q(color_rgb[2])
    return (r << 16) | (g << 8) | b


class AssetBank:
    """Manage user frame/target PNG slots and serve recolored, resized overlays.

    Construction takes the raw per-slot ``(IMAGE, MASK)`` pairs exactly as the
    node layer collected them from the optional Load Image inputs; entries are
    ``None`` for unconnected slots. The bank computes, once, the *eligible* index
    lists (connected slots only) and exposes :attr:`n_frame_assets` /
    :attr:`n_target_assets` (the contiguous eligible counts the tracker selects
    over) plus :meth:`frame_rgba` / :meth:`target_rgba` accessors.

    The accessors are pure and deterministic: the same
    ``(eligible_id, out_w, out_h, color_rgb, opacity)`` always yields the same
    overlay (a fresh array on each call — the cache stores the canonical result;
    callers may mutate the returned copy). A second :class:`AssetBank` built from
    the same pairs behaves identically.

    Args:
        frame_pairs: Per-slot frame ``(image, mask)`` pairs (``None`` = unconnected
            slot; either tuple element may be ``None``).
        target_pairs: Per-slot target ``(image, mask)`` pairs, same convention.
        trim: When ``True``, each rendered asset is cropped to its alpha bounding
            box (:func:`yonder_hud.alpha.trim_transparent`) before resize, removing
            transparent margins so the artwork fills the placement box.
        fit_frame: Frame asset fit mode (one of
            :data:`yonder_hud.types.ASSET_FIT_MODES`). Stored for the
            compositor's placement-rect computation; the bank itself resizes to the
            exact requested size (the compositor decides the box via
            ``geometry.asset_rect``). An unknown value is coerced to ``'stretch'``.
        fit_target: Target asset fit mode, same convention.
        max_cache_items: Capacity of the internal recolor/resize LRU cache.
    """

    def __init__(
        self,
        frame_pairs: Sequence[Pair],
        target_pairs: Sequence[Pair],
        *,
        trim: bool,
        fit_frame: str,
        fit_target: str,
        max_cache_items: int,
    ) -> None:
        self.trim: bool = bool(trim)
        self.fit_frame: str = fit_frame if fit_frame in ASSET_FIT_MODES else "stretch"
        self.fit_target: str = fit_target if fit_target in ASSET_FIT_MODES else "stretch"

        # Normalise the raw slot lists and compute eligible (connected) indices.
        self._frame_slots: List[Tuple[Optional[object], Optional[object]]] = (
            self._normalize_slots(frame_pairs)
        )
        self._target_slots: List[Tuple[Optional[object], Optional[object]]] = (
            self._normalize_slots(target_pairs)
        )

        # Eligible index lists: positions (into the normalised slot lists) of every
        # CONNECTED slot, in slot order. The eligible id the tracker hands us is an
        # index INTO these lists (compact 0..count-1), not the raw slot index.
        self._frame_eligible: List[int] = [
            i for i, slot in enumerate(self._frame_slots) if self._is_connected(slot)
        ]
        self._target_eligible: List[int] = [
            i for i, slot in enumerate(self._target_slots) if self._is_connected(slot)
        ]

        self._cache = utils_mod.LRUCache(max_items=int(max_cache_items))

    # ------------------------------------------------------------------ #
    # Slot normalisation / eligibility
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_slots(
        pairs: Optional[Sequence[Pair]],
    ) -> List[Tuple[Optional[object], Optional[object]]]:
        """Coerce a raw slot list into a list of ``(image_or_None, mask_or_None)``.

        Accepts ``None`` (no slots), and per-entry: ``None`` (unconnected slot,
        kept as ``(None, None)``), a 2-tuple/2-list ``(image, mask)``, or a single
        non-tuple object treated as an image-only slot ``(image, None)``. Any
        malformed entry degrades to ``(None, None)`` rather than raising, so a
        surprising node wiring can never crash the bank.
        """
        if pairs is None:
            return []
        out: List[Tuple[Optional[object], Optional[object]]] = []
        for entry in pairs:
            if entry is None:
                out.append((None, None))
                continue
            if isinstance(entry, (tuple, list)):
                if len(entry) >= 2:
                    out.append((entry[0], entry[1]))
                elif len(entry) == 1:
                    out.append((entry[0], None))
                else:
                    out.append((None, None))
                continue
            # A bare object -> image-only slot.
            out.append((entry, None))
        return out

    @staticmethod
    def _is_connected(slot: Tuple[Optional[object], Optional[object]]) -> bool:
        """A slot is connected (eligible) if its image OR its mask socket is wired.

        ``(None, None)`` -> not connected. ``(image, _)`` or ``(_, mask)`` with a
        non-``None`` element -> connected. Note a mask-only slot is *connected*
        (occupies an eligible index) but renders to ``None`` (brief §16).
        """
        img, mask = slot
        return img is not None or mask is not None

    # ------------------------------------------------------------------ #
    # Eligible counts (the tracker selects ``[0, n)``)
    # ------------------------------------------------------------------ #

    @property
    def n_frame_assets(self) -> int:
        """Number of eligible (connected) frame slots — the tracker picks ids in
        ``[0, n_frame_assets)``."""
        return len(self._frame_eligible)

    @property
    def n_target_assets(self) -> int:
        """Number of eligible (connected) target slots — the tracker picks ids in
        ``[0, n_target_assets)``."""
        return len(self._target_eligible)

    # ------------------------------------------------------------------ #
    # Public overlay accessors
    # ------------------------------------------------------------------ #

    def frame_rgba(
        self,
        eligible_id: int,
        out_w: int,
        out_h: int,
        color_rgb,
        opacity: float,
    ) -> Optional[np.ndarray]:
        """Return the recolored, resized frame asset at eligible index ``eligible_id``.

        See :meth:`_render` for the full recolor/trim/resize/cache pipeline.

        Args:
            eligible_id: Index into the *eligible* frame list, ``[0, n_frame_assets)``.
                Out-of-range / negative -> ``None``.
            out_w: Requested overlay width in pixels (snapped to an 8px bucket).
            out_h: Requested overlay height in pixels (snapped to an 8px bucket).
            color_rgb: Target ``(r, g, b)`` palette color in ``0..1``.
            opacity: Global alpha multiplier ``0..1``.

        Returns:
            Straight-alpha RGBA ``float32 [h, w, 4]`` in ``0..1`` (``h``/``w`` are
            the 8px-snapped sizes), or ``None`` when the slot is image-less
            (mask-only) or the index is out of range.
        """
        return self._render(
            self._frame_slots,
            self._frame_eligible,
            "F",
            eligible_id,
            out_w,
            out_h,
            color_rgb,
            opacity,
        )

    def target_rgba(
        self,
        eligible_id: int,
        out_w: int,
        out_h: int,
        color_rgb,
        opacity: float,
    ) -> Optional[np.ndarray]:
        """Return the recolored, resized target asset at eligible index ``eligible_id``.

        Identical contract to :meth:`frame_rgba` but over the target slot list /
        eligible numbering / fit mode. See :meth:`_render`.

        Args:
            eligible_id: Index into the *eligible* target list, ``[0, n_target_assets)``.
            out_w: Requested overlay width in pixels (snapped to an 8px bucket).
            out_h: Requested overlay height in pixels (snapped to an 8px bucket).
            color_rgb: Target ``(r, g, b)`` palette color in ``0..1``.
            opacity: Global alpha multiplier ``0..1``.

        Returns:
            Straight-alpha RGBA ``float32 [h, w, 4]`` or ``None``.
        """
        return self._render(
            self._target_slots,
            self._target_eligible,
            "T",
            eligible_id,
            out_w,
            out_h,
            color_rgb,
            opacity,
        )

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #

    def _render(
        self,
        slots: List[Tuple[Optional[object], Optional[object]]],
        eligible: List[int],
        kind: str,
        eligible_id: int,
        out_w: int,
        out_h: int,
        color_rgb,
        opacity: float,
    ) -> Optional[np.ndarray]:
        """Resolve, recolor, (optionally) trim, resize and cache one asset overlay.

        Pipeline (brief §8 / §16):

        1. Map ``eligible_id`` -> raw slot via ``eligible``. Out-of-range -> ``None``.
        2. ``(None, mask)`` (no image) -> ``None`` (mask-only slot, not rendered).
        3. Recover straight alpha:
           ``recover_alpha(image, mask)`` when a mask is present, else
           ``luminance_alpha(image)`` (image-only slot).
        4. If :attr:`trim`, crop to the alpha bbox (``trim_transparent``).
        5. Recolor to ``color_rgb`` at **full opacity** (``recolor_white_asset`` ->
           ``out_rgb = color * luma``, ``out_a = alpha``).
        6. Resize to the 8px-snapped ``(out_w, out_h)`` (``resize_rgba``).
        7. Apply ``opacity`` as a final alpha multiplier (post-cache).

        Steps 3-6 (the expensive recolor+resize) are cached on the contract key
        ``(kind, eligible_id, w_bucket, h_bucket, color_id)`` — opacity is
        deliberately **not** part of the key: it is a cheap scalar applied to the
        alpha channel of a fresh copy *after* the lookup. This keeps the cache hot
        when ``opacity`` varies per frame (the compositor's micro-variation can
        jitter it every frame) and matches the documented 4-part key exactly. A
        defensive copy is always returned so callers may mutate it safely.

        Returns:
            ``np.ndarray`` RGBA ``float32 [h, w, 4]`` or ``None``.
        """
        # 1. eligible index -> raw slot.
        if eligible_id is None:
            return None
        try:
            idx = int(eligible_id)
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(eligible):
            return None

        slot_index = eligible[idx]
        image, mask = slots[slot_index]

        # 2. mask-only slot -> not rendered.
        if image is None:
            return None

        # Snap requested size to the 8px bucket grid (cache + output stability).
        w_bucket = snap_bucket(out_w)
        h_bucket = snap_bucket(out_h)
        color_id = color_to_cache_id(color_rgb)

        cache_key = (kind, idx, w_bucket, h_bucket, color_id)
        cached = self._cache.get(cache_key)
        if cached is None:
            # 3. recover straight alpha (mask if present else luminance fallback).
            rgb = alpha_mod.to_numpy_img(image)
            if mask is not None:
                asset_alpha = alpha_mod.recover_alpha(rgb, mask, invert_auto=True)
            else:
                asset_alpha = alpha_mod.luminance_alpha(rgb)

            # 4. optional trim to alpha bbox (drop transparent margins).
            if self.trim:
                rgb, asset_alpha, _bbox = alpha_mod.trim_transparent(rgb, asset_alpha)

            # 5. recolor to the palette color at FULL opacity -> RGBA. Opacity is
            #    applied post-cache (step 7) so it is not baked into the cached art.
            rgba = alpha_mod.recolor_white_asset(rgb, asset_alpha, color_rgb, 1.0)

            # 6. resize to the snapped target size (Lanczos down / Bicubic up).
            rgba = alpha_mod.resize_rgba(rgba, w_bucket, h_bucket)
            cached = np.ascontiguousarray(rgba.astype(np.float32, copy=False))
            self._cache.put(cache_key, cached)

        # 7. apply opacity to a fresh copy (never mutate the cached canonical art).
        op = float(np.clip(opacity, 0.0, 1.0))
        out = np.array(cached, dtype=np.float32, copy=True)
        if op != 1.0:
            out[:, :, 3] *= op
        return out

    # ------------------------------------------------------------------ #
    # Introspection / housekeeping
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Drop all cached recolored/resized overlays (frees memory; results are
        recomputed deterministically on demand)."""
        self._cache.clear()

    def __len__(self) -> int:
        """Total eligible asset count (frame + target)."""
        return self.n_frame_assets + self.n_target_assets

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            "AssetBank("
            f"n_frame_assets={self.n_frame_assets}, "
            f"n_target_assets={self.n_target_assets}, "
            f"trim={self.trim}, fit_frame={self.fit_frame!r}, "
            f"fit_target={self.fit_target!r}, cache={len(self._cache)})"
        )
