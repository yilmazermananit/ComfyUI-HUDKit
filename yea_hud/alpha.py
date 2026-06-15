"""alpha.py — Load Image alpha recovery + asset preparation (brief §8).

Schema-agnostic. Depends only on the standard library, :mod:`numpy`, :mod:`PIL`
(Pillow), :mod:`torch`, and :mod:`yea_hud.types`. It performs **no** ComfyUI
imports, no network, no subprocess, no file IO — pure deterministic compute.

Image / mask conventions (see ``docs/INTERFACES.md``)
-----------------------------------------------------
* Images that cross *between pure modules* are numpy ``float32 [H, W, 3]`` in
  ``0..1`` (straight, non-premultiplied).
* Overlays / assets are numpy ``float32 [H, W, 4]`` RGBA (straight alpha).
* Alpha / mask single-channel arrays are numpy ``float32 [H, W]`` in ``0..1``.
* torch tensors only appear at the node boundary; this module accepts them on
  the way in (``to_numpy_img`` / ``to_numpy_mask``) and otherwise works in numpy.

ComfyUI ``Load Image`` MASK rule (VERIFIED — source of truth)
-------------------------------------------------------------
ComfyUI's native ``LoadImage`` node builds its MASK output as the **inverse** of
the source PNG alpha channel.  Verified against the reference ComfyUI 0.24.0
source ``nodes.py`` (``class LoadImage.load_image``, lines 1743-1747)::

    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)          # <-- INVERSE of alpha
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

Consequences this module relies on:

1. For a real RGBA PNG, ``MASK == 1.0 - alpha`` and therefore the *true* straight
   alpha is recovered as ``true_alpha = 1.0 - MASK``.  For the classic
   "white graphic on a transparent background" PNG this means the graphic
   (opaque, alpha≈1) shows up as MASK≈0 and the transparent surround
   (alpha≈0) shows up as MASK≈1 — i.e. the MASK is "background-is-white".

2. For a PNG with **no** alpha channel, ``LoadImage`` emits an all-zero
   ``64x64`` MASK (a sentinel, not aligned to the image).  In that case
   inverting would yield a constant ``1.0`` — a useless full-frame alpha — so we
   must instead fall back to the image *luminance* as the alpha.

``recover_alpha`` therefore: (a) inverts MASK to alpha by default, (b) detects
the degenerate / missing / fully-uniform MASK cases and falls back to luminance,
and (c) exposes an ``invert_auto`` override for callers that pass a MASK that is
already a straight alpha (``invert_auto=False`` -> use the mask as-is).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Rec.709 luma coefficients (linear-ish on already-display RGB, matches the
# convention used across the rest of the pack — see palette.luminance).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

__all__ = [
    "to_numpy_img",
    "to_numpy_mask",
    "recover_alpha",
    "luminance_alpha",
    "trim_transparent",
    "recolor_white_asset",
    "resize_rgba",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_torch_tensor(obj) -> bool:
    """True if ``obj`` is a ``torch.Tensor`` without importing torch eagerly.

    We avoid a hard top-level torch import only for robustness of duck-typing;
    torch *is* an allowed dependency, but checking by type name keeps this
    helper cheap and import-order independent.
    """
    return type(obj).__module__.split(".")[0] == "torch" and type(obj).__name__ == "Tensor"


def _to_numpy(obj) -> np.ndarray:
    """Convert a torch tensor or array-like to a contiguous numpy array.

    Always detaches and moves to CPU for tensors. Never mutates the input.
    """
    if _is_torch_tensor(obj):
        return obj.detach().to("cpu").contiguous().numpy()
    return np.asarray(obj)


def _clip01(a: np.ndarray) -> np.ndarray:
    """Clip to ``[0, 1]`` as float32 (out-of-place)."""
    return np.clip(a.astype(np.float32, copy=False), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Tensor / array ingestion
# ---------------------------------------------------------------------------

def to_numpy_img(image) -> np.ndarray:
    """Normalize any IMAGE-shaped input to numpy ``float32 [H, W, 3]`` in ``0..1``.

    Accepts:

    * torch tensors or numpy arrays shaped ``[H, W, C]`` or ``[1, H, W, C]``
      (the ComfyUI IMAGE batch convention is ``[B, H, W, C]``; a single frame is
      taken when a leading batch dim of length 1 is present, otherwise the first
      frame of a larger batch is used).
    * ``C`` may be 1 (grayscale -> replicated to 3), 3 (RGB), or 4 (RGBA ->
      RGB taken, alpha dropped here; use :func:`recover_alpha` for alpha).
    * A 2-D ``[H, W]`` array is treated as grayscale.

    Integer inputs (e.g. uint8 ``0..255``) are scaled to ``0..1``; float inputs
    are assumed already ``0..1`` and only clipped. The result is a fresh,
    contiguous, writable array — the caller may mutate it freely.

    Raises:
        TypeError: if ``image`` is None.
        ValueError: if the shape cannot be interpreted as an image.
    """
    if image is None:
        raise TypeError("to_numpy_img: image is None")

    arr = _to_numpy(image)

    # Collapse a leading batch dimension. A 4-D array is [B, H, W, C]; take the
    # first frame (batch of 1 is the common Load Image case).
    if arr.ndim == 4:
        if arr.shape[0] < 1:
            raise ValueError(f"to_numpy_img: empty batch shape {arr.shape}")
        arr = arr[0]

    if arr.ndim == 2:
        # [H, W] grayscale -> [H, W, 1]
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError(
            f"to_numpy_img: cannot interpret shape {arr.shape} as an image "
            "(expected [H,W], [H,W,C] or [1,H,W,C])"
        )

    # Scale integer dtypes from their max range to 0..1.
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = float(info.max) if info.max > 0 else 1.0
        arr = arr.astype(np.float32) / denom
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    c = arr.shape[2]
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif c == 2:
        # Unusual (gray+alpha): replicate gray to RGB, drop alpha.
        arr = np.repeat(arr[:, :, :1], 3, axis=2)
    elif c >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"to_numpy_img: unsupported channel count {c}")

    # Ensure a fresh, contiguous, writable float32 buffer in 0..1.
    return np.ascontiguousarray(_clip01(arr))


def to_numpy_mask(mask) -> Optional[np.ndarray]:
    """Normalize any MASK-shaped input to numpy ``float32 [H, W]`` in ``0..1``.

    Accepts torch tensors / numpy arrays shaped ``[H, W]``, ``[1, H, W]``,
    ``[B, H, W]`` (first frame taken), ``[H, W, 1]``, or ``[1, H, W, 1]``.

    Returns ``None`` when ``mask`` is ``None`` (an unconnected MASK socket).
    Integer inputs are scaled by their dtype max; floats are clipped to ``0..1``.
    The returned array is fresh and contiguous.

    Raises:
        ValueError: if the shape cannot be interpreted as a single-channel mask.
    """
    if mask is None:
        return None

    arr = _to_numpy(mask)

    # Squeeze trailing singleton channel: [..., 1] -> [...].
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # Collapse leading batch dim for [B, H, W] / [1, H, W].
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            raise ValueError(f"to_numpy_mask: empty batch shape {arr.shape}")
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(
            f"to_numpy_mask: cannot interpret shape {arr.shape} as a mask "
            "(expected [H,W], [1,H,W], [B,H,W], [H,W,1])"
        )

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        denom = float(info.max) if info.max > 0 else 1.0
        arr = arr.astype(np.float32) / denom
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    return np.ascontiguousarray(_clip01(arr))


# ---------------------------------------------------------------------------
# Alpha recovery (brief §8)
# ---------------------------------------------------------------------------

def luminance_alpha(image_np) -> np.ndarray:
    """Per-pixel Rec.709 luma of ``image_np`` as a straight-alpha fallback.

    ``image_np`` is anything :func:`to_numpy_img` accepts; the result is
    ``float32 [H, W]`` in ``0..1``. Used when no usable alpha/mask exists (the
    classic "white graphic with no real alpha channel" case): the brightness of
    the graphic doubles as its opacity, so a white-on-black logo recovers a
    clean alpha matte.
    """
    rgb = to_numpy_img(image_np)
    luma = rgb.reshape(-1, 3) @ _LUMA
    return _clip01(luma.reshape(rgb.shape[0], rgb.shape[1]))


def recover_alpha(image_np, mask_np, *, invert_auto: bool = True) -> np.ndarray:
    """Recover a straight alpha matte ``float32 [H, W]`` in ``0..1`` from a
    ComfyUI Load Image ``(IMAGE, MASK)`` pair.

    Implements the verified ComfyUI ``LoadImage`` MASK rule (see module
    docstring): the native MASK is the **inverse** of the source PNG alpha, so
    the true alpha is ``1.0 - mask``.

    Decision logic:

    * ``image_np`` is normalized to ``[H, W, 3]`` via :func:`to_numpy_img`. The
      output alpha always matches the image's ``(H, W)``.
    * If ``mask_np`` is ``None`` -> luminance fallback (no alpha was connected).
    * Else ``mask_np`` is normalized via :func:`to_numpy_mask`. The following
      "no usable alpha" cases also fall back to luminance:

      - the mask shape does **not** match the image ``(H, W)`` — this is the
        Load Image *no-alpha sentinel* (a degenerate ``64x64`` zero mask), and
        also any otherwise-misaligned mask we cannot trust;
      - the mask is (near-)constant everywhere — a fully-transparent
        (mask all ``1`` -> alpha all ``0``) or fully-opaque (mask all ``0``)
        recovery carries no shape information, so luminance is more useful.

    * Otherwise, when ``invert_auto`` is ``True`` (default), the alpha is
      ``1.0 - mask`` (the inverse rule). When ``invert_auto`` is ``False`` the
      caller is asserting that ``mask_np`` is *already* a straight alpha matte
      (not a Load Image inverse), and it is used verbatim.

    Args:
        image_np: source IMAGE (torch/np, any shape :func:`to_numpy_img` takes).
        mask_np: source MASK (torch/np) or ``None``.
        invert_auto: apply the ``1 - mask`` inverse rule (``True``) or treat the
            mask as straight alpha (``False``).

    Returns:
        ``np.ndarray`` straight alpha, ``float32 [H, W]`` in ``0..1``.
    """
    rgb = to_numpy_img(image_np)
    h, w = rgb.shape[0], rgb.shape[1]

    mask = to_numpy_mask(mask_np)

    # No mask connected at all -> luminance fallback.
    if mask is None:
        return luminance_alpha(rgb)

    # Mask present but misaligned to the image -> Load Image no-alpha sentinel
    # (degenerate 64x64 zeros) or otherwise untrustworthy -> luminance fallback.
    if mask.shape[0] != h or mask.shape[1] != w:
        return luminance_alpha(rgb)

    if invert_auto:
        # Detect a (near-)constant mask: either fully transparent or fully
        # opaque after inversion carries no spatial information -> fall back.
        mn = float(mask.min())
        mx = float(mask.max())
        if (mx - mn) < 1e-6:
            return luminance_alpha(rgb)
        return _clip01(1.0 - mask)

    # invert_auto False: trust the mask as a straight alpha matte verbatim.
    return _clip01(mask)


# ---------------------------------------------------------------------------
# Trimming (brief §8)
# ---------------------------------------------------------------------------

def trim_transparent(
    rgb_np,
    alpha_np,
    threshold: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Crop ``rgb_np`` / ``alpha_np`` to the bounding box of pixels whose alpha
    exceeds ``threshold``.

    Args:
        rgb_np: ``[H, W, 3]`` float (anything :func:`to_numpy_img` accepts).
        alpha_np: ``[H, W]`` float (anything :func:`to_numpy_mask` accepts).
        threshold: alpha values ``> threshold`` are considered opaque content.

    Returns:
        ``(cropped_rgb, cropped_alpha, bbox)`` where ``bbox`` is
        ``(x1, y1, x2, y2)`` in the **original** coordinate frame (xyxy,
        half-open: the crop is ``[y1:y2, x1:x2]``).

        If the asset is fully transparent (no pixel above ``threshold``), the
        inputs are returned **unchanged** with the full-frame bbox
        ``(0, 0, W, H)``.

    The returned crops are contiguous views' copies (safe to mutate). RGB and
    alpha are resized to a common ``(H, W)`` first (nearest) if they differ, so a
    mismatched pair still trims coherently.
    """
    rgb = to_numpy_img(rgb_np)
    alpha = to_numpy_mask(alpha_np)
    if alpha is None:
        # Treat a missing alpha as fully opaque -> nothing to trim.
        alpha = np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)

    h, w = rgb.shape[0], rgb.shape[1]

    # Coerce alpha to the image grid if they disagree (defensive).
    if alpha.shape[0] != h or alpha.shape[1] != w:
        alpha = _resize_single_channel_nearest(alpha, w, h)

    opaque = alpha > float(threshold)
    if not opaque.any():
        # Fully transparent -> return inputs unchanged, full-frame bbox.
        return rgb, alpha, (0, 0, w, h)

    rows = np.where(opaque.any(axis=1))[0]
    cols = np.where(opaque.any(axis=0))[0]
    y1 = int(rows[0])
    y2 = int(rows[-1]) + 1          # half-open
    x1 = int(cols[0])
    x2 = int(cols[-1]) + 1

    cropped_rgb = np.ascontiguousarray(rgb[y1:y2, x1:x2, :])
    cropped_alpha = np.ascontiguousarray(alpha[y1:y2, x1:x2])
    return cropped_rgb, cropped_alpha, (x1, y1, x2, y2)


def _resize_single_channel_nearest(a: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Nearest-neighbour resize of a single-channel ``[H, W]`` array.

    Used internally to reconcile a mask whose dims drifted from the image.
    Pure numpy index sampling (no PIL) so it is dtype/range agnostic.
    """
    in_h, in_w = a.shape[0], a.shape[1]
    out_h = max(1, int(out_h))
    out_w = max(1, int(out_w))
    ys = (np.arange(out_h) * (in_h / out_h)).astype(np.int64)
    xs = (np.arange(out_w) * (in_w / out_w)).astype(np.int64)
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)
    return np.ascontiguousarray(a[ys][:, xs].astype(np.float32, copy=False))


# ---------------------------------------------------------------------------
# Recolor (brief §8)
# ---------------------------------------------------------------------------

def recolor_white_asset(rgb_np, alpha_np, color_rgb, opacity: float) -> np.ndarray:
    """Tint a white/grayscale asset to ``color_rgb`` and return straight-alpha RGBA.

    Brief §8 rule::

        out_rgb = color_rgb * source_luminance
        out_a   = alpha * opacity

    The source's *luminance* (Rec.709) modulates the chosen color, so a white
    graphic becomes a flat ``color_rgb`` shape, mid-grays become darker tints,
    and black stays black — preserving the graphic's internal shading while
    recoloring it to the track's palette color.

    Args:
        rgb_np: source ``[H, W, 3]`` (anything :func:`to_numpy_img` accepts).
        alpha_np: source straight alpha ``[H, W]`` in ``0..1`` (or ``None`` ->
            treated as fully opaque).
        color_rgb: target ``(r, g, b)`` floats in ``0..1`` (clamped defensively).
        opacity: global multiplier on the output alpha, in ``0..1`` (clamped).

    Returns:
        ``np.ndarray`` RGBA ``float32 [H, W, 4]`` in ``0..1`` (straight alpha,
        NOT premultiplied).
    """
    rgb = to_numpy_img(rgb_np)
    h, w = rgb.shape[0], rgb.shape[1]

    alpha = to_numpy_mask(alpha_np)
    if alpha is None:
        alpha = np.ones((h, w), dtype=np.float32)
    elif alpha.shape[0] != h or alpha.shape[1] != w:
        alpha = _resize_single_channel_nearest(alpha, w, h)

    # Source luminance modulates the target color.
    luma = (rgb.reshape(-1, 3) @ _LUMA).reshape(h, w)
    luma = _clip01(luma)

    cr, cg, cb = (
        float(np.clip(color_rgb[0], 0.0, 1.0)),
        float(np.clip(color_rgb[1], 0.0, 1.0)),
        float(np.clip(color_rgb[2], 0.0, 1.0)),
    )
    op = float(np.clip(opacity, 0.0, 1.0))

    out = np.empty((h, w, 4), dtype=np.float32)
    out[:, :, 0] = luma * cr
    out[:, :, 1] = luma * cg
    out[:, :, 2] = luma * cb
    out[:, :, 3] = _clip01(alpha * op)
    return out


# ---------------------------------------------------------------------------
# Resize (brief §8)
# ---------------------------------------------------------------------------

def _pil_resampling():
    """Return ``(LANCZOS, BICUBIC)`` resampling constants across Pillow versions.

    Pillow >= 9.1 exposes ``Image.Resampling.*``; older versions only have the
    top-level ``Image.LANCZOS`` / ``Image.BICUBIC`` aliases.
    """
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS, resampling.BICUBIC


def resize_rgba(rgba_np, out_w: int, out_h: int) -> np.ndarray:
    """Resize a straight-alpha RGBA array to ``(out_w, out_h)`` without alpha bleed.

    Uses **Lanczos** when downscaling (either axis shrinks) and **Bicubic** when
    upscaling, via PIL. Each channel — including alpha — is resampled
    independently on a straight (non-premultiplied) RGBA image, so no color
    leaks across the alpha edge (no premultiply/unpremultiply round-trip is
    performed; PIL resamples the raw straight channels).

    Args:
        rgba_np: ``[H, W, 4]`` (or ``[H, W, 3]`` -> alpha assumed opaque, or
            ``[H, W]`` -> treated as a single-channel alpha promoted to RGBA with
            white RGB). Anything :func:`to_numpy_img`-ish; accepted broadly.
        out_w: target width in pixels (``>= 1``).
        out_h: target height in pixels (``>= 1``).

    Returns:
        ``np.ndarray`` RGBA ``float32 [out_h, out_w, 4]`` in ``0..1``.

    Notes:
        If the input already matches ``(out_h, out_w)`` it is returned as a
        clipped float32 copy (no resample, no quality loss).
    """
    from PIL import Image

    arr = _to_numpy(rgba_np)

    # Promote to [H, W, 4] straight RGBA.
    if arr.ndim == 2:
        # single-channel -> alpha matte on white RGB
        h, w = arr.shape
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, 3] = _clip01(_scale_to_unit(arr))
        arr = rgba
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        rgb = _scale_to_unit(arr)
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = _clip01(rgb)
        rgba[:, :, 3] = 1.0
        arr = rgba
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = _clip01(_scale_to_unit(arr))
    else:
        raise ValueError(
            f"resize_rgba: expected [H,W], [H,W,3] or [H,W,4], got {np.shape(rgba_np)}"
        )

    in_h, in_w = arr.shape[0], arr.shape[1]
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))

    if in_w == out_w and in_h == out_h:
        return np.ascontiguousarray(arr.astype(np.float32, copy=True))

    lanczos, bicubic = _pil_resampling()
    # Downscale on EITHER axis -> Lanczos (best antialiasing); pure upscale -> Bicubic.
    downscaling = (out_w < in_w) or (out_h < in_h)
    resample = lanczos if downscaling else bicubic

    # Round-trip through PIL in 8-bit per channel on the STRAIGHT alpha image.
    u8 = np.ascontiguousarray((arr * 255.0 + 0.5).astype(np.uint8))
    img = Image.fromarray(u8, mode="RGBA")
    img = img.resize((out_w, out_h), resample=resample)
    out = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(_clip01(out))


def _scale_to_unit(a: np.ndarray) -> np.ndarray:
    """Scale an integer array to ``0..1`` float32; pass float arrays through.

    Booleans -> ``{0,1}``. Used so ``resize_rgba`` accepts uint8 RGBA too.
    """
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        denom = float(info.max) if info.max > 0 else 1.0
        return a.astype(np.float32) / denom
    if a.dtype == np.bool_:
        return a.astype(np.float32)
    return a.astype(np.float32, copy=False)
