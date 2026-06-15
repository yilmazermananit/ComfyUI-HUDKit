"""Tests for ``yonder_hud.alpha`` (brief §25.2).

Runnable two ways:

    /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_alpha.py -q
    /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_alpha.py

The ``__main__`` block runs every ``test_*`` function, prints a summary, and
``sys.exit(1)`` on any failure. No ComfyUI, no network, no model files — only
small synthetic numpy/torch fixtures.
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

import torch  # noqa: E402  (allowed dep; imported after sys.path fix)

from yonder_hud import alpha  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _comfy_loadimage_mask_from_alpha(alpha_2d: np.ndarray) -> torch.Tensor:
    """Replicate ComfyUI LoadImage's MASK construction for a known alpha.

    Mirrors reference ``nodes.py`` lines 1743-1745 exactly:
        mask = alpha_channel / 255   ; mask = 1.0 - mask
    Returns a torch tensor shaped [1, H, W] like LoadImage's batched output.
    """
    a8 = np.clip(alpha_2d * 255.0 + 0.5, 0, 255).astype(np.uint8)
    m = a8.astype(np.float32) / 255.0
    m = 1.0 - torch.from_numpy(m)
    return m.unsqueeze(0)  # [1, H, W]


def _white_on_transparent(h=8, w=8):
    """A white square graphic on a transparent background.

    Returns (image_rgb [H,W,3] 0..1, true_alpha [H,W] 0..1).
    Center 4x4 is opaque white; surround is transparent (alpha 0, RGB 0 — the
    typical exported-PNG state where transparent pixels have undefined/black RGB).
    """
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    rgb[2:6, 2:6, :] = 1.0
    alpha[2:6, 2:6] = 1.0
    return rgb, alpha


# ---------------------------------------------------------------------------
# to_numpy_img
# ---------------------------------------------------------------------------

def test_to_numpy_img_torch_batched():
    """torch [1,H,W,3] 0..1 -> np float32 [H,W,3] 0..1."""
    t = torch.rand(1, 5, 7, 3)
    out = alpha.to_numpy_img(t)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (5, 7, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # values preserved
    assert np.allclose(out, t[0].numpy(), atol=1e-6)


def test_to_numpy_img_hwc_and_rgba_drop():
    """[H,W,4] keeps only RGB; [H,W] grayscale replicates to 3."""
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[..., 0] = 0.5
    rgba[..., 3] = 0.9
    out = alpha.to_numpy_img(rgba)
    assert out.shape == (4, 4, 3)
    assert np.allclose(out[..., 0], 0.5)

    gray = np.full((3, 3), 0.25, dtype=np.float32)
    outg = alpha.to_numpy_img(gray)
    assert outg.shape == (3, 3, 3)
    assert np.allclose(outg, 0.25)


def test_to_numpy_img_uint8_scaled():
    """uint8 0..255 is scaled to 0..1."""
    u8 = np.zeros((2, 2, 3), dtype=np.uint8)
    u8[..., 1] = 255
    out = alpha.to_numpy_img(u8)
    assert out.dtype == np.float32
    assert np.allclose(out[..., 1], 1.0)
    assert np.allclose(out[..., 0], 0.0)


def test_to_numpy_img_none_raises():
    """None image is a hard error."""
    raised = False
    try:
        alpha.to_numpy_img(None)
    except TypeError:
        raised = True
    assert raised


def test_to_numpy_img_large_batch_takes_first():
    """[B,H,W,C] with B>1 takes the first frame deterministically."""
    t = torch.zeros(3, 2, 2, 3)
    t[0] = 0.1
    t[1] = 0.9
    out = alpha.to_numpy_img(t)
    assert out.shape == (2, 2, 3)
    assert np.allclose(out, 0.1)


# ---------------------------------------------------------------------------
# to_numpy_mask
# ---------------------------------------------------------------------------

def test_to_numpy_mask_shapes():
    """[H,W], [1,H,W], [H,W,1] all -> [H,W]."""
    a = alpha.to_numpy_mask(torch.rand(6, 6))
    assert a.shape == (6, 6) and a.dtype == np.float32
    b = alpha.to_numpy_mask(torch.rand(1, 6, 6))
    assert b.shape == (6, 6)
    c = alpha.to_numpy_mask(np.random.rand(6, 6, 1).astype(np.float32))
    assert c.shape == (6, 6)


def test_to_numpy_mask_none_returns_none():
    """None mask -> None (unconnected socket)."""
    assert alpha.to_numpy_mask(None) is None


def test_to_numpy_mask_uint8_scaled_and_clipped():
    """uint8 mask scaled to 0..1."""
    m = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    out = alpha.to_numpy_mask(m)
    assert np.isclose(out[0, 0], 0.0)
    assert np.isclose(out[0, 1], 1.0)
    assert 0.0 <= out.min() and out.max() <= 1.0


# ---------------------------------------------------------------------------
# recover_alpha — the core inverse-alpha rule (§25.2)
# ---------------------------------------------------------------------------

def test_recover_alpha_inverse_rule_white_on_transparent():
    """The headline case: white graphic on transparent PNG.

    ComfyUI LoadImage MASK == 1 - alpha. recover_alpha must invert it back so
    the white square recovers alpha 1 and the surround recovers alpha 0.
    """
    rgb, true_alpha = _white_on_transparent()
    mask = _comfy_loadimage_mask_from_alpha(true_alpha)  # native MASK = 1 - alpha
    rec = alpha.recover_alpha(rgb, mask, invert_auto=True)
    assert rec.shape == true_alpha.shape
    assert rec.dtype == np.float32
    assert np.allclose(rec, true_alpha, atol=1e-2)
    # specifically: opaque center ~1, transparent corner ~0
    assert rec[3, 3] > 0.9
    assert rec[0, 0] < 0.1


def test_recover_alpha_partial_alpha_inverse():
    """A graded alpha round-trips through the inverse rule."""
    rgb = np.ones((4, 4, 3), dtype=np.float32)
    true_alpha = np.array(
        [[0.0, 0.25, 0.5, 1.0]] * 4, dtype=np.float32
    )
    mask = _comfy_loadimage_mask_from_alpha(true_alpha)
    rec = alpha.recover_alpha(rgb, mask, invert_auto=True)
    assert np.allclose(rec, true_alpha, atol=1e-2)


def test_recover_alpha_none_mask_luminance_fallback():
    """mask None -> luminance of the image is used as alpha."""
    rgb, _ = _white_on_transparent()
    rec = alpha.recover_alpha(rgb, None, invert_auto=True)
    expected = alpha.luminance_alpha(rgb)
    assert np.allclose(rec, expected, atol=1e-6)
    # white square -> high alpha, black surround -> 0 alpha
    assert rec[3, 3] > 0.9
    assert rec[0, 0] < 0.05


def test_recover_alpha_sentinel_64_zero_mask_falls_back():
    """LoadImage no-alpha sentinel: a 64x64 zero mask misaligned to the image.

    recover_alpha must detect the shape mismatch and use luminance, NOT invert
    the sentinel (which would give a useless full-frame alpha=1).
    """
    rgb, _ = _white_on_transparent(h=8, w=8)
    sentinel = torch.zeros((1, 64, 64), dtype=torch.float32)  # native no-alpha MASK
    rec = alpha.recover_alpha(rgb, sentinel, invert_auto=True)
    assert rec.shape == (8, 8)
    expected = alpha.luminance_alpha(rgb)
    assert np.allclose(rec, expected, atol=1e-6)
    # NOT a constant 1.0 frame
    assert rec.min() < 0.1


def test_recover_alpha_constant_mask_falls_back_to_luminance():
    """A fully-uniform aligned mask carries no shape info -> luminance fallback."""
    rgb, _ = _white_on_transparent(h=8, w=8)
    # fully transparent PNG => alpha all 0 => MASK all 1 (uniform)
    uniform = np.ones((8, 8), dtype=np.float32)
    rec = alpha.recover_alpha(rgb, uniform, invert_auto=True)
    expected = alpha.luminance_alpha(rgb)
    assert np.allclose(rec, expected, atol=1e-6)


def test_recover_alpha_invert_auto_false_uses_mask_verbatim():
    """invert_auto=False trusts the mask as a straight alpha matte."""
    rgb = np.ones((4, 4, 3), dtype=np.float32)
    straight = np.array([[0.0, 0.3, 0.7, 1.0]] * 4, dtype=np.float32)
    rec = alpha.recover_alpha(rgb, straight, invert_auto=False)
    assert np.allclose(rec, straight, atol=1e-6)


# ---------------------------------------------------------------------------
# luminance_alpha
# ---------------------------------------------------------------------------

def test_luminance_alpha_rec709():
    """Pure channels match Rec.709 luma coefficients."""
    red = np.zeros((1, 1, 3), dtype=np.float32)
    red[..., 0] = 1.0
    green = np.zeros((1, 1, 3), dtype=np.float32)
    green[..., 1] = 1.0
    blue = np.zeros((1, 1, 3), dtype=np.float32)
    blue[..., 2] = 1.0
    assert np.isclose(alpha.luminance_alpha(red)[0, 0], 0.2126, atol=1e-4)
    assert np.isclose(alpha.luminance_alpha(green)[0, 0], 0.7152, atol=1e-4)
    assert np.isclose(alpha.luminance_alpha(blue)[0, 0], 0.0722, atol=1e-4)
    white = np.ones((2, 2, 3), dtype=np.float32)
    assert np.allclose(alpha.luminance_alpha(white), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# trim_transparent
# ---------------------------------------------------------------------------

def test_trim_transparent_basic_bbox():
    """Crops to the opaque bbox in original coords (xyxy half-open)."""
    rgb, a = _white_on_transparent(h=8, w=8)  # opaque [2:6, 2:6]
    crgb, ca, bbox = alpha.trim_transparent(rgb, a, threshold=0.02)
    assert bbox == (2, 2, 6, 6)
    assert crgb.shape == (4, 4, 3)
    assert ca.shape == (4, 4)
    assert np.all(ca > 0.5)  # all cropped pixels opaque


def test_trim_transparent_fully_transparent_unchanged():
    """Fully transparent -> inputs returned unchanged, full-frame bbox."""
    rgb = np.zeros((5, 6, 3), dtype=np.float32)
    a = np.zeros((5, 6), dtype=np.float32)
    crgb, ca, bbox = alpha.trim_transparent(rgb, a, threshold=0.02)
    assert bbox == (0, 0, 6, 5)
    assert crgb.shape == (5, 6, 3)
    assert ca.shape == (5, 6)


def test_trim_transparent_single_opaque_pixel():
    """Degenerate one-pixel opaque region -> 1x1 crop."""
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    a = np.zeros((4, 4), dtype=np.float32)
    a[1, 2] = 1.0
    rgb[1, 2] = 1.0
    crgb, ca, bbox = alpha.trim_transparent(rgb, a)
    assert bbox == (2, 1, 3, 2)
    assert crgb.shape == (1, 1, 3)
    assert np.isclose(ca[0, 0], 1.0)


def test_trim_transparent_threshold_respected():
    """Pixels at/below threshold are excluded."""
    rgb = np.ones((4, 4, 3), dtype=np.float32)
    a = np.zeros((4, 4), dtype=np.float32)
    a[1:3, 1:3] = 0.5  # above default threshold 0.02
    a[0, 0] = 0.01     # below threshold -> excluded
    _, _, bbox = alpha.trim_transparent(rgb, a, threshold=0.02)
    assert bbox == (1, 1, 3, 3)


# ---------------------------------------------------------------------------
# recolor_white_asset — out_rgb = color*luma ; out_a = alpha*opacity (§8)
# ---------------------------------------------------------------------------

def test_recolor_white_asset_rule():
    """White stays full color; mid-gray gives half color; black stays black."""
    rgb = np.zeros((1, 3, 3), dtype=np.float32)
    rgb[0, 0] = 1.0   # white -> luma 1
    rgb[0, 1] = 0.5   # mid gray -> luma 0.5
    rgb[0, 2] = 0.0   # black -> luma 0
    a = np.ones((1, 3), dtype=np.float32)
    color = (1.0, 0.0, 0.0)  # red
    out = alpha.recolor_white_asset(rgb, a, color, opacity=1.0)
    assert out.shape == (1, 3, 4)
    assert out.dtype == np.float32
    # white pixel -> full red
    assert np.allclose(out[0, 0, :3], (1.0, 0.0, 0.0), atol=1e-5)
    # mid-gray -> half red
    assert np.allclose(out[0, 1, :3], (0.5, 0.0, 0.0), atol=1e-5)
    # black -> black
    assert np.allclose(out[0, 2, :3], (0.0, 0.0, 0.0), atol=1e-5)
    # alpha unchanged (opacity 1)
    assert np.allclose(out[0, :, 3], 1.0)


def test_recolor_white_asset_opacity_multiplies_alpha():
    """out_a = alpha * opacity, clamped."""
    rgb = np.ones((2, 2, 3), dtype=np.float32)
    a = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float32)
    out = alpha.recolor_white_asset(rgb, a, (0.0, 1.0, 0.0), opacity=0.5)
    expected_a = a * 0.5
    assert np.allclose(out[..., 3], expected_a, atol=1e-6)
    # green channel == luma(=1) * 1.0 = 1.0 everywhere
    assert np.allclose(out[..., 1], 1.0, atol=1e-5)


def test_recolor_white_asset_none_alpha_opaque():
    """None alpha is treated as fully opaque before opacity scaling."""
    rgb = np.ones((2, 2, 3), dtype=np.float32)
    out = alpha.recolor_white_asset(rgb, None, (0.0, 0.0, 1.0), opacity=0.8)
    assert np.allclose(out[..., 3], 0.8, atol=1e-6)
    assert np.allclose(out[..., 2], 1.0, atol=1e-5)  # blue full


def test_recolor_white_asset_color_clamped():
    """Out-of-range color components are clamped to 0..1."""
    rgb = np.ones((1, 1, 3), dtype=np.float32)
    out = alpha.recolor_white_asset(rgb, None, (2.0, -1.0, 0.5), opacity=1.0)
    assert np.isclose(out[0, 0, 0], 1.0)  # clamped from 2.0
    assert np.isclose(out[0, 0, 1], 0.0)  # clamped from -1.0
    assert np.isclose(out[0, 0, 2], 0.5)


# ---------------------------------------------------------------------------
# resize_rgba — Lanczos down / Bicubic up, straight alpha preserved
# ---------------------------------------------------------------------------

def test_resize_rgba_upscale_shape_and_alpha_preserved():
    """Upscale keeps RGBA, full-opaque alpha stays ~1 (no bleed)."""
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 1.0
    out = alpha.resize_rgba(rgba, 8, 8)
    assert out.shape == (8, 8, 4)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.allclose(out[..., 3], 1.0, atol=1e-2)


def test_resize_rgba_downscale_runs():
    """Downscale produces the requested size and valid range."""
    rgba = np.random.rand(16, 16, 4).astype(np.float32)
    out = alpha.resize_rgba(rgba, 4, 4)
    assert out.shape == (4, 4, 4)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_rgba_chooses_lanczos_vs_bicubic():
    """Verify the resampling selection matches the contract (down=Lanczos, up=Bicubic).

    We monkeypatch PIL's resize to record which resample constant is requested.
    """
    from PIL import Image

    lanczos, bicubic = alpha._pil_resampling()
    calls = []
    orig_resize = Image.Image.resize

    def spy_resize(self, size, resample=None, *args, **kwargs):
        calls.append(resample)
        # Forward exactly as received: resample positionally (so trailing
        # positional args like box/reducing_gap line up correctly).
        return orig_resize(self, size, resample, *args, **kwargs)

    # NOTE: PIL's Image.resize may internally invoke resize more than once per
    # call, so we capture the set of resample constants used per resize_rgba()
    # invocation rather than assuming exactly one underlying call.
    def resample_used(*resize_args):
        calls.clear()
        alpha.resize_rgba(*resize_args)
        used = set(calls)
        assert len(used) == 1, f"mixed resample constants in one call: {used}"
        return used.pop()

    Image.Image.resize = spy_resize
    try:
        rgba = np.zeros((10, 10, 4), dtype=np.float32)
        rgba[..., 3] = 1.0
        down = resample_used(rgba, 5, 5)     # both shrink -> Lanczos
        up = resample_used(rgba, 20, 20)     # both grow  -> Bicubic
        mixed = resample_used(rgba, 20, 5)   # one shrinks -> Lanczos (downscaling)
    finally:
        Image.Image.resize = orig_resize

    assert down == lanczos
    assert up == bicubic
    assert mixed == lanczos


def test_resize_rgba_straight_alpha_no_color_bleed():
    """Straight-alpha edge: transparent black region must not bleed into the
    opaque colored region's RGB after resize (sanity that we did NOT premultiply).

    Build a half-opaque/half-transparent tile where the transparent half has a
    different RGB; after upscale the opaque region's RGB stays its color.
    """
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    # left half: opaque pure green
    rgba[:, :2, 1] = 1.0
    rgba[:, :2, 3] = 1.0
    # right half: transparent pure red (alpha 0)
    rgba[:, 2:, 0] = 1.0
    rgba[:, 2:, 3] = 0.0
    out = alpha.resize_rgba(rgba, 4, 4)  # same size -> exact copy
    # interior opaque pixel keeps green, no red contamination
    assert out[1, 0, 1] > 0.99
    assert out[1, 0, 0] < 0.01


def test_resize_rgba_same_size_is_exact_copy():
    """No resample when dims already match; values preserved, fresh buffer."""
    rgba = np.random.rand(5, 5, 4).astype(np.float32)
    out = alpha.resize_rgba(rgba, 5, 5)
    assert out.shape == (5, 5, 4)
    assert np.allclose(out, rgba, atol=1e-6)
    assert out is not rgba  # copy, not alias


def test_resize_rgba_rgb_input_promoted():
    """[H,W,3] input is promoted to RGBA with opaque alpha."""
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 2] = 1.0
    out = alpha.resize_rgba(rgb, 8, 8)
    assert out.shape == (8, 8, 4)
    assert np.allclose(out[..., 3], 1.0, atol=1e-2)


def test_resize_rgba_uint8_input():
    """uint8 RGBA input is scaled to 0..1."""
    u8 = np.zeros((4, 4, 4), dtype=np.uint8)
    u8[..., 1] = 255
    u8[..., 3] = 255
    out = alpha.resize_rgba(u8, 2, 2)
    assert out.shape == (2, 2, 4)
    assert out[..., 1].max() <= 1.0 and out[..., 1].max() > 0.9


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
    print(f"alpha tests: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFAILURES:")
        for name, exc in failures:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
