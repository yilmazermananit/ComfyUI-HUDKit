"""Unit tests for ``yea_hud.assets`` (brief §16 / §25 asset cases).

Runnable BOTH ways:

* pytest:   /Users/differenterman/Documents/ComfyUI/.venv/bin/python -m pytest tests/test_assets.py -q
* script:   /Users/differenterman/Documents/ComfyUI/.venv/bin/python tests/test_assets.py
            (runs every test, prints a summary, ``sys.exit(1)`` on any failure)

No ComfyUI, no network, no model files. Tiny synthetic numpy/torch fixtures only.
The MASK fixtures replicate ComfyUI ``LoadImage``'s inverse-alpha rule
(``mask = 1 - alpha``) verbatim so the alpha-recovery path is exercised against
the real native convention (see ``yea_hud.alpha`` module docstring).
"""

from __future__ import annotations

import os
import sys

# Make the package importable regardless of CWD when run as a bare script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from yea_hud import assets as A  # noqa: E402
from yea_hud import alpha as alpha_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _loadimage_mask(alpha_2d: np.ndarray) -> torch.Tensor:
    """Build a ComfyUI LoadImage MASK from a known straight alpha.

    Mirrors reference ``nodes.py`` LoadImage: ``mask = 1.0 - (alpha/255)``.
    Returns a torch tensor shaped ``[1, H, W]`` (LoadImage's batched output).
    """
    a8 = np.clip(alpha_2d * 255.0 + 0.5, 0, 255).astype(np.uint8)
    m = a8.astype(np.float32) / 255.0
    m = 1.0 - torch.from_numpy(m)
    return m.unsqueeze(0)


def _white_square_png(h: int = 16, w: int = 16, c0: int = 4, c1: int = 12):
    """A white square graphic on a transparent background.

    Returns ``(image_torch [1,H,W,3] 0..1, mask_torch [1,H,W] loadimage-inverse)``.
    Center ``[c0:c1, c0:c1]`` is opaque white; surround is transparent (alpha 0).
    Transparent RGB is left black (the typical exported-PNG state).
    """
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    a = np.zeros((h, w), dtype=np.float32)
    rgb[c0:c1, c0:c1, :] = 1.0
    a[c0:c1, c0:c1] = 1.0
    img = torch.from_numpy(rgb).unsqueeze(0)  # [1,H,W,3]
    mask = _loadimage_mask(a)                 # [1,H,W]
    return img, mask


def _gray_image_no_alpha(h: int = 16, w: int = 16, value: float = 0.5):
    """A flat-gray image with NO mask (image-only slot). Returns torch [1,H,W,3]."""
    rgb = np.full((h, w, 3), float(value), dtype=np.float32)
    return torch.from_numpy(rgb).unsqueeze(0)


def _bank(frame_pairs, target_pairs=None, *, trim=False, fit="stretch", cache=64):
    """Construct an AssetBank with sane test defaults."""
    return A.AssetBank(
        frame_pairs,
        target_pairs if target_pairs is not None else [],
        trim=trim,
        fit_frame=fit,
        fit_target=fit,
        max_cache_items=cache,
    )


RED = (1.0, 0.0, 0.0)
BLUE = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# snap_bucket / color_to_cache_id helpers
# ---------------------------------------------------------------------------

def test_snap_bucket_floor_grid_with_floor():
    assert A.snap_bucket(8) == 8
    assert A.snap_bucket(15) == 8
    assert A.snap_bucket(16) == 16
    assert A.snap_bucket(23) == 16
    assert A.snap_bucket(64) == 64
    assert A.snap_bucket(100) == 96


def test_snap_bucket_degenerate_inputs_floor_to_one_bucket():
    # zero, sub-bucket and negative all floor to a single 8px bucket (never 0).
    assert A.snap_bucket(0) == 8
    assert A.snap_bucket(1) == 8
    assert A.snap_bucket(7) == 8
    assert A.snap_bucket(-4) == 8


def test_color_to_cache_id_distinct_and_stable():
    # Distinct colors -> distinct ids; same color -> same id.
    assert A.color_to_cache_id(RED) != A.color_to_cache_id(BLUE)
    assert A.color_to_cache_id(RED) == A.color_to_cache_id((1.0, 0.0, 0.0))
    # 8-bit quantisation: a sub-1/255 nudge collides, a >=1/255 change does not.
    assert A.color_to_cache_id((0.5, 0.5, 0.5)) == A.color_to_cache_id((0.5 + 1e-4, 0.5, 0.5))
    assert A.color_to_cache_id((0.5, 0.5, 0.5)) != A.color_to_cache_id((0.5 + 2.0 / 255.0, 0.5, 0.5))


def test_color_to_cache_id_clamps_and_handles_nan():
    # Out-of-range / non-finite components never raise and fold to valid bytes.
    assert A.color_to_cache_id((2.0, -1.0, 0.0)) == A.color_to_cache_id((1.0, 0.0, 0.0))
    assert A.color_to_cache_id((float("nan"), 0.0, 0.0)) == A.color_to_cache_id((0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Eligibility: building compact connected-slot index lists (brief §16)
# ---------------------------------------------------------------------------

def test_eligible_counts_skip_unconnected_slots():
    # Slots: connected, None (gap), connected, (None,None) (gap).
    img, mask = _white_square_png()
    bank = _bank(
        frame_pairs=[(img, mask), None, (img, mask), (None, None)],
        target_pairs=[None, (img, mask)],
    )
    assert bank.n_frame_assets == 2   # the two real (img,mask) slots
    assert bank.n_target_assets == 1  # only slot index 1
    assert len(bank) == 3


def test_eligible_empty_and_none_lists():
    # Empty list and None both yield zero eligible assets (no crash).
    b1 = _bank(frame_pairs=[], target_pairs=[])
    assert b1.n_frame_assets == 0 and b1.n_target_assets == 0
    b2 = _bank(frame_pairs=None, target_pairs=None)
    assert b2.n_frame_assets == 0 and b2.n_target_assets == 0
    # All-None slots -> still zero eligible.
    b3 = _bank(frame_pairs=[None, None, (None, None)])
    assert b3.n_frame_assets == 0


def test_mask_only_slot_is_eligible_but_renders_none():
    # Brief §16: mask connected, image not -> occupies an eligible index but
    # frame_rgba returns None (asset NOT rendered).
    _img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(None, mask)])
    assert bank.n_frame_assets == 1
    out = bank.frame_rgba(0, 32, 32, RED, 1.0)
    assert out is None


def test_eligible_index_is_compact_over_gaps():
    # eligible_id 0 maps to slot 0, eligible_id 1 maps to slot 2 (slot 1 is a gap).
    # Distinguish the two connected slots structurally: slot 0 is a centered white
    # square WITH a mask (transparent corners); slot 2 is a fully-white image-only
    # slot (luminance fallback -> opaque everywhere).
    img_a, mask_a = _white_square_png()
    rgb_b = np.ones((16, 16, 3), dtype=np.float32)
    img_b = torch.from_numpy(rgb_b).unsqueeze(0)   # full white, image-only (no mask)
    bank = _bank(frame_pairs=[(img_a, mask_a), None, img_b])
    assert bank.n_frame_assets == 2
    out0 = bank.frame_rgba(0, 16, 16, RED, 1.0)  # the masked white square
    out1 = bank.frame_rgba(1, 16, 16, RED, 1.0)  # the full-white image-only block
    assert out0 is not None and out1 is not None
    # out0 has fully-transparent corners (square mask); out1 is opaque everywhere
    # (luminance of white == 1).
    assert out0[0, 0, 3] < 0.01
    assert out1[0, 0, 3] > 0.99


# ---------------------------------------------------------------------------
# frame_rgba / target_rgba: render shape, range, recolor (brief §8)
# ---------------------------------------------------------------------------

def test_frame_rgba_shape_and_range():
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)])
    out = bank.frame_rgba(0, 40, 24, RED, 1.0)
    assert out is not None
    assert out.dtype == np.float32
    # sizes are the 8px-snapped buckets (40 -> 40, 24 -> 24).
    assert out.shape == (24, 40, 4)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_frame_rgba_recolor_white_to_palette_color():
    # White graphic -> recolored to RED: opaque region's RGB == (1,0,0).
    img, mask = _white_square_png(h=16, w=16, c0=4, c1=12)
    bank = _bank(frame_pairs=[(img, mask)])
    # Use a snapped size equal to source so the center stays solid after resize.
    out = bank.frame_rgba(0, 16, 16, RED, 1.0)
    assert out is not None
    # The center pixel is inside the (recolored) white square: red, fully opaque.
    cy, cx = 8, 8
    assert out[cy, cx, 0] > 0.9
    assert out[cy, cx, 1] < 0.05
    assert out[cy, cx, 2] < 0.05
    assert out[cy, cx, 3] > 0.9


def test_frame_rgba_opacity_scales_alpha():
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)])
    full = bank.frame_rgba(0, 16, 16, RED, 1.0)
    half = bank.frame_rgba(0, 16, 16, RED, 0.5)
    assert full is not None and half is not None
    cy, cx = 8, 8
    # Opaque-region alpha halves with opacity; RGB (straight) unchanged.
    assert abs(full[cy, cx, 3] - 1.0) < 0.05
    assert abs(half[cy, cx, 3] - 0.5) < 0.05
    assert np.allclose(full[cy, cx, :3], half[cy, cx, :3], atol=1e-4)


def test_target_rgba_independent_of_frame_slots():
    # target_rgba reads the target list, not the frame list.
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[None], target_pairs=[(img, mask)])
    assert bank.n_frame_assets == 0
    assert bank.n_target_assets == 1
    assert bank.frame_rgba(0, 16, 16, RED, 1.0) is None       # frame slot is a gap
    assert bank.target_rgba(0, 16, 16, RED, 1.0) is not None  # target slot exists


# ---------------------------------------------------------------------------
# image-without-mask -> luminance fallback (brief §16)
# ---------------------------------------------------------------------------

def test_image_only_uses_luminance_alpha_fallback():
    # A flat mid-gray image with NO mask: alpha = luminance(image) (constant ~0.5),
    # so the recolored RGBA is uniformly semi-opaque (NOT all-zero, NOT all-one).
    img = _gray_image_no_alpha(value=0.5)
    bank = _bank(frame_pairs=[img])  # bare image object -> image-only slot
    assert bank.n_frame_assets == 1
    out = bank.frame_rgba(0, 16, 16, RED, 1.0)
    assert out is not None
    expected_a = float(alpha_mod.luminance_alpha(np.full((16, 16, 3), 0.5, np.float32)).mean())
    assert abs(float(out[..., 3].mean()) - expected_a) < 0.02
    # And the luminance fallback (not the all-1 sentinel) really kicked in:
    assert 0.2 < float(out[..., 3].mean()) < 0.8


def test_image_with_constant_mask_falls_back_to_luminance():
    # A degenerate/constant MASK (no spatial info) -> alpha.recover_alpha falls back
    # to luminance. Pair an opaque-gray image with an all-zero (loadimage "no alpha"
    # style) constant mask of matching size.
    img = _gray_image_no_alpha(value=0.5)
    const_mask = torch.zeros(1, 16, 16, dtype=torch.float32)  # constant -> fallback
    bank = _bank(frame_pairs=[(img, const_mask)])
    out = bank.frame_rgba(0, 16, 16, RED, 1.0)
    assert out is not None
    # luminance of 0.5 gray ~ 0.5 -> semi-opaque, not full.
    assert 0.2 < float(out[..., 3].mean()) < 0.8


# ---------------------------------------------------------------------------
# trim behaviour (brief §8)
# ---------------------------------------------------------------------------

def test_trim_crops_transparent_margin_before_resize():
    # With trim=True the centered white square fills the whole output box (its
    # transparent margin is cropped away), so EVERY output pixel is opaque.
    img, mask = _white_square_png(h=16, w=16, c0=4, c1=12)
    bank_trim = _bank(frame_pairs=[(img, mask)], trim=True)
    out_t = bank_trim.frame_rgba(0, 16, 16, RED, 1.0)
    assert out_t is not None
    assert float(out_t[..., 3].min()) > 0.9   # fully opaque everywhere after trim

    # Without trim, the corners stay transparent (margin preserved).
    bank_notrim = _bank(frame_pairs=[(img, mask)], trim=False)
    out_n = bank_notrim.frame_rgba(0, 16, 16, RED, 1.0)
    assert out_n is not None
    assert float(out_n[0, 0, 3]) < 0.1


# ---------------------------------------------------------------------------
# Caching + 8px bucketing
# ---------------------------------------------------------------------------

def test_cache_hit_returns_equal_but_distinct_array():
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)])
    a = bank.frame_rgba(0, 32, 32, RED, 1.0)
    b = bank.frame_rgba(0, 32, 32, RED, 1.0)
    assert a is not None and b is not None
    # Equal content...
    assert np.array_equal(a, b)
    # ...but distinct buffers (caller may mutate the returned copy safely).
    assert a is not b
    a[0, 0, 0] = 0.123
    c = bank.frame_rgba(0, 32, 32, RED, 1.0)
    assert c[0, 0, 0] != 0.123  # cache was not corrupted by mutating a returned copy


def test_sub_bucket_sizes_share_one_cache_entry():
    # Sizes 33..39 all snap to bucket 32 -> one cache entry, identical output shape.
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)], cache=8)
    out33 = bank.frame_rgba(0, 33, 33, RED, 1.0)
    out39 = bank.frame_rgba(0, 39, 39, RED, 1.0)
    assert out33 is not None and out39 is not None
    assert out33.shape == (32, 32, 4) == out39.shape
    assert np.array_equal(out33, out39)


def test_different_color_and_size_are_separate_cache_keys():
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)])
    red = bank.frame_rgba(0, 16, 16, RED, 1.0)
    blue = bank.frame_rgba(0, 16, 16, BLUE, 1.0)
    big = bank.frame_rgba(0, 64, 64, RED, 1.0)
    assert red is not None and blue is not None and big is not None
    # Different color -> different opaque RGB.
    cy, cx = 8, 8
    assert red[cy, cx, 0] > 0.9 and red[cy, cx, 2] < 0.1
    assert blue[cy, cx, 2] > 0.9 and blue[cy, cx, 0] < 0.1
    # Different size -> different shape.
    assert big.shape == (64, 64, 4)


# ---------------------------------------------------------------------------
# Out-of-range / degenerate ids
# ---------------------------------------------------------------------------

def test_out_of_range_and_negative_ids_return_none():
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)])
    assert bank.frame_rgba(1, 16, 16, RED, 1.0) is None    # only id 0 exists
    assert bank.frame_rgba(-1, 16, 16, RED, 1.0) is None
    assert bank.frame_rgba(99, 16, 16, RED, 1.0) is None
    assert bank.frame_rgba(None, 16, 16, RED, 1.0) is None  # type: ignore[arg-type]


def test_no_eligible_assets_any_id_returns_none():
    bank = _bank(frame_pairs=[None, None])
    assert bank.n_frame_assets == 0
    assert bank.frame_rgba(0, 16, 16, RED, 1.0) is None


# ---------------------------------------------------------------------------
# Determinism (two banks from the same pairs behave identically)
# ---------------------------------------------------------------------------

def test_determinism_across_two_banks():
    img, mask = _white_square_png()
    b1 = _bank(frame_pairs=[(img, mask)])
    b2 = _bank(frame_pairs=[(img, mask)])
    o1 = b1.frame_rgba(0, 24, 24, RED, 0.8)
    o2 = b2.frame_rgba(0, 24, 24, RED, 0.8)
    assert o1 is not None and o2 is not None
    assert np.array_equal(o1, o2)


# ---------------------------------------------------------------------------
# Constructor option handling
# ---------------------------------------------------------------------------

def test_fit_mode_coercion_and_unknown_defaults_to_stretch():
    img, mask = _white_square_png()
    bank = A.AssetBank(
        [(img, mask)], [(img, mask)],
        trim=True, fit_frame="contain", fit_target="bananas",
        max_cache_items=16,
    )
    assert bank.fit_frame == "contain"
    assert bank.fit_target == "stretch"  # unknown -> coerced
    assert bank.trim is True


def test_clear_cache_recomputes_identically():
    img, mask = _white_square_png()
    bank = _bank(frame_pairs=[(img, mask)])
    a = bank.frame_rgba(0, 16, 16, RED, 1.0)
    bank.clear_cache()
    b = bank.frame_rgba(0, 16, 16, RED, 1.0)
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_numpy_pair_inputs_supported():
    # Pairs may carry numpy arrays (not only torch); the bank handles both.
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    rgb[4:12, 4:12, :] = 1.0
    a = np.zeros((16, 16), dtype=np.float32)
    a[4:12, 4:12] = 1.0
    mask = (1.0 - a).astype(np.float32)  # loadimage-style inverse, numpy [H,W]
    bank = _bank(frame_pairs=[(rgb, mask)])
    out = bank.frame_rgba(0, 16, 16, RED, 1.0)
    assert out is not None
    assert out[8, 8, 0] > 0.9 and out[8, 8, 3] > 0.9


# ---------------------------------------------------------------------------
# Aspect-aware frame selection (pick the design that matches the box proportion)
# ---------------------------------------------------------------------------

def _white_rect_png(content_w: int, content_h: int, pad: int = 4):
    """A solid white rect of (content_h x content_w) on a transparent canvas.

    With ``trim=True`` the bank measures aspect = content_w / content_h, so this
    lets a test pin an asset's *content* aspect exactly regardless of canvas pad.
    """
    H = content_h + 2 * pad
    W = content_w + 2 * pad
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    a = np.zeros((H, W), dtype=np.float32)
    rgb[pad:pad + content_h, pad:pad + content_w, :] = 1.0
    a[pad:pad + content_h, pad:pad + content_w] = 1.0
    return torch.from_numpy(rgb).unsqueeze(0), _loadimage_mask(a)


def test_frame_aspect_measures_trimmed_content():
    # Three designs with KNOWN content aspect: portrait 0.33, square 1.0, wide 4.0.
    bank = _bank(
        [_white_rect_png(20, 60), _white_rect_png(40, 40), _white_rect_png(80, 20)],
        trim=True,
    )
    assert abs(bank.frame_aspect(0) - (20.0 / 60.0)) < 0.06   # portrait
    assert abs(bank.frame_aspect(1) - 1.0) < 0.06             # square
    assert abs(bank.frame_aspect(2) - (80.0 / 20.0)) < 0.2    # wide
    # Out-of-range -> None (never raises).
    assert bank.frame_aspect(9) is None
    assert bank.frame_aspect(-1) is None


def test_frame_asset_for_aspect_picks_closest_design():
    # portrait(0)=0.33, square(1)=1.0, wide(2)=4.0
    bank = _bank(
        [_white_rect_png(20, 60), _white_rect_png(40, 40), _white_rect_png(80, 20)],
        trim=True,
    )
    assert bank.frame_asset_for_aspect(0.30) == 0   # tall box -> portrait design
    assert bank.frame_asset_for_aspect(0.95) == 1   # square-ish box -> square design
    assert bank.frame_asset_for_aspect(1.15) == 1   # near-square stays square
    assert bank.frame_asset_for_aspect(3.6) == 2    # wide box -> wide design
    # Log-symmetric: 2:1 and 1:2 are each "one octave" from square -> equidistant,
    # so a 2.0 box is closer to the wide design than to square.
    assert bank.frame_asset_for_aspect(2.2) == 2


def test_frame_asset_for_aspect_single_and_bad_inputs_fall_back():
    one = _bank([_white_rect_png(40, 40)], trim=True)
    assert one.frame_asset_for_aspect(5.0) == 0          # only one design -> id 0
    bank = _bank(
        [_white_rect_png(20, 60), _white_rect_png(80, 20)], trim=True
    )
    # Non-finite / non-positive aspects fall back to the provided id, never raise.
    assert bank.frame_asset_for_aspect(float("nan"), fallback_id=1) == 1
    assert bank.frame_asset_for_aspect(0.0, fallback_id=0) == 0
    assert bank.frame_asset_for_aspect(-3.0, fallback_id=1) == 1


# ---------------------------------------------------------------------------
# Script runner (mirrors the other test modules)
# ---------------------------------------------------------------------------

def _run_all() -> int:
    """Run every ``test_*`` in this module; print a summary; return failure count."""
    g = dict(globals())
    tests = sorted((n, f) for n, f in g.items() if n.startswith("test_") and callable(f))
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 — test harness reports all errors
            failures += 1
            import traceback
            print(f"FAIL {name}: {exc!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed; {failures} failed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
