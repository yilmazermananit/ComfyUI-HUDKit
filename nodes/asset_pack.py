"""YEA HUD Asset Pack — collapse the renderer's 16 asset sockets into ONE input.

Two ways to fill it, simplest first:
  * `folder`: a subfolder of ComfyUI's `input/` dir — every PNG is auto-loaded;
    filenames containing "target" become target points, the rest become frames
    (sorted). Zero Load Image wiring.
  * manual `frame_*` / `target_*` IMAGE+MASK pairs (optional) — override / fill slots.

Output `asset_pack` (YEA_HUD_ASSET_PACK) feeds the renderer's single `asset_pack`
input. Cloud-safe: folder reads are scoped to ComfyUI's input directory only.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("YEAHUD.asset_pack")

DISPLAY_NAME = "YEA HUD Asset Pack"
NODE_ID = "YEAHUDAssetPack"
CATEGORY = "YEA/HUD"
ASSET_PACK_TYPE = "YEA_HUD_ASSET_PACK"


def _load_pair_from_file(path):
    """ComfyUI Load Image convention: (IMAGE [1,H,W,3], MASK [1,H,W] = 1 - alpha)."""
    import numpy as np
    import torch
    from PIL import Image
    img = Image.open(path)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.asarray(img.convert("RGBA"), dtype=np.float32)[:, :, 3] / 255.0
    return torch.from_numpy(rgb)[None], torch.from_numpy(1.0 - alpha)[None]


def _scan_folder(folder):
    """Return (frame_pairs, target_pairs) loaded from <ComfyUI input>/<folder>.
    Cloud-safe: never escapes the input directory. Returns ([],[]) if unavailable."""
    if not folder:
        return [], []
    try:
        import folder_paths
        base = folder_paths.get_input_directory()
    except Exception:
        return [], []
    safe = os.path.normpath(folder).lstrip("/").replace("..", "")
    root = os.path.join(base, safe)
    if not os.path.isdir(root):
        logger.warning("YEA Asset Pack: folder %r not found under input dir.", folder)
        return [], []
    frames, targets = [], []
    for name in sorted(os.listdir(root)):
        if not name.lower().endswith((".png", ".webp")):
            continue
        try:
            pair = _load_pair_from_file(os.path.join(root, name))
        except Exception as exc:  # noqa: BLE001 — a bad file must not break the graph
            logger.warning("YEA Asset Pack: skipped %s (%r)", name, exc)
            continue
        (targets if "target" in name.lower() else frames).append(pair)
    return frames[:6], targets[:2]


def _dim_if_solid(pair, solid_opacity, threshold=0.35):
    """Auto-translucency: a SOLID frame asset (high alpha coverage, e.g. a filled block)
    is dimmed to ``solid_opacity`` so it doesn't bury the footage; line-art / bracket /
    outline designs (low coverage) are left fully opaque. Returns a (image, mask) pair."""
    if pair is None or solid_opacity >= 0.999:
        return pair
    import torch
    img, mask = pair
    if mask is None:
        return pair
    alpha = 1.0 - mask.float()
    coverage = float((alpha > 0.5).float().mean())
    if coverage <= threshold:
        return pair
    new_mask = 1.0 - (alpha * float(solid_opacity))
    return img, new_mask


def run_asset_pack(folder="", manual=None, solid_opacity=0.45):
    """Build the asset_pack dict. ``manual`` maps slot name -> tensor (frame_1_image …).
    Solid frame assets are auto-dimmed to ``solid_opacity`` (1.0 disables)."""
    manual = manual or {}
    frame_pairs, target_pairs = [None] * 6, [None] * 2
    for i in range(6):
        im, mk = manual.get(f"frame_{i+1}_image"), manual.get(f"frame_{i+1}_mask")
        if im is not None:
            frame_pairs[i] = (im, mk)
    for i in range(2):
        im, mk = manual.get(f"target_{i+1}_image"), manual.get(f"target_{i+1}_mask")
        if im is not None:
            target_pairs[i] = (im, mk)
    # folder-loaded assets fill the still-empty slots.
    f_loaded, t_loaded = _scan_folder(folder)
    fi = iter(f_loaded)
    for i in range(6):
        if frame_pairs[i] is None:
            frame_pairs[i] = next(fi, None)
    ti = iter(t_loaded)
    for i in range(2):
        if target_pairs[i] is None:
            target_pairs[i] = next(ti, None)
    frame_pairs = [_dim_if_solid(p, solid_opacity) for p in frame_pairs]
    n_f = sum(p is not None for p in frame_pairs)
    n_t = sum(p is not None for p in target_pairs)
    logger.info("YEA HUD Asset Pack: %d frame + %d target assets (solid_opacity=%.2f).",
                n_f, n_t, solid_opacity)
    return {"frame_pairs": frame_pairs, "target_pairs": target_pairs,
            "n_frame_assets": n_f, "n_target_assets": n_t}


try:
    from comfy_api.latest import io

    YEAAssetPack = io.Custom(ASSET_PACK_TYPE)

    class YEAHUDAssetPack(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            ins = [io.String.Input("folder", default="", optional=True,
                                   tooltip="Subfolder of ComfyUI's input/ dir. PNGs auto-load; "
                                           "names with 'target' → target points, rest → frames."),
                   io.Float.Input("solid_opacity", default=0.45, min=0.0, max=1.0, step=0.05, optional=True,
                                  tooltip="Opacity for SOLID/filled frame assets (auto-detected). "
                                          "Brackets/outlines stay fully opaque. 1.0 disables.")]
            for i in range(1, 7):
                ins.append(io.Image.Input(f"frame_{i}_image", optional=True))
                ins.append(io.Mask.Input(f"frame_{i}_mask", optional=True))
            for i in range(1, 3):
                ins.append(io.Image.Input(f"target_{i}_image", optional=True))
                ins.append(io.Mask.Input(f"target_{i}_mask", optional=True))
            return io.Schema(
                node_id=NODE_ID, display_name=DISPLAY_NAME, category=CATEGORY,
                description=("Bundle up to 6 frame + 2 target white-on-transparent PNG assets into a "
                             "single asset_pack for the renderer. Auto-load from an input subfolder, "
                             "or wire Load Image pairs manually."),
                search_aliases=["asset", "pack", "hud", "frame", "target"],
                inputs=ins,
                outputs=[YEAAssetPack.Output("asset_pack", display_name="asset_pack")],
            )

        @classmethod
        def execute(cls, folder="", solid_opacity=0.45, **manual) -> io.NodeOutput:
            return io.NodeOutput(run_asset_pack(folder, manual, solid_opacity))

    NODE = YEAHUDAssetPack
except ImportError:  # pragma: no cover
    NODE = None
