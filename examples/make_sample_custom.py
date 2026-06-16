"""Render the sample using the USER's real HUD assets (~/Desktop/TARGET/*.png).

Loads the 6 Frame PNGs + 2 TARGET POINT PNGs exactly as ComfyUI's Load Image would
(IMAGE = RGB, MASK = 1 - alpha), feeds them to the renderer as frame_/target_ inputs
with Tektur + category colors, and writes a custom-asset sample + a dark-bg asset
preview so the white-on-transparent designs are actually visible.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import av
from PIL import Image

from nodes.hud_renderer import run_hud_renderer
import make_sample as ms  # scene builder (_build, B, H, W, FPS)

# Prefer the assets bundled in the repo; fall back to the original design folder.
TARGET = os.path.join(_HERE, "assets")
if not os.path.isdir(TARGET):
    TARGET = os.path.expanduser("~/Desktop/TARGET")
import glob as _glob
_allp = sorted(_glob.glob(os.path.join(TARGET, "*.png")))
FRAME_FILES = [os.path.basename(p) for p in _allp if "target" not in os.path.basename(p).lower()][:6]
TARGET_FILES = [os.path.basename(p) for p in _allp if "target" in os.path.basename(p).lower()][:2]
PALETTE = [(1.0, 0.16, 0.10), (0.13, 0.36, 1.0), (0.55, 1.0, 0.16), (1.0, 0.69, 0.87)]


def load_pair(path):
    """Replicate ComfyUI Load Image: returns (IMAGE [1,H,W,3], MASK [1,H,W] = 1-alpha)."""
    img = Image.open(path)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.asarray(img.convert("RGBA"), dtype=np.float32)[:, :, 3] / 255.0
    mask = 1.0 - alpha
    return torch.from_numpy(rgb)[None], torch.from_numpy(mask)[None]


def _slots(files):
    imgs, msks = [], []
    for fn in files:
        p = os.path.join(TARGET, fn)
        if os.path.exists(p):
            im, mk = load_pair(p)
            imgs.append(im); msks.append(mk)
        else:
            imgs.append(None); msks.append(None)
    return imgs, msks


def _assets_preview():
    """Recolor each design (white→palette) over a dark bg in a grid, so it's visible."""
    cell, cols = 260, 4
    files = FRAME_FILES + TARGET_FILES
    rows = (len(files) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 0.11, dtype=np.float32)
    for i, fn in enumerate(files):
        p = os.path.join(TARGET, fn)
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGBA").resize((cell - 24, cell - 24))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        lum = arr[:, :, :3].mean(-1)
        col = PALETTE[i % len(PALETTE)]
        out_rgb = np.stack([col[0] * lum, col[1] * lum, col[2] * lum], -1)
        a3 = arr[:, :, 3:4]
        r, c = divmod(i, cols)
        y, x = r * cell + 12, c * cell + 12
        h, w = out_rgb.shape[:2]
        sheet[y:y + h, x:x + w] = out_rgb * a3 + sheet[y:y + h, x:x + w] * (1 - a3)
    Image.fromarray((sheet * 255).round().astype("uint8"), "RGB").save(
        os.path.join(_ROOT, "docs", "assets_preview.png"))


def main():
    frames, track_data, annotations = ms._build()
    fimg, fmsk = _slots(FRAME_FILES)
    timg, tmsk = _slots(TARGET_FILES)

    images, _dbg, tj = run_hud_renderer(
        frames, track_data=track_data, annotations_json=json.dumps(annotations),
        seed=77, style_mode="assets_only", max_objects=24, display_probability=1.0,
        padding_x=10, padding_y=10, smoothing=0.55, track_iou_threshold=0.25, track_memory_frames=6,
        frame_opacity=1.0, fill_opacity=0.0, label_opacity=0.92, line_width=2, target_scale=0.30,
        target_probability=0.0, detail_density=0.0,
        caption_mode="detailed", label_length_mode="adaptive", show_child_annotations=False,
        show_action_label=True, color_mode="category_based", primary_group_label="PERSON",
        frame_images=fimg, frame_masks=fmsk, target_images=timg, target_masks=tmsk)
    print("tracking:", tj)

    out = os.path.join(_HERE, "sample_custom_output.mp4")
    with av.open(out, "w") as c:
        st = c.add_stream("h264", rate=Fraction(round(ms.FPS * 1000), 1000))
        st.width, st.height, st.pix_fmt = ms.W, ms.H, "yuv420p"
        for f in range(images.shape[0]):
            arr = (images[f].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            c.mux(st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24").reformat(format="yuv420p")))
        c.mux(st.encode(None))
    Image.fromarray((images[ms.B // 2].clamp(0, 1) * 255).round().to(torch.uint8).numpy(), "RGB").save(
        os.path.join(_ROOT, "docs", "sample_custom_after.png"))
    _assets_preview()
    print(f"wrote {out}, docs/sample_custom_after.png, docs/assets_preview.png")


if __name__ == "__main__":
    main()
