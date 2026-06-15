"""Render a synthetic dense-HUD sample video (no SAM3 weights needed).

Builds a small synthetic clip with moving "subjects", a native-format
SAM3_TRACK_DATA (bit-packed masks), and a YONDER_ANNOTATIONS set with detailed
labels + child regions (face / footwear) + a frame annotation (ball), then runs the
real Yonder HUD Renderer and encodes the result to MP4 with PyAV. Also saves a
before/after still for the README.

Run:
  /path/to/.venv/bin/python examples/make_sample.py
Outputs: examples/sample_output.mp4, docs/sample_before.png, docs/sample_after.png
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from nodes.hud_renderer import run_hud_renderer

B, H, W = 60, 360, 640
HM, WM = 180, 320  # mask res (W%8==0)
FPS = 24.0


def _bg():
    """Dark editorial gradient + faint grid, [B,H,W,3]."""
    torch.manual_seed(7)
    yy = torch.linspace(0.05, 0.16, H).view(1, H, 1, 1)
    xx = torch.linspace(0.04, 0.10, W).view(1, 1, W, 1)
    base = (yy + xx).expand(B, H, W, 3).clone()
    base += 0.015 * torch.rand(B, H, W, 3)
    base[:, ::40, :, :] += 0.02
    base[:, :, ::40, :] += 0.02
    return base.clamp(0, 1)


# 4 subjects: (w,h, x0, y0, vx, vy, body_color)
SUBJECTS = [
    (70, 200, 60, 120, 1.4, 0.0, (0.30, 0.26, 0.24)),   # big foreground person
    (50, 150, 300, 150, -0.8, 0.2, (0.24, 0.24, 0.28)),  # medium
    (44, 130, 460, 160, 0.6, -0.1, (0.26, 0.22, 0.22)),  # medium
    (26, 70, 200, 240, 1.0, 0.0, (0.22, 0.24, 0.22)),    # small background
]


def _pack(masks_bool):
    shifts = torch.arange(8)
    return (masks_bool.view(*masks_bool.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def _build():
    frames = _bg()
    sx = W / WM
    sy = H / HM
    packed_frames = []
    for f in range(B):
        objs = torch.zeros(len(SUBJECTS), HM, WM, dtype=torch.bool)
        for i, (w, h, x0, y0, vx, vy, col) in enumerate(SUBJECTS):
            x = int(x0 + vx * f) % (W - w)
            y = int(y0 + vy * f) % (H - h)
            # paint the body onto the frame
            frames[f, y:y + h, x:x + w, :] = torch.tensor(col)
            # mask in mask-res
            mx, my, mw, mh = int(x / sx), int(y / sy), max(1, int(w / sx)), max(1, int(h / sy))
            objs[i, my:my + mh, mx:mx + mw] = True
        packed_frames.append(_pack(objs))
    track_data = {"packed_masks": torch.stack(packed_frames), "n_frames": B,
                  "scores": [0.95, 0.88, 0.83, 0.71], "orig_size": (H, W)}
    annotations = {
        "version": 1,
        "video": {"width": W, "height": H, "fps": FPS, "frame_count": B},
        "tracks": [
            {"track_id": "person:0", "category": "person", "short_label": "PERSON",
             "detailed_label": "SHIRTLESS MAN WITH TATTOOS",
             "action_label": "KICKING SOCCER BALL",
             "display_label": "SHIRTLESS MAN WITH TATTOOS KICKING SOCCER BALL",
             "confidence": 0.95, "style_overrides": {"priority": 100},
             "children": [
                 {"annotation_id": "person:0:face", "category": "human_face", "short_label": "HUMAN FACE",
                  "bbox_by_frame": {str(f): [60 + int(1.4 * f) + 14, 124, 60 + int(1.4 * f) + 56, 168]
                                    for f in range(0, B, 4)}},
                 {"annotation_id": "person:0:foot", "category": "footwear", "short_label": "FOOTWEAR",
                  "bbox_by_frame": {str(f): [60 + int(1.4 * f) + 6, 290, 60 + int(1.4 * f) + 60, 320]
                                    for f in range(0, B, 4)}},
             ]},
            {"track_id": "person:1", "category": "person", "short_label": "PERSON",
             "detailed_label": "PERSON IN DARK JACKET", "display_label": "PERSON IN DARK JACKET",
             "confidence": 0.88, "style_overrides": {"priority": 80}},
            {"track_id": "person:2", "category": "person", "short_label": "PERSON",
             "detailed_label": "PERSON WALKING", "display_label": "PERSON WALKING",
             "confidence": 0.83, "style_overrides": {"priority": 70}},
            {"track_id": "person:3", "category": "person", "short_label": "PERSON",
             "display_label": "PERSON", "confidence": 0.71, "style_overrides": {"priority": 40}},
        ],
        "frame_annotations": {str(f): [{"annotation_id": "ball:0", "category": "soccer_ball",
                                        "short_label": "SOCCER BALL",
                                        "bbox": [40 + int(1.4 * f), 300, 70 + int(1.4 * f), 330],
                                        "priority": 75}] for f in range(B)},
    }
    return frames, track_data, annotations


def _encode_mp4(images, path):
    with av.open(path, mode="w") as container:
        stream = container.add_stream("h264", rate=Fraction(round(FPS * 1000), 1000))
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        for f in range(images.shape[0]):
            arr = (images[f].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            vframe = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            container.mux(stream.encode(vframe.reformat(format="yuv420p")))
        container.mux(stream.encode(None))


def _save_png(frame_hwc, path):
    from PIL import Image
    arr = (frame_hwc.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(arr, mode="RGB").save(path)


def main():
    import av as _av  # noqa
    global av
    av = _av
    frames, track_data, annotations = _build()
    images, _dbg, tj = run_hud_renderer(
        frames, seed=77, style_mode="mixed", max_objects=24, display_probability=1.0,
        padding_x=8, padding_y=8, smoothing=0.55, track_iou_threshold=0.25, track_memory_frames=6,
        frame_opacity=1.0, fill_opacity=0.28, label_opacity=0.92, line_width=2, target_scale=0.16,
        track_data=track_data, annotations_json=json.dumps(annotations), caption_mode="detailed",
        label_length_mode="adaptive", label_collision_mode="reposition_and_shorten",
        show_child_annotations=True, show_action_label=True, color_mode="category_based",
        primary_group_label="PERSON")
    print("tracking:", tj)
    out_mp4 = os.path.join(_HERE, "sample_output.mp4")
    _encode_mp4(images, out_mp4)
    docs = os.path.join(_ROOT, "docs")
    _save_png(frames[B // 2], os.path.join(docs, "sample_before.png"))
    _save_png(images[B // 2], os.path.join(docs, "sample_after.png"))
    size = os.path.getsize(out_mp4)
    print(f"wrote {out_mp4} ({size} bytes), {images.shape[0]} frames {W}x{H} @ {FPS}fps")
    print("wrote docs/sample_before.png + docs/sample_after.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
