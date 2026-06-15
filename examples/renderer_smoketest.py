"""RENDERER SMOKE TEST — NOT a SAM3 integration test.

Renders on a real decoded video with a STAND-IN track (a synthetic centred box) to
exercise the render path on real footage: asset compositing, Tektur, track_random
colors, per-asset opacity, stable IDs, audio+fps mux, first/mid/last PNGs. It does NOT
run SAM3 and the box does NOT follow the real subject.

The REAL SAM3 integration test = run `examples/real_video_sam3_workflow.json` in ComfyUI
>=0.20.1 with `models/checkpoints/sam3.1_multiplex_fp16.safetensors` (Comfy-Org/sam3.1):
Load Video -> SAM3 Video Track -> YEA HUD Renderer -> Create Video. Replace SRC with any clip.
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

import av
import numpy as np
import torch
from PIL import Image

from nodes.hud_renderer import run_hud_renderer
from nodes.asset_pack import run_asset_pack, _load_pair_from_file

SRC = os.path.expanduser(
    "~/Downloads/magnific_langr-adam-topa-vursun-sadece-sound-fx-olsun-mzik-_kling_4K_21-9_24fps_8724.mp4")
OUT = os.path.join(_HERE, "renderer_smoketest_output.mp4")
TARGET_W = 1280
ASSET_DIR = os.path.join(_HERE, "assets")
FRAME_FILES = [f"Untitled-1_Frame {i}.png" for i in range(1, 7)]
TARGET_FILES = ["Untitled-1_TARGET POINT.png", "Untitled-1_TARGET POINT 2.png"]


def decode(src, target_w):
    cont = av.open(src)
    vs = cont.streams.video[0]
    fps = float(vs.average_rate)
    frames = []
    for frame in cont.decode(vs):
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
    cont.close()
    H0, W0 = frames[0].shape[:2]
    scale = target_w / W0
    W = (int(round(W0 * scale)) // 2) * 2
    H = (int(round(H0 * scale)) // 2) * 2
    out = np.empty((len(frames), H, W, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(f).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(out), fps, W, H


def _pack_masks(m):
    shifts = torch.arange(8)
    return (m.view(*m.shape[:-1], -1, 8) * (1 << shifts)).sum(-1).byte()


def standin_track_data(B, H, W, hm=180, wm=320):
    """One centred PERSON track with slight sway (stand-in for SAM3)."""
    fp = []
    for f in range(B):
        objs = torch.zeros(1, hm, wm, dtype=torch.bool)
        cx = wm * 0.5 + np.sin(f / 12.0) * wm * 0.04
        bw, bh = wm * 0.26, hm * 0.82
        x0 = int(cx - bw / 2); y0 = int(hm * 0.10)
        objs[0, y0:y0 + int(bh), max(0, x0):x0 + int(bw)] = True
        fp.append(_pack_masks(objs))
    return {"packed_masks": torch.stack(fp), "n_frames": B, "scores": [0.95], "orig_size": (H, W)}


def main():
    if not os.path.exists(SRC):
        print("SRC video not found:", SRC); return 1
    frames, fps, W, H = decode(SRC, TARGET_W)
    B = frames.shape[0]
    print(f"decoded {B} frames {W}x{H} @ {fps}fps")

    track_data = standin_track_data(B, H, W)
    annotations = {"version": 1, "video": {"width": W, "height": H, "fps": fps, "frame_count": B},
                   "tracks": [{"track_id": "person:0", "category": "person", "short_label": "PERSON",
                               "detailed_label": "MAN IN MOTION WITH SOCCER BALL",
                               "action_label": "KICKING SOCCER BALL",
                               "display_label": "MAN KICKING SOCCER BALL", "confidence": 0.95,
                               "children": [{"annotation_id": "person:0:foot", "category": "footwear",
                                             "short_label": "FOOTWEAR",
                                             "bbox_by_frame": {str(f): [int(W*0.46), int(H*0.74), int(W*0.56), int(H*0.9)] for f in range(0, B, 6)}}]}],
                   "frame_annotations": {str(f): [{"annotation_id": "ball:0", "category": "soccer_ball",
                                                   "short_label": "SOCCER BALL",
                                                   "bbox": [int(W*0.47), int(H*0.82), int(W*0.55), int(H*0.93)],
                                                   "priority": 75}] for f in range(B)}}

    manual = {}
    for i, fn in enumerate(FRAME_FILES, 1):
        p = os.path.join(ASSET_DIR, fn)
        if os.path.exists(p):
            im, mk = _load_pair_from_file(p); manual[f"frame_{i}_image"] = im; manual[f"frame_{i}_mask"] = mk
    for i, fn in enumerate(TARGET_FILES, 1):
        p = os.path.join(ASSET_DIR, fn)
        if os.path.exists(p):
            im, mk = _load_pair_from_file(p); manual[f"target_{i}_image"] = im; manual[f"target_{i}_mask"] = mk
    asset_pack = run_asset_pack(folder="", manual=manual)

    images, _dbg, tj = run_hud_renderer(
        frames, track_data=track_data, annotations_json=json.dumps(annotations),
        seed=77, style_mode="mixed", max_objects=24, display_probability=1.0, padding_x=10, padding_y=10,
        smoothing=0.6, track_iou_threshold=0.25, track_memory_frames=8, frame_opacity=1.0,
        fill_opacity=0.22, label_opacity=0.92, line_width=2, target_scale=0.34,
        caption_mode="mixed", label_length_mode="adaptive", show_child_annotations=True,
        show_action_label=True, color_mode="track_random", asset_pack=asset_pack)
    print("tracking:", tj)

    # encode video + copy original audio + preserve fps
    inp = av.open(SRC)
    out = av.open(OUT, "w")
    vst = out.add_stream("h264", rate=Fraction(round(fps * 1000), 1000))
    vst.width, vst.height, vst.pix_fmt = W, H, "yuv420p"
    ast = None
    if inp.streams.audio:
        try:
            ast = out.add_stream_from_template(inp.streams.audio[0])
        except Exception as exc:
            print("audio copy unavailable:", exc)
    for f in range(B):
        arr = (images[f].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24").reformat(format="yuv420p")
        for pkt in vst.encode(vf):
            out.mux(pkt)
    for pkt in vst.encode(None):
        out.mux(pkt)
    if ast is not None:
        for pkt in inp.demux(inp.streams.audio[0]):
            if pkt.dts is None:
                continue
            pkt.stream = ast
            out.mux(pkt)
    out.close(); inp.close()

    docs = os.path.join(_ROOT, "docs")
    for name, idx in (("first", 0), ("mid", B // 2), ("last", B - 1)):
        Image.fromarray((images[idx].clamp(0, 1) * 255).round().to(torch.uint8).numpy(), "RGB").save(
            os.path.join(docs, f"smoketest_{name}.png"))
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes), audio={'yes' if ast else 'no'}")
    print("wrote docs/real_first.png, docs/real_mid.png, docs/real_last.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
