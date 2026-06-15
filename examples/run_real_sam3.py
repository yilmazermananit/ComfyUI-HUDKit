"""REAL SAM3 integration test (headless).

Runs ComfyUI's native SAM3 Video Track on a real decoded clip to produce REAL
per-frame masks + persistent-object SAM3_TRACK_DATA, then feeds that into the YEA HUD
Renderer with the user's assets (Tektur, track_random) and exports an MP4 with the
original audio + fps. Saves first/mid/last PNGs.

Requires: reference ComfyUI >=0.20.1 clone with native SAM3, the venv (torch), and
models/checkpoints/sam3.1_multiplex_fp16.safetensors. Slow on MPS — keep N_FRAMES small.

Run:  /path/to/.venv/bin/python examples/run_real_sam3.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from fractions import Fraction

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # let unsupported MPS ops fall back to CPU

REF = "/Users/differenterman/yonder-hud-build/comfyui-ref"
PROJ = "/Users/differenterman/Documents/ComfyUI/custom_nodes/ComfyUI-YEAHUD"
MODELS = "/Users/differenterman/Documents/ComfyUI/models"
SRC = os.path.expanduser(
    "~/Downloads/magnific_langr-adam-topa-vursun-sadece-sound-fx-olsun-mzik-_kling_4K_21-9_24fps_8724.mp4")
CKPT = "sam3.1_multiplex_fp16.safetensors"
PROMPT = "person"
N_FRAMES = 24          # keep small on MPS
TARGET_W = 896
OUT = os.path.join(PROJ, "examples", "real_sam3_output.mp4")

# ref clone first so `import nodes` / comfy / comfy_extras resolve to ComfyUI core;
# PROJ also on path so `import yea_hud` (our business pkg) resolves.
sys.path.insert(0, PROJ)
sys.path.insert(0, REF)
os.chdir(REF)


def _load_file_module(name, relpath):
    """Load one of OUR modules by file path (since `nodes` resolves to ComfyUI's core)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROJ, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _node_output_first(no):
    for attr in ("args", "result", "values"):
        v = getattr(no, attr, None)
        if isinstance(v, (list, tuple)) and v:
            return v[0]
    try:
        return no[0]
    except Exception:
        return no


def decode(src, n, target_w):
    import av
    import numpy as np
    from PIL import Image
    cont = av.open(src)
    vs = cont.streams.video[0]
    fps = float(vs.average_rate)
    out = []
    for i, frame in enumerate(cont.decode(vs)):
        if i >= n:
            break
        out.append(frame.to_ndarray(format="rgb24"))
    cont.close()
    H0, W0 = out[0].shape[:2]
    sc = target_w / W0
    W = (int(round(W0 * sc)) // 8) * 8
    H = (int(round(H0 * sc)) // 8) * 8
    import torch
    arr = np.stack([np.asarray(Image.fromarray(f).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
                    for f in out])
    return torch.from_numpy(arr), fps, W, H


def main():
    ckpt_path = os.path.join(MODELS, "checkpoints", CKPT)
    if not os.path.exists(ckpt_path):
        print("CHECKPOINT MISSING:", ckpt_path); return 2
    import torch
    torch.set_grad_enabled(False)  # ComfyUI runs nodes under no_grad; we call execute() directly
    import folder_paths
    folder_paths.add_model_folder_path("checkpoints", os.path.join(MODELS, "checkpoints"))
    import comfy.model_management as mm
    from nodes import CheckpointLoaderSimple, CLIPTextEncode
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack

    hud = _load_file_module("yea_hud_renderer", "nodes/hud_renderer.py")
    ap = _load_file_module("yea_asset_pack", "nodes/asset_pack.py")

    print("device:", mm.get_torch_device())
    # Cache the (slow) decode + SAM3 result so HUD visuals can be iterated fast.
    cache = os.path.join(PROJ, "examples", ".sam3_cache.pt")
    if os.path.exists(cache):
        print("loading cached frames + REAL SAM3 track_data…")
        blob = torch.load(cache, weights_only=False)
        frames, fps, W, H, track_data = blob["frames"], blob["fps"], blob["W"], blob["H"], blob["track_data"]
    else:
        frames, fps, W, H = decode(SRC, N_FRAMES, TARGET_W)
        print(f"decoded {frames.shape[0]} frames {W}x{H} @ {fps}fps")
        print("loading SAM3 checkpoint…")
        model, clip, _vae = CheckpointLoaderSimple().load_checkpoint(ckpt_name=CKPT)
        cond = CLIPTextEncode().encode(clip=clip, text=PROMPT)[0]
        print("running SAM3 Video Track (real)…")
        no = SAM3_VideoTrack.execute(images=frames, model=model, conditioning=cond,
                                     detection_threshold=0.5, max_objects=8, detect_interval=1)
        track_data = _node_output_first(no)
        td_cpu = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in track_data.items()}
        torch.save({"frames": frames, "fps": fps, "W": W, "H": H, "track_data": td_cpu}, cache)
        track_data = td_cpu
    pm = track_data.get("packed_masks")
    pm = track_data.get("packed_masks")
    print("REAL track_data: packed_masks=",
          (tuple(pm.shape), str(pm.dtype)) if pm is not None else None,
          "| n_frames=", track_data.get("n_frames"),
          "| n_obj=", (pm.shape[1] if pm is not None else 0),
          "| scores=", [round(float(s), 3) for s in (track_data.get("scores") or [])][:8],
          "| orig_size=", track_data.get("orig_size"))

    # assets
    manual = {}
    for i in range(1, 7):
        p = os.path.join(PROJ, "examples", "assets", f"Untitled-1_Frame {i}.png")
        if os.path.exists(p):
            im, mk = ap._load_pair_from_file(p); manual[f"frame_{i}_image"] = im; manual[f"frame_{i}_mask"] = mk
    for i, fn in enumerate(["Untitled-1_TARGET POINT.png", "Untitled-1_TARGET POINT 2.png"], 1):
        p = os.path.join(PROJ, "examples", "assets", fn)
        if os.path.exists(p):
            im, mk = ap._load_pair_from_file(p); manual[f"target_{i}_image"] = im; manual[f"target_{i}_mask"] = mk
    asset_pack = ap.run_asset_pack(folder="", manual=manual)

    annotations = {"version": 1, "tracks": [{"track_id": f"person:{i}", "category": "person",
                   "short_label": "PERSON", "detailed_label": "PERSON DETECTED BY SAM3",
                   "display_label": "PERSON"} for i in range(pm.shape[1] if pm is not None else 0)]}

    print("rendering HUD on real SAM3 tracks…")
    images, _dbg, tj = hud.run_hud_renderer(
        frames, track_data=track_data, annotations_json=json.dumps(annotations),
        seed=77, style_mode="mixed", max_objects=12, display_probability=1.0, padding_x=8, padding_y=8,
        smoothing=0.5, track_iou_threshold=0.25, track_memory_frames=8, frame_opacity=1.0,
        fill_opacity=0.0, label_opacity=0.92, line_width=2, target_scale=0.34,
        caption_mode="mixed", label_length_mode="adaptive", color_mode="track_random", asset_pack=asset_pack)
    print("tracking_json:", tj)

    import av
    import numpy as np
    inp = av.open(SRC); out = av.open(OUT, "w")
    vst = out.add_stream("h264", rate=Fraction(round(fps * 1000), 1000))
    vst.width, vst.height, vst.pix_fmt = W, H, "yuv420p"
    ast = out.add_stream_from_template(inp.streams.audio[0]) if inp.streams.audio else None
    for f in range(images.shape[0]):
        a = (images[f].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        for pkt in vst.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(a), format="rgb24").reformat(format="yuv420p")):
            out.mux(pkt)
    for pkt in vst.encode(None):
        out.mux(pkt)
    if ast is not None:
        for pkt in inp.demux(inp.streams.audio[0]):
            if pkt.dts is not None and pkt.pts is not None and float(pkt.pts * pkt.time_base) <= N_FRAMES / fps + 0.5:
                pkt.stream = ast; out.mux(pkt)
    out.close(); inp.close()
    docs = os.path.join(PROJ, "docs")
    B = images.shape[0]
    from PIL import Image
    for nm, ix in (("first", 0), ("mid", B // 2), ("last", B - 1)):
        Image.fromarray((images[ix].clamp(0, 1) * 255).round().to(torch.uint8).numpy(), "RGB").save(
            os.path.join(docs, f"real_sam3_{nm}.png"))
    print(f"WROTE {OUT} ({os.path.getsize(OUT)} bytes) + docs/real_sam3_first/mid/last.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
