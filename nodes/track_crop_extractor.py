"""YEA Track Crop Extractor — representative crops per stable SAM3 track.

So an external VLM describes each track ONCE (from a few good crops) instead of
captioning every frame. Scores frames per track by visible area, centeredness and
sharpness, picks the best samples, crops the padded bbox, and emits a uniform IMAGE
batch + a manifest (crop_index -> track_id/frame/bbox) + a contact sheet.
No network, no model.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("YEAHUD.crop_extractor")

DISPLAY_NAME = "YEA Track Crop Extractor"
NODE_ID = "YEATrackCropExtractor"
CATEGORY = "YEA/HUD/Annotation"

SELECTION_MODES = ("best", "largest", "sharpest", "most_centered", "multi_sample")
_CROP_SIZE = 384
_SHEET_THUMB = 128


def _resize_contain(crop, size):
    """crop [h,w,3] float -> [size,size,3] float, aspect-preserved, black-padded."""
    import torch
    import torch.nn.functional as F
    h, w = crop.shape[0], crop.shape[1]
    if h < 1 or w < 1:
        return torch.zeros(size, size, 3, dtype=torch.float32)
    scale = min(size / h, size / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = F.interpolate(crop.permute(2, 0, 1).unsqueeze(0), size=(nh, nw),
                            mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    out = torch.zeros(size, size, 3, dtype=torch.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw, :] = resized.clamp(0, 1)
    return out


def _sharpness(crop):
    """Gradient-variance sharpness proxy on a [h,w,3] crop."""
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return 0.0
    gray = crop.mean(dim=-1)
    dx = gray[:, 1:] - gray[:, :-1]
    dy = gray[1:, :] - gray[:-1, :]
    return float(dx.var().item() + dy.var().item())


def _contact_sheet(crops):
    """Tile a [K,S,S,3] batch into a single [1,GH,GW,3] grid of thumbnails."""
    import math
    import torch
    import torch.nn.functional as F
    k = crops.shape[0]
    if k == 0:
        return torch.zeros(1, _SHEET_THUMB, _SHEET_THUMB, 3, dtype=torch.float32)
    thumbs = F.interpolate(crops.permute(0, 3, 1, 2), size=(_SHEET_THUMB, _SHEET_THUMB),
                           mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))
    sheet = torch.zeros(rows * _SHEET_THUMB, cols * _SHEET_THUMB, 3, dtype=torch.float32)
    for i in range(k):
        r, c = divmod(i, cols)
        sheet[r * _SHEET_THUMB:(r + 1) * _SHEET_THUMB, c * _SHEET_THUMB:(c + 1) * _SHEET_THUMB, :] = thumbs[i]
    return sheet.unsqueeze(0)


def run_track_crop_extractor(frames, track_data, samples_per_track=3, crop_padding=0.15,
                             minimum_track_area=0.0, selection_mode="best"):
    """Return (crops_image, crop_manifest_json, contact_sheet_image, track_indices_json)."""
    import torch
    from yea_hud import track_adapter
    from yea_hud.geometry import clamp_box

    frames = frames.detach().to("cpu", dtype=torch.float32)
    B, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    frame_area = float(max(1, W * H))
    selection_mode = selection_mode if selection_mode in SELECTION_MODES else "best"

    per_frame, _n_obj = (track_adapter.adapt_track_data(track_data)
                         if track_adapter.is_track_data(track_data) else ([], 0))

    per_track = {}
    for f, ds in enumerate(per_frame):
        if f >= B:
            break
        for d in ds:
            per_track.setdefault(d.source_track_id, []).append((f, d))

    crops, manifest, track_idx = [], [], {}
    for tid in sorted(per_track, key=lambda k: str(k)):
        occ = per_track[tid]
        if max(d.area for _, d in occ) / frame_area < float(minimum_track_area):
            continue
        scored = []
        for f, d in occ:
            box = clamp_box((d.x1 - d.width * crop_padding, d.y1 - d.height * crop_padding,
                             d.x2 + d.width * crop_padding, d.y2 + d.height * crop_padding), W, H)
            x1, y1, x2, y2 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = frames[f, y1:y2, x1:x2, :3]
            area_ratio = d.area / frame_area
            cx, cy = d.cx, d.cy
            center = 1.0 - min(1.0, (((cx - W / 2) / (W / 2)) ** 2 + ((cy - H / 2) / (H / 2)) ** 2) ** 0.5 / 1.4142)
            sharp = _sharpness(crop)
            if selection_mode == "largest":
                score = area_ratio
            elif selection_mode == "most_centered":
                score = center
            elif selection_mode == "sharpest":
                score = sharp
            else:  # best / multi_sample base score
                score = 0.5 * area_ratio + 0.3 * center + 0.2 * min(1.0, sharp * 50)
            scored.append((score, f, (x1, y1, x2, y2)))
        if not scored:
            continue
        if selection_mode == "multi_sample":
            by_frame = sorted(scored, key=lambda s: s[1])
            n = min(samples_per_track, len(by_frame))
            step = max(1, len(by_frame) // max(1, n))
            picks = by_frame[::step][:n]
        else:
            picks = sorted(scored, key=lambda s: -s[0])[:max(1, samples_per_track)]
        sample_frames = []
        for score, f, (x1, y1, x2, y2) in picks:
            sq = _resize_contain(frames[f, y1:y2, x1:x2, :3], _CROP_SIZE)
            manifest.append({"crop_index": len(crops), "track_id": str(tid), "frame_index": int(f),
                             "bbox": [x1, y1, x2, y2], "score": round(float(score), 4)})
            crops.append(sq)
            sample_frames.append(int(f))
        track_idx[str(tid)] = sample_frames

    if crops:
        crops_t = torch.stack(crops, dim=0)
    else:
        crops_t = torch.zeros(1, _CROP_SIZE, _CROP_SIZE, 3, dtype=torch.float32)
    contact = _contact_sheet(crops_t if crops else torch.zeros(0, _CROP_SIZE, _CROP_SIZE, 3))
    manifest_json = json.dumps({"crops": manifest, "n_tracks": len(track_idx)}, ensure_ascii=False)
    return crops_t, manifest_json, contact, json.dumps(track_idx, ensure_ascii=False)


try:
    from comfy_api.latest import io

    SAM3TrackData = io.Custom("SAM3_TRACK_DATA")

    class YEATrackCropExtractor(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=NODE_ID,
                display_name=DISPLAY_NAME,
                category=CATEGORY,
                description=(
                    "Extract a few representative crops per stable SAM3 track (best/largest/"
                    "sharpest/most_centered/multi_sample) so a VLM captions each track ONCE. "
                    "Outputs crops (IMAGE batch), a manifest, a contact sheet, and track indices."
                ),
                search_aliases=["crop", "sam3", "track", "vlm", "sample", "caption"],
                inputs=[
                    io.Image.Input("frames", tooltip="Decoded video frames [B,H,W,3]."),
                    SAM3TrackData.Input("track_data", tooltip="Native SAM3_TRACK_DATA."),
                    io.Int.Input("samples_per_track", default=3, min=1, max=12),
                    io.Float.Input("crop_padding", default=0.15, min=0.0, max=1.0, step=0.01),
                    io.Float.Input("minimum_track_area", default=0.0, min=0.0, max=1.0, step=0.0005,
                                   tooltip="Skip tracks whose max bbox area ratio is below this."),
                    io.Combo.Input("selection_mode", options=list(SELECTION_MODES), default="best"),
                ],
                outputs=[
                    io.Image.Output("crops", display_name="crops"),
                    io.String.Output("crop_manifest_json", display_name="crop_manifest_json"),
                    io.Image.Output("contact_sheet", display_name="contact_sheet"),
                    io.String.Output("track_indices_json", display_name="track_indices_json"),
                ],
            )

        @classmethod
        def execute(cls, frames, track_data, samples_per_track=3, crop_padding=0.15,
                    minimum_track_area=0.0, selection_mode="best") -> io.NodeOutput:
            crops, manifest, sheet, indices = run_track_crop_extractor(
                frames, track_data, samples_per_track, crop_padding, minimum_track_area, selection_mode)
            return io.NodeOutput(crops, manifest, sheet, indices)

    NODE = YEATrackCropExtractor
except ImportError:  # pragma: no cover
    NODE = None
