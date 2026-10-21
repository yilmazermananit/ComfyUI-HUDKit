# INTERFACES — yea_hud module contract (v1)

Every module below lives in `yea_hud/` and is **schema-agnostic** (no ComfyUI
imports). Allowed deps: stdlib, `numpy`, `PIL` (Pillow), `torch` (already in the
ComfyUI env). Each module imports `from .types import ...`. Implement EXACTLY these
public signatures so the node layer and tests wire together with zero drift. No
stubs, no TODOs, no `pass`-only bodies. Every public function gets a docstring and
a unit test in `tests/test_<module>.py`.

Coordinate standard everywhere: absolute-pixel **xyxy** unless a name says otherwise.
RGB colors are `(r,g,b)` floats 0..1. Images passed between pure modules are numpy
`float32 [H,W,3] 0..1`; overlays are numpy `float32 [H,W,4]` RGBA (premultiplied=NO,
straight alpha). torch tensors only cross the node<->compositor boundary.

---

## utils.py — determinism + caching
```python
def stable_hash(*parts) -> int
    # 64-bit deterministic hash of the given parts (ints/strs/floats), independent
    # of PYTHONHASHSEED. Use hashlib.blake2b over a '|'-joined repr. Stable across runs/OS.

def rng_for(*parts) -> random.Random
    # returns random.Random seeded with stable_hash(*parts). Track-level identity:
    # rng_for(seed, track_id); frame micro-variation: rng_for(seed, track_id, frame_index).

def unit_float(*parts) -> float
    # deterministic float in [0,1) from stable_hash(*parts) (== rng_for(*parts).random()-equivalent,
    # but a single direct draw). Used for probability gates.

class LRUCache:
    def __init__(self, max_items: int = 512)
    def get(self, key)            # -> value | None
    def put(self, key, value)     # evicts oldest beyond max_items
    def clear(self)
    def __len__(self)
```

## geometry.py — box math, padding, smoothing, placement
```python
Box = tuple[float,float,float,float]  # xyxy (import from types if you prefer)

def xywh_to_xyxy(x,y,w,h) -> Box
def xyxy_to_xywh(x1,y1,x2,y2) -> tuple  # (x,y,w,h)
def normalize_box(x1,y1,x2,y2) -> Box   # swap if reversed (x2<x1 / y2<y1)
def clamp_box(box: Box, W:int, H:int) -> Box      # clip to [0,W]/[0,H]
def is_degenerate(box: Box, min_size: float = 1.0) -> bool   # width<min or height<min
def pad_box(box: Box, px: float, py: float, W:int, H:int, mode:str="pixels") -> Box
    # mode 'pixels': add px,py each side. mode 'percent': px,py are fractions of box w/h. Then clamp.
def iou(a: Box, b: Box) -> float
def centroid(box: Box) -> tuple[float,float]
def centroid_dist(a: Box, b: Box) -> float
def area(box: Box) -> float

def smoothing_to_alpha(smoothing: float) -> float
    # UI smoothing 0..1 -> EMA alpha. 0.0 => alpha 1.0 (raw, no smoothing);
    # 0.9 => alpha ~0.1 (very smooth/laggy). Monotonic. alpha = clamp(1.0 - smoothing, 0.02, 1.0).
def ema_box(current: Box, previous: Box|None, alpha: float) -> Box
    # previous None -> return current. else per-coord alpha*current + (1-alpha)*previous.
def adaptive_alpha(base_alpha: float, current: Box, previous: Box|None, frame_diag: float) -> float
    # bigger motion -> alpha closer to 1 (less smoothing). motion = centroid_dist/frame_diag.

def importance(box: Box, W:int, H:int, center_bias: float = 0.0) -> float
    # bbox_area/frame_area, optionally * center weight (1+center_bias*(1-dist_to_center_norm)).

def label_rect(box: Box, text_w:int, text_h:int, W:int, H:int, side:str) -> tuple[Box,str]
    # compute a label-strip rect that stays inside the image; flip side top<->bottom and
    # align left/right when it would overflow. returns (rect_xyxy, resolved_side 'top'|'bottom').
def target_center(box: Box) -> tuple[float,float]   # always bbox center
def asset_rect(box: Box, fit:str, inner_padding:float, aspect:tuple[int,int]|None) -> Box
    # placement rect for an asset given fit mode (stretch/contain/cover) + inner padding fraction.
```

## palette.py — colors
```python
def parse_hex(s: str) -> tuple[float,float,float] | None   # '#RRGGBB'/'RRGGBB' -> 0..1, else None
def resolve_palette(palette_json: str|None) -> list[tuple[float,float,float]]
    # parse JSON list of hex; drop invalid; if empty/None -> DEFAULT_PALETTE. Always >=1 color.
def color_for_track(palette, seed:int, track_id) -> tuple[int, tuple[float,float,float]]
    # deterministic: idx = stable_hash(seed, track_id) % len(palette). returns (idx, rgb).
def contrast_text_color(bg_rgb, mode:str="auto") -> tuple[float,float,float]
    # mode 'white'/'black' force; 'auto' -> luminance(bg)>0.55 ? black : white (rec.709 luma).
def luminance(rgb) -> float
```

## alpha.py — Load Image alpha recovery + asset prep (brief §8)
```python
def to_numpy_img(image) -> "np.ndarray"   # torch [H,W,C] or [1,H,W,C] or np -> np float32 [H,W,3] 0..1
def to_numpy_mask(mask) -> "np.ndarray|None"  # torch [H,W]/[1,H,W]/[H,W,1] -> np float32 [H,W] 0..1, or None
def recover_alpha(image_np, mask_np, *, invert_auto=True) -> "np.ndarray"
    # Returns straight alpha [H,W] 0..1. ComfyUI Load Image MASK = 1 - alpha (inverse) for the
    # transparent-PNG case. Heuristic (VERIFY against findings): Load Image builds MASK as
    # 1.-alpha_channel. So true_alpha = 1 - mask. If mask is None/empty/full -> fall back to
    # luminance of image as alpha. invert decision: if mask looks like background-is-1
    # (mean>0.5 and white-graphic case) invert. Document the rule; expose invert_auto override.
def luminance_alpha(image_np) -> "np.ndarray"   # per-pixel rec.709 luma as alpha fallback
def trim_transparent(rgb_np, alpha_np, threshold=0.02) -> tuple[rgb,alpha,bbox]
    # crop to alpha bounding box; returns cropped arrays + (x1,y1,x2,y2) in original coords.
    # if fully transparent -> return inputs unchanged with full-frame bbox.
def recolor_white_asset(rgb_np, alpha_np, color_rgb, opacity:float) -> "np.ndarray"
    # brief §8: out_rgb = color * source_luminance ; out_a = alpha * opacity. returns RGBA [H,W,4].
def resize_rgba(rgba_np, out_w:int, out_h:int) -> "np.ndarray"
    # Lanczos on downscale, Bicubic on upscale (PIL). preserves straight alpha (no premult bleed).
```

## text_render.py — fonts + label strips (brief §13.5)
```python
def get_font(size:int, font_path:str|None=None) -> "PIL.ImageFont"
    # cached by (path,size). Try a bundled/system open monospace; fallback ImageFont.load_default().
    # Never require a user font file. Cache in a module-level LRUCache.
def measure(text:str, font) -> tuple[int,int]   # (w,h) in px
def truncate_ellipsis(text:str, font, max_w:int) -> str  # add '…' to fit max_w
def render_label_strip(draw_or_size, *, text, font, fg_rgb, bg_rgb, bg_opacity, pad=(6,3)) -> "np.ndarray"
    # returns an RGBA [h,w,4] float32 strip: solid bg (bg_rgb @ bg_opacity) + text (fg_rgb).
    # uppercase handled by caller (captions). monospace. returns straight-alpha RGBA.
```

## procedural_hud.py — vector HUD primitives (brief §9). Draw onto an RGBA layer.
All functions take an `ImageDraw.Draw` over a straight-alpha RGBA canvas and a
`color_rgba=(r,g,b,a)` 0..1, draw in place, and clip to canvas bounds internally.
```python
def draw_corners(draw, box, color_rgba, line_width:int, corner_length_ratio=0.25, corner_gap=0)
def draw_outline(draw, box, color_rgba, line_width:int, glitch_segments:int=0, rng=None)
def draw_solid(draw, box, color_rgba)                 # fill_opacity already baked into color_rgba.a
def draw_crosshair(draw, center, size:int, color_rgba, line_width:int, gap:int, dot=True, ticks=False)
def draw_registration(draw, box, color_rgba, density:float, rng=None)  # diamonds/plus/ticks (brief §9.6)
# helper:
def new_rgba_layer(W:int,H:int) -> "PIL.Image"  # mode 'RGBA', fully transparent
```

## captions.py — caption text resolution (brief §13)
```python
def parse_captions_json(s:str|None) -> dict   # {} on None/invalid; supports by-track {"1":"..."} and {"frames":{...}}
def resolve_caption(*, source:str, det: Detection, track_id:int, order_index:int,
                    fixed_label:str, captions:dict, template:str|None,
                    frame_index:int) -> str
    # source in CAPTION_SOURCES. 'fixed'->fixed_label; 'bbox_label'->det.label or ''(SAM3 has none);
    # 'json_by_track'->captions[str(track_id)]; 'json_by_frame'->captions['frames'][str(frame)] by source_index;
    # 'template'-> expand tokens {track}{frame}{index}{label}{score}{x}{y}{w}{h}; 'none'->''.
    # Invalid template tokens must NOT raise (leave literal / blank). Returns the raw string (NOT uppercased).
def format_order_label(order_index:int, text:str) -> str   # e.g. '03. PERSON' (caller uppercases)
```

## bbox_adapter.py — native BOUNDING_BOX -> detections (brief §3,§4.3)
```python
def adapt_bboxes(bboxes, B:int, W:int, H:int) -> list[list[Detection]]
    # Accept the 3 native shapes (findings §3): single dict, list[dict], list[list[dict]].
    # Each dict: keys x,y,width,height (xywh, abs px) (+ optional score,label). ALSO defensively
    # accept x1,y1,x2,y2 dicts and 4-tuples/lists (xywh OR xyxy — detect: if 3rd/4th look like
    # absolute >x and >y treat as xyxy else xywh). Normalized (all<=1.0) -> *W/*H.
    # Output: per-frame list (length B) of Detections (xyxy, normalized/clamped/swapped,
    # degenerate dropped via geometry). frame_index/source_index set. score/label carried.
    # Broadcasting: single dict / list[dict] => same boxes for ALL B frames. list[list] shorter
    # than B => remaining frames get [] (NEVER auto-repeat a sequence — brief §7). Longer => ignore extra.
def summarize_bboxes(bboxes, max_frames=3, max_dets=10) -> dict
    # safe structural summary for the inspector / error messages (type, nesting, keys, counts,
    # coord-range guess pixel-vs-normalized). No giant tensor dumps.
```

## mask_adapter.py — native MASK handling + connected components (brief §6.2 node, §8 align)
```python
def normalize_mask_batch(mask, B:int) -> "list[np.ndarray]"
    # torch/np MASK -> list (len B) of float32 [H,W] 0..1. Handles [B,H,W], [H,W] (broadcast if B==1
    # else error-report), [B,1,H,W], flattened [N,H,W] (cannot split w/o counts -> return as-is list).
def mask_to_detections(mask_hw, frame_index:int, *, threshold:float, min_area:int,
                       merge_distance:int) -> list[Detection]
    # connected components on (mask>threshold). numpy/torch implementation (NO cv2 required;
    # optional cv2 fast path if importable). merge components whose boxes are within merge_distance.
    # drop area<min_area. returns Detections (xyxy) with source_index = component order.
def masks_to_native_bboxes(mask, threshold:float, min_area:int, merge_distance:int) -> list[list[dict]]
    # for the 'HUDKit Mask to Native BBoxes' node: returns native BOUNDING_BOX value
    # (list[list[{"x","y","width","height","score":1.0}]]) — abs-pixel xywh. NO custom type.
```

## track_adapter.py — SAM3_TRACK_DATA -> detections (PRIMARY path, findings §3.5)
```python
def unpack_packed_masks(packed)  # reimplement bit-unpack: [*,H,W//8] uint8 -> bool [*,H,W]. torch.
def is_track_data(obj) -> bool   # dict with 'packed_masks' and 'n_frames' keys
def adapt_track_data(track_data: dict, *, area_threshold:int=1) -> tuple[list[list[Detection]], int]
    # returns (per-frame detections length n_frames, n_obj).
    # For frame t, object i: unpack packed_masks[t,i] -> bool mask; if pixel area >= area_threshold:
    #   bbox = mask extent (xyxy in mask res) scaled to orig_size (H,W);
    #   Detection(frame_index=t, source_index=i, xyxy=..., score=scores[i] if i<len(scores) else None,
    #             source_track_id=i, mask_index=i). Empty mask -> skip (occlusion) but identity i persists.
    # packed_masks None -> ([[],...]*n_frames, 0). Resize bbox by (H/H_m, W/W_m).
def summarize_track_data(track_data) -> dict   # for the inspector node (shapes, n_frames, n_obj, scores, visibility/frame).
```

## tracker.py — identity unification + smoothing + lifecycle (brief §10,§11)
```python
class Tracker:
    def __init__(self, *, seed:int, iou_threshold:float, max_centroid_ratio:float,
                 memory_frames:int, smoothing:float, palette, n_frame_assets:int,
                 n_target_assets:int, style_mode:str, display_probability:float,
                 fill_probability:float=0.25, target_probability:float=0.7,
                 W:int, H:int, adaptive:bool=False)
    def update(self, frame_index:int, detections: list[Detection]) -> list[tuple[Track, Detection]]
        # If detections carry source_track_id (not None) -> PASSTHROUGH: map id->Track directly,
        # no IoU. Else greedy match by combined IoU + centroid + area score against live tracks
        # (deterministic order). New det -> new Track (id from a monotonic counter; ids NEVER reused).
        # Apply EMA smoothing (geometry.ema_box) to each matched track's box. Missing tracks age;
        # killed after memory_frames. Returns the (Track, Detection) pairs visible THIS frame, in a
        # deterministic order (by track_id). Track.style assigned ONCE via assign_style().
    def assign_style(self, track_id) -> TrackStyle
        # deterministic from hash(seed, track_id): color_id, primitive_style (respect style_mode:
        # e.g. style_mode 'corners' forces corners), frame_asset_id in [0,n_frame_assets) or -1,
        # target_asset_id, show (unit_float<display_probability), use_fill, use_target, use_caption,
        # caption_side, glitch_phase. Cached per track_id.
    @property
    def tracks(self) -> dict[int, Track]
def micro_variation(seed:int, track_id:int, frame_index:int, *, flicker_p:float,
                    style_flash_p:float, jitter_px:int, opacity_jitter:float) -> dict
    # deterministic per-frame dict: {'opacity_mul','line_dropout','target_flash','fill_flash','jitter'}.
```

## assets.py — asset slot management (brief §16)
```python
class AssetBank:
    def __init__(self, frame_pairs: list[tuple[img,mask]|None], target_pairs: list[...],
                 *, trim:bool, fit_frame:str, fit_target:str, max_cache_items:int)
        # pairs may contain None for unconnected slots. Build eligible index lists of CONNECTED slots.
    @property
    def n_frame_assets(self) -> int      # count of eligible (connected) frame assets
    @property
    def n_target_assets(self) -> int
    def frame_rgba(self, eligible_id:int, out_w:int, out_h:int, color_rgb, opacity:float) -> "np.ndarray|None"
        # recolor+resize the eligible frame asset to out_w/out_h; cached by (id,w-bucket,h-bucket,color_id).
    def target_rgba(self, eligible_id:int, out_w:int, out_h:int, color_rgb, opacity:float) -> "np.ndarray|None"
    # Image connected but mask not -> luminance fallback (alpha.luminance_alpha).
    # Mask connected but image not -> asset NOT rendered (return None) (brief §16).
```

## compositor.py — per-frame draw orchestration (brief §17). The integration point.
```python
def render_frames(frames,                      # torch IMAGE [B,H,W,3] 0..1 (or [B,H,W,4])
                  plans: list[list[RenderItem]],# per-frame render items (length B)
                  *, assets: AssetBank, palette, params: dict,
                  font_path:str|None, progress=None, interrupt=None) -> "torch.Tensor"
    # Returns IMAGE [B,H,W,3] float32 0..1. For each frame: move to CPU numpy ONCE, build a single
    # RGBA overlay (PIL) in draw order: solid blocks -> frame assets/procedural frames -> crosshair ->
    # label strips -> text -> micro details; alpha-composite over base; convert back ONCE; stack.
    # Respect per-item opacity_mul/flash/jitter. progress.update(1) per frame; interrupt() may raise.
    # Empty plan frame -> copy base unchanged. Never mutate input tensor in place.
def composite_rgba_over(base_rgb_np, overlay_rgba_np) -> "np.ndarray"  # straight-alpha over, returns [H,W,3].
```

---

### Test rules (every `tests/test_<module>.py`)
- Use plain `assert` + a `if __name__=='__main__'` runner OR pytest-style `def test_*`.
- Run with: `python -m pytest tests/test_<module>.py -q`
  (pytest is in the venv; if not, the file must also run via `python tests/test_<module>.py`).
- Cover the brief §25 cases relevant to the module (e.g. bbox: all the §25.1 shapes;
  alpha: §25.2; tracker: §25.3; compositor: §25.4).
- Tests must NOT require ComfyUI, network, or model files. Build tiny synthetic numpy/torch fixtures.
