# Example workflows

Two importable graphs are provided:

| File | Use |
|------|-----|
| `sam3_hud_video_workflow.json` | **Primary.** Load Video → SAM3 Video Track (`SAM3_TRACK_DATA`) → YEA HUD Renderer → Create Video → Save Video. Stable native identities. |
| `sam3_hud_image_batch_workflow.json` | Fallback. SAM3 Detect (`BOUNDING_BOX` + `MASK`) → YEA HUD Renderer, for per-frame image-batch detection without persistent tracking. |

## What you do
1. **Load Video** — pick your clip.
2. **Load Checkpoint** — pick your SAM3 checkpoint (provides MODEL + CLIP).
3. **CLIP Text Encode** — type a prompt, e.g. `person`.
4. **Load Image** ×N — pick your transparent frame PNGs and target PNGs; connect each node's
   **IMAGE** and **MASK** outputs to the matching renderer inputs.
5. Press **Run**.

You never convert bbox types, add SEGS/Impact/KJNodes, or wire a crop/resize bridge.

## Preparing PNG assets
- **Background transparent** (real alpha channel).
- **Graphic in white** (the renderer recolors white → the track's palette color via
  `out_rgb = palette_color × source_luminance`).
- Canvas can be square; the renderer trims transparent margins (`trim_transparent_assets`)
  then fits the art to each detection box (`asset_fit_mode`: `stretch` for frames, `contain`
  for targets).
- Export as **PNG** with the alpha channel preserved.
- In ComfyUI, drop a **Load Image** node per asset and connect **both** its `IMAGE` and
  `MASK` outputs. (ComfyUI's Load Image MASK is the inverse alpha; the renderer recovers
  true alpha automatically — see `docs/troubleshooting.md`.)

## Frame / target slots
- Up to **6** frame assets (`frame_1..6_image` + `_mask`) and **2** targets
  (`target_1..2_image` + `_mask`). Connect any subset — the renderer auto-detects which
  slots are connected and only assigns those. Each track deterministically picks one
  eligible asset from `hash(seed, track_id)`, so a given person keeps the same asset.

## No assets? Still works
Leave all asset inputs empty and the renderer draws **procedural** HUD (brackets,
outlines, crosshairs, captions) using the palette. Set `style_mode` to taste
(`mixed` is the default rich look).

## Audio & fps
`Get Video Components` exposes the original `audio` and `fps`; both are passed straight
into `Create Video`, so the saved MP4 keeps them. The renderer never touches audio/fps.
