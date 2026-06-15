# Draft — Comfy Cloud Custom-Node Inclusion Request

> Submit via the custom-node author intake linked from the Comfy Cloud
> announcement (blog.comfy.org/p/comfy-cloud) and/or support@comfy.org. There is
> **no documented self-serve path**; this is a curation request (see
> `cloud_compatibility.md` §4). Fill the bracketed fields before sending.

**Node pack:** ComfyUI-YEAHUD
**Repository:** [https://github.com/<org>/ComfyUI-YEAHUD]
**Registry id:** comfyui-yeahud (PublisherId: [your-publisher-id])
**Author / contact:** DifferentGuys Studio (YEA) — erman@differentguys.studio
**License:** MIT

## What it does
A server-side compositing pack that turns **native SAM3 tracking output** into
designed HUD video overlays (corner brackets, outlines, translucent blocks,
crosshairs, captioned track IDs, and the user's own recolored transparent PNG
frame/target assets). It consumes the **native** `SAM3_TRACK_DATA` (preferred) or
`BOUNDING_BOX` + `MASK` and outputs an `IMAGE` batch for Create Video / Save Video.

## Why it fits Cloud
- **Pairs with native SAM3** already shipping in core ComfyUI (v0.20.1+). No new model.
- **Zero third-party node dependencies** (no Impact Pack / KJNodes / SEGS / bbox bridges).
- **Dependencies already in the Cloud image:** torch (env) + numpy + Pillow. No OpenCV/SciPy hard dep.

## Cloud safety posture (maps to Registry security standards)
- ❌ no model downloads · ❌ no network/HTTP · ❌ no runtime pip · ❌ no subprocess
- ❌ no `eval`/`exec`/obfuscation · ❌ no custom HTTP routes/aiohttp · ❌ no arbitrary FS access
- ✅ pure deterministic compute in `execute()`; frontend JS is optional (rendering works headless)
- ✅ progress reporting + interruption checks for long videos
- A self-audit test (`tests/test_cloud_safety.py`) greps for banned patterns in CI.

## Verification material included
- `docs/runtime_findings.md` — exact native type structures the pack was built against (source-cited).
- `docs/cloud_compatibility.md` — honest target-version + inclusion analysis.
- Example workflow + integration test report against the real native SAM3 type contract.
- Inspector nodes (`YEA Inspect SAM3 Track Data`, `YEA Inspect Bounding Boxes`) so reviewers can confirm socket structures empirically inside Cloud.

## Open question for the Comfy team
Is core SAM3 (`comfy_extras/nodes_sam3.py`) + its weights enabled on Cloud today, or
only the community `comfyui-sam3` pack? Our nodes target the **native core** types;
we want to confirm socket compatibility on the actual Cloud build before launch.
