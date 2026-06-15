# Comfy Cloud Compatibility Report — ComfyUI-YonderHUD

> Goal: document, from **official sources**, the real target environment and the
> real path (and limits) to running this node on official Comfy Cloud. Every
> non-obvious claim is cited. Where Comfy Org has **not** published something, it
> is tagged **NOT DOCUMENTED** rather than guessed (per the project rules).

## 1. Target ComfyUI version

| Fact | Value | Source |
|------|-------|--------|
| Dev/test baseline (this machine, verified) | ComfyUI **0.24.0**, HEAD `7d4194d` (2026-06-15) | local `comfyui_version.py` |
| Native SAM3 first shipped | **v0.20.1** (2026-04-27); PR #13408 merged 2026-04-23 | docs.comfy.org/changelog ; github.com/Comfy-Org/ComfyUI/pull/13408 |
| Comfy Cloud version channel | "keeps up with the latest ComfyUI releases" — **no published pin** | blog.comfy.org/p/comfy-cloud |

**Decision:** target the native types as they exist in **0.20.1 → 0.24.x** (the
window in which `SAM3_Detect`, `SAM3_VideoTrack`, the `BOUNDING_BOX` and
`SAM3_TRACK_DATA` types all exist). `pyproject.toml` sets `requires-comfyui >= 0.20.1`.
We do **not** assume a specific Cloud version number — Cloud's exact build is
**NOT DOCUMENTED**; we only rely on the native type *contract* (verified in
`runtime_findings.md`) plus shipped inspector nodes for empirical confirmation.

## 2. Is native SAM3 actually on Cloud?

- Native SAM3 (`comfy_extras/nodes_sam3.py`) is **core** ComfyUI, weights hosted on
  the Comfy-Org HuggingFace org. (github.com/Comfy-Org/ComfyUI/pull/13408)
- **Whether the native core `SAM3_*` nodes + weights are enabled on Cloud is NOT
  DOCUMENTED.** The Cloud `supported-nodes` listing documents a *community*
  `comfyui-sam3` pack (14 nodes), distinct from core `comfy_extras/nodes_sam3.py`.
  (comfy.org/cloud/supported-nodes/)

**Risk + mitigation.** The community pack may emit *different* socket types than the
core nodes this pack was verified against. We do **not** assume compatibility from
socket color or name (project rule). Mitigation = the shipped inspector nodes
(`Yonder Inspect SAM3 Track Data`, `Yonder Inspect Bounding Boxes`): the user runs
them on whatever SAM3 their Cloud has and reads the **actual** runtime structure.
The adapters are defensive (multiple shapes), and the renderer accepts whichever of
`track_data` / `bboxes` / `masks` is actually present.

## 3. Can a custom node run on Cloud, and how?

- **Custom-node UPLOAD is DISABLED on Comfy Cloud.** Official: *"Custom node uploads
  are currently disabled… Comfy Cloud already includes the most widely used custom
  nodes, and they'll keep expanding the list over time."*
  (support.comfy.org compatibility article; docs.comfy.org/get_started/cloud)
- Cloud runs a **curated, pre-installed allowlist**, checked via the **"Nodes" tab**
  in the Cloud side-panel — **not** Manager/Registry self-install, **not** user upload.
  (comfy.org/cloud/ ; comfy.org/cloud/supported-nodes/)

**Therefore:** this node cannot be "installed" on Cloud by us or the user today. It
runs on Cloud **only if Comfy Org adds it to the pre-installed set.**

## 4. Registry publish does NOT equal Cloud availability

This is called out explicitly because it is a common misconception:

- The Registry publishing docs contain **no statement** that publishing grants Cloud
  availability. (docs.comfy.org/registry/publishing)
- Cloud disables upload and curates a pre-installed set it expands "over time."
- **There is NO documented self-serve Registry→Cloud inclusion process.** The only
  documented pathway is an **intake form** for custom-node authors from the Cloud
  announcement: *"If you are a custom node author who is interested in working with
  us… put down your info here."* (blog.comfy.org/p/comfy-cloud) Beyond "submit the
  form and Comfy Org curates," the criteria are **NOT DOCUMENTED**.

See `docs/cloud_inclusion_request.md` for the drafted submission.

## 5. How this node satisfies Cloud / Registry constraints

The Registry runs an **AI + static-analysis security scanner**; flagged patterns →
`#security-review-council` Discord. Banned/flagged: `eval`/`exec`, runtime
`pip`/subprocess installs, code obfuscation; HTTP requests are flagged.
(docs.comfy.org/registry/standards ; blog.comfy.org/p/comfyui-2025-jan-security-update)

| Requirement | Status in ComfyUI-YonderHUD |
|-------------|------------------------------|
| No model downloads | ✅ none — pure compositor over inputs |
| No network / HTTP requests | ✅ none anywhere in `yonder_hud/` or `nodes/` |
| No runtime package install | ✅ deps (`numpy`,`Pillow`) declared in pyproject; torch from env |
| No subprocess / arbitrary command exec | ✅ none |
| No `eval` / `exec` / obfuscation | ✅ none — readable source |
| No custom HTTP server / aiohttp routes | ✅ none; optional `web/js` is UX-only |
| No arbitrary filesystem access | ✅ reads only ComfyUI-provided tensors + optional bundled font; no user-path reads |
| Deps already in ComfyUI | ✅ torch (env) + numpy + Pillow only; **no** OpenCV/SciPy hard dep (cv2 = optional fast path) |
| No third-party bbox bridge (Impact/KJ/SEGS) | ✅ consumes native `SAM3_TRACK_DATA`/`BOUNDING_BOX`/`MASK` only |
| Server-side compositing | ✅ all rendering in `execute()`; frontend JS optional |
| Frontend not required for rendering | ✅ backend-only works headless |

A static self-audit script is provided: `tests/test_cloud_safety.py` greps the
package for banned patterns and fails CI if any appear.

## 6. Registry publishing (when ready)

1. Create a publisher at registry.comfy.org → note your **PublisherId** (immutable).
2. Set `[tool.comfy].PublisherId` and `[project.urls].Repository` in `pyproject.toml`
   (currently placeholders).
3. Generate a Registry API key; store as GitHub secret **`REGISTRY_ACCESS_TOKEN`**
   (the literal name `COMFY_API_KEY` is **NOT** the documented secret name).
4. `comfy node publish` (CLI) or the GitHub Action on version bump.
   (docs.comfy.org/registry/publishing ; docs.comfy.org/registry/specifications)

`pyproject.toml` already conforms: semver `version`, `name` (lowercase, valid chars),
`license`, required `Repository` URL, `[tool.comfy].PublisherId`, `requires-comfyui`.

## 7. Generic core-contribution evaluation — "SAM3 Track Overlay"

Phase 4 (brief): a **non-branded** subset — *SAM3 Track Overlay* — could be proposed
as a ComfyUI **core** contribution (`comfy_extras`). Rationale: it pairs naturally
with the existing `SAM3_TrackPreview` (which already draws colored masks + numbers)
by adding bracket/label/crosshair overlays driven by `SAM3_TRACK_DATA`, with zero
extra dependencies (torch-only, matching SAM3's "extra-dependency-free" goal). See
`docs/core_contribution_proposal.md` for the scoped proposal. This is the most viable
route to Cloud presence, because **core nodes ship with Cloud's base ComfyUI**,
sidestepping the undocumented custom-node curation gate.

## 8. Confidence & open items to verify with Comfy Org

- **NOT DOCUMENTED:** exact Cloud ComfyUI version/pin; whether core SAM3 (+weights)
  is enabled on Cloud vs only the community `comfyui-sam3` pack; Cloud's per-node
  runtime sandbox specifics; the concrete Registry→Cloud inclusion criteria.
- **Action:** confirm the above empirically inside a Cloud session (Nodes tab +
  inspector nodes) and via the Cloud author intake form / support@comfy.org, before
  claiming Cloud support in any user-facing copy.
