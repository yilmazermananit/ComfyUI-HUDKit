# Bundled fonts

Drop a `.ttf` / `.otf` here to make it the **default** HUD label font (matches the
reference editorial-mono look). The renderer auto-discovers fonts in this folder
(sorted order; first one wins) when no explicit `font` is selected.

- Prefer a **monospace / mono-grotesk** for the technical HUD look.
- Only ship a font you have the license to redistribute (add its license file here).
- You can also select a specific font at runtime via the renderer's `font` input
  (a filename in this folder, or an absolute path). Empty `font` = first bundled
  font → curated system monospace → Pillow default. A user font is never *required*.
