# Changelog

## 0.1.1 — 2026-10-21

- Publish the project as **ComfyUI-HUDKit**, with updated node display names,
  menu categories, repository links, and manual installation instructions.
- Preserve existing node IDs, socket types, and the Python module name for saved
  workflow compatibility.
- Restrict asset folder reads to ComfyUI's input directory, including symlink targets.
- Replace machine-specific example paths with environment variables and load the
  optional local SAM3 cache using PyTorch's restricted weights-only loader.
- Include bundled font files and their license in Python package data.
- Add contribution instructions and regression tests for asset directory containment.
- Configure pytest to avoid importing ComfyUI's root loader as a test package.

## 0.1.0 — June 2026

- Initial private development: native SAM3 tracking adapters, HUD compositing,
  annotations, asset packs, workflow examples, and synthetic tests.
