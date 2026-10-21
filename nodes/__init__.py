"""Thin ComfyUI node layer (schema + wiring) for ComfyUI-HUDKit.

Each node module exposes a schema-agnostic ``run_*`` function (reused by both the
V3 node and the V1 shim) and, when ``comfy_api`` is importable, a V3 ``io.ComfyNode``
subclass assigned to module-level ``NODE``. Business logic lives in ``yea_hud``.
"""
