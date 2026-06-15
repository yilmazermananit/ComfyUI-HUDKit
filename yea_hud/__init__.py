"""yea_hud — schema-agnostic business logic for ComfyUI-YEAHUD.

This package intentionally performs NO heavy imports at package-init time
(no torch / PIL at module load) so that ``yea_hud.types`` and pure helpers
can be imported in any context (including isolated unit tests) without pulling
in the full ComfyUI / torch stack. Import submodules explicitly where needed.
"""

__version__ = "0.1.0"
