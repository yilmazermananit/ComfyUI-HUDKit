# Contributing to ComfyUI-HUDKit

Bug reports, sample workflows, documentation improvements, and focused pull requests
are welcome. For a bug, include your ComfyUI and Python versions, the node involved,
the error message, and a minimal workflow with private paths and media removed.
Do not include credentials, model checkpoints, or footage you cannot redistribute.

## Local checks

Use your ComfyUI Python environment, which already provides PyTorch:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
```

Synthetic tests do not require ComfyUI itself, checkpoints, or network access.
Optional example video generation also requires `av` (`python -m pip install av`).
Live SAM3 integration requires ComfyUI with native SAM3 support and a locally
installed checkpoint. Set `COMFYUI_PATH` and `HUDKIT_INPUT_VIDEO` before running
`examples/run_real_sam3.py`; optionally set `COMFYUI_MODELS_PATH` for external models.

Keep existing node IDs and socket types compatible with saved workflows. New
dependencies should have a clear need and a compatible redistribution license.

## License and attribution

Contributions are made under the project's MIT license. The bundled Tektur font
retains its SIL Open Font License. Include the source and license for any additional
third-party assets.
