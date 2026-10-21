"""Asset inputs must not read images outside ComfyUI's input directory."""
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nodes.asset_pack import _scan_folder


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    base = tmp_path / "input"
    base.mkdir()
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_input_directory=lambda: str(base)))
    return base


def test_loads_normal_assets_in_order(input_dir):
    folder = input_dir / "hud"
    folder.mkdir()
    for name in ("frame.png", "target.webp", "ignore.txt"):
        (folder / name).touch()
    with patch("nodes.asset_pack._load_pair_from_file", side_effect=lambda p: p) as load:
        frames, targets = _scan_folder("hud")
    assert frames == [str(folder / "frame.png")]
    assert targets == [str(folder / "target.webp")]
    assert load.call_count == 2


@pytest.mark.parametrize("folder", ["../input-other", "../input-other/../input-other"])
def test_rejects_parent_traversal_and_sibling_prefix(input_dir, folder):
    outside = input_dir.parent / "input-other"
    outside.mkdir()
    (outside / "frame.png").touch()
    with patch("nodes.asset_pack._load_pair_from_file") as load:
        assert _scan_folder(folder) == ([], [])
    load.assert_not_called()


def test_rejects_absolute_folder(input_dir):
    with patch("nodes.asset_pack._load_pair_from_file") as load:
        assert _scan_folder(str(input_dir)) == ([], [])
    load.assert_not_called()


@pytest.mark.parametrize("link_folder", [True, False])
def test_rejects_symlinks_outside_input(input_dir, link_folder):
    outside = input_dir.parent / "outside"
    outside.mkdir()
    image = outside / "frame.png"
    image.touch()
    folder = input_dir / "hud"
    try:
        if link_folder:
            folder.symlink_to(outside, target_is_directory=True)
        else:
            folder.mkdir()
            (folder / "frame.png").symlink_to(image)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    with patch("nodes.asset_pack._load_pair_from_file") as load:
        assert _scan_folder("hud") == ([], [])
    load.assert_not_called()


def test_allows_symlink_within_input(input_dir):
    real = input_dir / "assets"
    real.mkdir()
    (real / "frame.png").touch()
    try:
        (input_dir / "hud").symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    with patch("nodes.asset_pack._load_pair_from_file", side_effect=lambda p: p):
        assert _scan_folder("hud") == ([str(real / "frame.png")], [])
