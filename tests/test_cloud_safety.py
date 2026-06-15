"""Cloud / Registry security self-audit.

Greps the shipped package source for patterns banned or flagged by the Comfy
Registry security standards (eval/exec, runtime pip/subprocess, network, arbitrary
command exec). Fails if any appear. See docs/cloud_compatibility.md §5.

Runnable as: python tests/test_cloud_safety.py   (also pytest-discoverable).
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Directories that ship in the package (per .comfyignore: tests/ excluded).
SCAN_DIRS = ["yea_hud", "nodes"]
SCAN_ROOT_FILES = ["__init__.py"]

# (regex, human reason). Strict CODE forms only (import X / X.call / call()),
# so prose like "no subprocess" in a docstring does NOT false-positive.
BANNED = [
    (r"\beval\s*\(", "eval() is banned (RCE vector)"),
    (r"\bexec\s*\(", "exec() is banned (RCE vector)"),
    (r"\bimport\s+subprocess\b|\bsubprocess\.", "subprocess is banned (runtime command exec)"),
    (r"\bos\.system\s*\(", "os.system() is banned"),
    (r"\.[Pp]open\s*\(", "Popen is banned"),
    (r"pip\s+install", "runtime pip install is banned"),
    (r"\bimport\s+requests\b|\brequests\.(get|post|put|delete|head)\s*\(", "network (requests) not allowed"),
    (r"\bimport\s+urllib\b|urllib\.request", "network (urllib) not allowed"),
    (r"\bimport\s+socket\b|\bsocket\.socket\s*\(", "raw sockets not allowed"),
    (r"\bimport\s+aiohttp\b|\baiohttp\.", "custom HTTP server/client not allowed"),
    (r"\bhttp\.client\b", "network (http.client) not allowed"),
]


def _iter_py_files():
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if fn.endswith(".py"):
                    yield os.path.join(dirpath, fn)
    for fn in SCAN_ROOT_FILES:
        p = os.path.join(ROOT, fn)
        if os.path.exists(p):
            yield p


def test_no_banned_patterns():
    violations = []
    for path in _iter_py_files():
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue  # skip pure comment lines
                for pat, reason in BANNED:
                    if re.search(pat, line):
                        rel = os.path.relpath(path, ROOT)
                        violations.append(f"{rel}:{lineno}: {reason} -> {line.strip()}")
    assert not violations, "Cloud-safety violations found:\n" + "\n".join(violations)


def test_no_network_imports_resolve():
    # Importing the pure types module must not pull torch/PIL/network at import time.
    import importlib
    mod = importlib.import_module("yea_hud.types")
    assert hasattr(mod, "Detection") and hasattr(mod, "DEFAULTS")


if __name__ == "__main__":
    import sys
    failed = 0
    for name in ("test_no_banned_patterns", "test_no_network_imports_resolve"):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"ERROR {name}: {exc!r}")
    print("-" * 60)
    print(f"cloud_safety: {'OK' if failed == 0 else str(failed) + ' FAILED'}")
    sys.exit(1 if failed else 0)
