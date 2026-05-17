# GhostKV Lab — github.com/manishklach/ghostkv-lab
# Patent: IN 202641062451
"""Pytest path setup for repo-root experiment imports."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
