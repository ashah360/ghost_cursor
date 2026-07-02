"""Test harness for the standalone ghost_cursor repo.

In the Hermes tree, an autouse conftest fixture isolates HERMES_HOME to a
temp dir so persisted state (e.g. the handle table's JSON file) never leaks
between tests or from the developer's real ~/.hermes. Standalone, we provide
the same isolation here so `pytest` is hermetic — which is also what CI needs.
"""
import os
import sys
from pathlib import Path

import pytest

# Make `import plugins.ghost_cursor...` resolve to this directory's files by
# exposing them under a `plugins/ghost_cursor` package shim if needed. In the
# Hermes tree the real package is importable; standalone, tests import the
# modules directly, so we just ensure CWD is on the path.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _isolate_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a per-test temp dir so the handle-table JSON and
    any other persisted state are fresh every test and never touch the real
    ~/.hermes. Mirrors the Hermes-tree autouse fixture the plugin tests assume.
    """
    home = tmp_path / "hermes_home"
    (home / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home
