"""Packaging contract: PEP 621 metadata in pyproject.toml, no setup.py."""
import pathlib

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_setup_py_is_gone():
    assert not (ROOT / "setup.py").exists()


def test_pyproject_metadata():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    proj = meta["project"]
    assert proj["name"] == "gldpy-gpu"
    assert proj["readme"] == "README.md"
    assert any(d.startswith("numpy") for d in proj["dependencies"])
    assert "build-system" in meta


def test_version_matches_package():
    import gldpy_gpu as gldpy

    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert gldpy.__version__ == meta["project"]["version"]
