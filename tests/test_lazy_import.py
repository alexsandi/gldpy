import sys

import pytest


def test_cpu_gld_importable_without_touching_gpu_module():
    import gldpy_gpu as gldpy

    assert hasattr(gldpy, "GLD")


def test_gpu_attribute_lazily_loads():
    pytest.importorskip("torch")
    import gldpy_gpu as gldpy

    assert gldpy.gpu.GLD is not None
    assert "gldpy_gpu.gpu" in sys.modules


def test_unknown_attribute_raises():
    import gldpy_gpu as gldpy

    with pytest.raises(AttributeError):
        gldpy.does_not_exist
