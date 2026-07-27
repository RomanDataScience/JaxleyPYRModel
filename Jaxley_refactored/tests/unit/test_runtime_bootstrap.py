import os

from jaxley_refactored.config.schema import RuntimeSpec
from jaxley_refactored.runtime.bootstrap import configure_environment


def test_cpu_is_pinned_but_gpu_uses_vendor_auto_discovery(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    configure_environment(RuntimeSpec(backend="cpu"))
    assert os.environ["JAX_PLATFORMS"] == "cpu"

    configure_environment(RuntimeSpec(backend="gpu"))
    assert "JAX_PLATFORMS" not in os.environ
