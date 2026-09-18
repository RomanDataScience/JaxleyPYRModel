from types import SimpleNamespace

from newjaxley_multiobjective.runner import _create_sampler


class _FakeCmaEsSampler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOptuna:
    samplers = SimpleNamespace(CmaEsSampler=_FakeCmaEsSampler)


def test_cma_es_sampler_uses_optunas_single_objective_sampler():
    config = SimpleNamespace(sampler="cma_es", population_size=40, seed=1234)

    sampler = _create_sampler(config, {}, _FakeOptuna)

    assert isinstance(sampler, _FakeCmaEsSampler)
    assert sampler.kwargs == {"popsize": 40, "seed": 1234}
