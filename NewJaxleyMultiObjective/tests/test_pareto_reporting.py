from types import SimpleNamespace

from newjaxley_multiobjective.pareto import annotate_front
from newjaxley_multiobjective.runner import _storage_url


def trial(number, values):
    return SimpleNamespace(number=number, values=values, params={"x": number}, state="COMPLETE")


def test_front_annotations_identify_per_objective_winners():
    labels = ("a", "b")
    rows = annotate_front(
        (trial(1, (1.0, 3.0)), trial(2, (2.0, 1.0))), labels, 1e-12
    )

    assert rows[0]["best_for_objectives"] == ["a"]
    assert rows[1]["best_for_objectives"] == ["b"]
    assert rows[0]["objective_ranks"] == {"a": 1, "b": 2}


def test_auto_storage_is_inside_run_directory(tmp_path):
    assert _storage_url("auto", tmp_path).startswith("sqlite:///")
    assert _storage_url("auto", tmp_path).endswith("/study.db")
