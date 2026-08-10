from pathlib import Path

import numpy as np
import pytest

from jaxley_refactored.fitting.checkpoints import CheckpointManager
from jaxley_refactored.fitting.optimizer import AdamState


def test_atomic_checkpoint_roundtrip_and_best(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "compatible", ("conductance",))
    optimizer = AdamState(
        step=3,
        first_moment=np.asarray([0.1]),
        second_moment=np.asarray([0.2]),
        current_learning_rate=0.0125,
    )
    manager.save(
        epoch=2,
        normalized=np.asarray([0.4]),
        optimizer=optimizer,
        best_normalized=np.asarray([0.3]),
        best_loss=1.25,
        is_best=True,
    )

    restored = manager.load()
    assert restored["epoch"] == 2
    assert restored["optimizer"].step == 3
    assert restored["optimizer"].current_learning_rate == 0.0125
    np.testing.assert_allclose(restored["best_normalized"], [0.3])
    assert (tmp_path / "best.npz").is_file()
    assert (tmp_path / "best.json").is_file()

    with np.load(tmp_path / "latest.npz", allow_pickle=False) as arrays:
        assert arrays["parameter_names"].tolist() == ["conductance"]

    import json

    metadata = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert metadata["parameters_normalized"] == {"conductance": 0.4}

    with pytest.raises(ValueError, match="incompatible"):
        CheckpointManager(tmp_path, "different").load()


def test_checkpoint_load_reorders_all_parameter_state_by_name(tmp_path: Path):
    saved = CheckpointManager(tmp_path, "compatible", ("b", "a"))
    saved.save(
        epoch=0,
        normalized=np.asarray([0.2, 0.1]),
        optimizer=AdamState(
            step=1,
            first_moment=np.asarray([2.0, 1.0]),
            second_moment=np.asarray([20.0, 10.0]),
        ),
        best_normalized=np.asarray([0.4, 0.3]),
        best_loss=1.0,
    )

    restored = CheckpointManager(tmp_path, "compatible", ("a", "b")).load()

    np.testing.assert_allclose(restored["normalized"], [0.1, 0.2])
    np.testing.assert_allclose(restored["best_normalized"], [0.3, 0.4])
    np.testing.assert_allclose(restored["optimizer"].first_moment, [1.0, 2.0])
    np.testing.assert_allclose(restored["optimizer"].second_moment, [10.0, 20.0])
