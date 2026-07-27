from pathlib import Path

import numpy as np
import pytest

from jaxley_refactored.fitting.checkpoints import CheckpointManager
from jaxley_refactored.fitting.optimizer import AdamState


def test_atomic_checkpoint_roundtrip_and_best(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "compatible")
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

    with pytest.raises(ValueError, match="incompatible"):
        CheckpointManager(tmp_path, "different").load()
