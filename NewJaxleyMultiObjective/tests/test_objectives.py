from types import SimpleNamespace

import numpy as np

from newjaxley_multiobjective.config import ObjectiveConfig
from newjaxley_multiobjective.objectives import trajectory_overlap


def test_trajectory_overlap_penalizes_shift_and_offset_without_mean_subtraction():
    time = np.arange(0.0, 10.0, 1.0)
    observed = np.sin(time)
    record = SimpleNamespace(
        time_ms=time,
        voltage_mV=observed,
        score_mask=np.ones(time.size, dtype=bool),
        metadata={"epoch_start_ms": 4.0, "epoch_stop_ms": 8.0},
    )
    config = ObjectiveConfig(overlap_sigma_mV=0.5)
    exact = trajectory_overlap(record, observed, config)
    offset = trajectory_overlap(record, observed + 2.0, config)

    assert exact["loss"] == 0.0
    assert offset["loss"] > exact["loss"]
