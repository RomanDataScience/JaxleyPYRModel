from pathlib import Path

import numpy as np

from jaxley_refactored.data import TraceRecord, bucket_records
from jaxley_refactored.reporting import plot_epoch_traces
from jaxley_refactored.reporting.plots import _plot_sample_mask


def _record(trace_id: str, protocol: str, size: int) -> TraceRecord:
    time = np.arange(size, dtype=float) * 0.05
    return TraceRecord(
        cell_id="cell",
        trace_id=trace_id,
        protocol=protocol,
        voltage_mV=np.linspace(-70.0, -60.0, size),
        current_nA=np.zeros(size),
        time_ms=time,
        score_mask=np.ones(size, dtype=bool),
        dt_ms=0.05,
        initial_voltage_mV=-70.0,
        weight=0.5,
        metadata={},
        checksums={},
    )


def test_epoch_plot_contains_real_samples_and_updates_latest(tmp_path: Path):
    bucket = bucket_records(
        (
            _record("trace", "depolarizing_step", 10),
            _record("trace", "hyperpolarizing_pulse", 6),
        ),
        pad_to_longest=True,
    )[0]
    predictions = {bucket.key: bucket.observed_mV + 1.0}

    destination = plot_epoch_traces(
        tmp_path, (bucket,), predictions, epoch=3, loss=2.5
    )

    assert destination.name == "epoch_0003.png"
    assert destination.stat().st_size > 0
    assert (tmp_path / "latest.png").stat().st_size == destination.stat().st_size


def test_epoch_plot_accepts_experimental_style_preview(tmp_path: Path):
    bucket = bucket_records((_record("trace", "depolarizing_step", 10),))[0]
    predictions = {bucket.key: bucket.observed_mV + 1.0}

    destination = plot_epoch_traces(
        tmp_path,
        (bucket,),
        predictions,
        epoch=0,
        loss=1.0,
        experimental_alpha=0.6,
        experimental_linestyle="--",
        filename="preview.png",
    )

    assert destination.name == "preview.png"
    assert destination.stat().st_size > 0
    assert not (tmp_path / "latest.png").exists()


def test_hyperpolarizing_plot_hides_samples_before_400_ms():
    time_ms = np.array([0.0, 399.95, 400.0, 500.0, 650.0])

    mask = _plot_sample_mask("hyperpolarizing_pulse", time_ms)

    np.testing.assert_array_equal(mask, [False, False, True, True, True])
    np.testing.assert_array_equal(
        _plot_sample_mask("depolarizing_step", time_ms),
        np.ones(time_ms.shape, dtype=bool),
    )


def test_short_hyperpolarizing_plot_keeps_available_smoke_test_samples():
    time_ms = np.array([0.0, 50.0, 100.0])

    np.testing.assert_array_equal(
        _plot_sample_mask("hyperpolarizing_pulse", time_ms),
        np.ones(time_ms.shape, dtype=bool),
    )
