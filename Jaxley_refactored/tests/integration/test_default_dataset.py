from pathlib import Path

import numpy as np

from jaxley_refactored.config import load_config
from jaxley_refactored.data import SegmentedTraceLoader, bucket_records, weight_records


PROJECT = Path(__file__).resolve().parents[2]


def test_default_cell_loads_all_segmented_traces():
    config = load_config(PROJECT / "configs/runtimes/cpu_x64.yaml")
    records = SegmentedTraceLoader().load(config.dataset)
    weighted = weight_records(
        records,
        aggregation=config.fit.aggregation,
        protocol_weights=config.fit.protocol_weights,
    )
    buckets = bucket_records(weighted)

    assert len(records) == 8
    assert {(bucket.n_steps, len(bucket.records)) for bucket in buckets} == {
        (13_000, 4),
        (24_000, 4),
    }
    assert np.isclose(sum(record.weight for record in weighted), 1.0)
    assert all(np.isclose(record.dt_ms, 0.05) for record in records)
