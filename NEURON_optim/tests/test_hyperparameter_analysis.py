import json

import numpy as np

from hyperparameter_analyses.analyze_importance import analyze_stage, load_stage_data


def test_importance_analysis_loads_calibrated_values_and_writes_reports(tmp_path):
    run_dir = tmp_path / "run"
    study_dir = run_dir / "stage0_passive" / "seed_000"
    study_dir.mkdir(parents=True)
    keys = ["Epas", "CmSoma"]
    (study_dir / "manifest.json").write_text(
        json.dumps({"parameter_keys": keys}), encoding="utf-8"
    )
    for generation in (1, 2):
        epas = np.linspace(0.1, 0.9, 8)
        cm = np.linspace(0.2, 0.8, 8)
        population = np.column_stack([epas, cm])
        losses = (epas - 0.7) ** 2 + 0.1 * (cm - 0.4) ** 2
        np.savez_compressed(
            study_dir / f"population_generation_{generation:04d}.npz",
            population=population,
            losses=losses,
        )

    data = load_stage_data(run_dir, "stage0", max_records=0)
    assert data.normalized.shape == (16, 2)
    assert data.physical.shape == (16, 2)
    assert data.n_candidates_available == 16
    capped = load_stage_data(run_dir, "stage0", max_records=5)
    assert len(capped.losses) == 5

    output_dir = tmp_path / "analysis"
    summary = analyze_stage(data, output_dir)
    assert summary["n_records_analyzed"] == 16
    assert summary["parameters"][0]["parameter"] in keys
    assert (output_dir / "stage0_importance.json").exists()
    assert (output_dir / "stage0_importance.csv").exists()
    assert (output_dir / "stage0_importance.png").exists()
    assert (output_dir / "stage0_top_slices.png").exists()
    assert (output_dir / "stage0_history.png").exists()
    assert (output_dir / "stage0_contour.png").exists()
