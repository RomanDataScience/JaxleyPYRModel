"""Analyze calibrated-parameter importance for the optimization stages.

The optimizer stores CMA-ES populations rather than an Optuna Study.  This
module reconstructs an in-memory Optuna study from those evaluated candidates,
then applies Optuna's fANOVA importance evaluator.  The analysis is performed
in normalized coordinates so parameter units do not affect the estimator, but
all reports and plots also include the corresponding physical values.

Example
-------
From ``NEURON_optim``::

    python -m hyperparameter_analyses.analyze_importance \
        --config configs/.smoke-config.DDZfoG \
        --run-dir runs/smoke_full_workers16 \
        --output-dir hyperparameter_analyses/smoke_full_workers16

The output contains one JSON/CSV report and three PNG figures per available
stage, plus a combined summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.importance import (
    FanovaImportanceEvaluator,
    PedAnovaImportanceEvaluator,
    get_param_importances,
)

from neuron_optim.config import load_config
from neuron_optim.parameters import KINETIC, PASSIVE, make_parameter_space


optuna.logging.set_verbosity(optuna.logging.WARNING)


STAGE_NAMES = {
    "stage0": "passive",
    "stage1": "hyperpolarizing",
    "stage2": "depolarizing",
}
STAGE_DIRECTORIES = {
    "stage0": "stage0_passive",
    "stage1": "stage1_hyper",
    "stage2": "stage2_depolarizing",
}


@dataclass
class StageData:
    """Evaluated candidates loaded for one optimization stage."""

    stage: str
    parameter_keys: tuple[str, ...]
    normalized: np.ndarray
    physical: np.ndarray
    losses: np.ndarray
    generations: np.ndarray
    study_ids: np.ndarray
    basin_ids: np.ndarray
    n_files: int
    n_candidates_available: int


def _fallback_parameter_keys(stage: str, config_path: Path | None) -> tuple[str, ...]:
    if stage == "stage0":
        return tuple(make_parameter_space(include=PASSIVE).keys)
    if config_path is None:
        return tuple(make_parameter_space().keys)
    return tuple(load_config(config_path).parameters.keys)


def _study_directories(root: Path, stage: str) -> list[Path]:
    if stage == "stage2":
        return sorted(path for path in root.glob("b*/seed_*") if path.is_dir())
    return sorted(path for path in root.glob("seed_*") if path.is_dir())


def _study_id(study_dir: Path, stage: str) -> tuple[str, str]:
    if stage == "stage2":
        return f"{study_dir.parent.name}/{study_dir.name}", study_dir.parent.name
    return study_dir.name, ""


def _manifest_keys(study_dir: Path) -> tuple[str, ...] | None:
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = payload.get("parameter_keys")
    return tuple(str(key) for key in keys) if keys else None


def _select_indices(losses: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Keep the best candidates and a deterministic exploration sample."""
    count = len(losses)
    if budget < 0:
        return np.arange(count, dtype=int)
    if budget == 0:
        return np.empty(0, dtype=int)
    if budget >= count:
        return np.arange(count, dtype=int)
    ranked = np.argsort(losses, kind="stable")
    keep_top = min(3, budget)
    selected = list(ranked[:keep_top])
    remaining = ranked[keep_top:]
    if len(selected) < budget and len(remaining):
        extra = rng.choice(remaining, size=budget - len(selected), replace=False)
        selected.extend(int(index) for index in extra)
    return np.asarray(sorted(set(selected)), dtype=int)


def load_stage_data(
    run_dir: str | Path,
    stage: str,
    *,
    config_path: str | Path | None = None,
    max_records: int = 50_000,
    seed: int = 13,
) -> StageData:
    """Load evaluated population candidates for one stage.

    ``max_records`` is a deterministic per-file stratified cap.  It keeps the
    analysis practical for a full Stage 2 run while retaining the best three
    candidates from every saved population and sampling the remainder.
    """
    if stage not in STAGE_DIRECTORIES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_DIRECTORIES)}")
    root = Path(run_dir) / STAGE_DIRECTORIES[stage]
    study_dirs = _study_directories(root, stage)
    population_paths = [
        path
        for study_dir in study_dirs
        for path in sorted(study_dir.glob("population_generation_*.npz"))
    ]
    if not population_paths:
        raise FileNotFoundError(f"No population_generation_*.npz files found under {root}")

    fallback_keys = _fallback_parameter_keys(
        stage, Path(config_path).resolve() if config_path is not None else None
    )
    parameter_keys: tuple[str, ...] | None = None
    normalized_parts: list[np.ndarray] = []
    loss_parts: list[np.ndarray] = []
    generation_parts: list[np.ndarray] = []
    study_parts: list[np.ndarray] = []
    basin_parts: list[np.ndarray] = []
    available = 0
    rng = np.random.default_rng(seed)

    for file_index, population_path in enumerate(population_paths):
        study_dir = population_path.parent
        keys = _manifest_keys(study_dir) or fallback_keys
        if parameter_keys is None:
            parameter_keys = keys
        elif keys != parameter_keys:
            raise ValueError(
                f"Parameter keys differ in {population_path}: {keys} != {parameter_keys}"
            )
        generation = int(population_path.stem.rsplit("_", 1)[1])
        with np.load(population_path, allow_pickle=False) as values:
            population = np.asarray(values["population"], dtype=float)
            losses = np.asarray(values["losses"], dtype=float)
        if population.ndim != 2 or population.shape[1] != len(parameter_keys):
            raise ValueError(
                f"Unexpected population shape {population.shape} in {population_path}; "
                f"expected (*, {len(parameter_keys)})"
            )
        if len(population) != len(losses):
            raise ValueError(f"Population/loss length mismatch in {population_path}")
        finite = np.isfinite(losses) & np.all(np.isfinite(population), axis=1)
        population, losses = population[finite], losses[finite]
        available += len(losses)
        if not len(losses):
            continue
        if max_records > 0:
            base_budget, remainder = divmod(max_records, len(population_paths))
            file_budget = base_budget + int(file_index < remainder)
        else:
            file_budget = -1
        selected = _select_indices(losses, file_budget, rng)
        if not len(selected):
            continue
        population = population[selected]
        losses = losses[selected]
        space = make_parameter_space(include=parameter_keys)
        study_id, basin_id = _study_id(study_dir, stage)
        normalized_parts.append(np.clip(population, 0.0, 1.0))
        loss_parts.append(losses)
        generation_parts.append(np.full(len(losses), generation, dtype=int))
        study_parts.append(np.full(len(losses), study_id, dtype=object))
        basin_parts.append(np.full(len(losses), basin_id, dtype=object))

    if parameter_keys is None or not normalized_parts:
        raise ValueError(f"No finite candidates found under {root}")
    normalized = np.concatenate(normalized_parts, axis=0)
    losses = np.concatenate(loss_parts)
    space = make_parameter_space(include=parameter_keys)
    return StageData(
        stage=stage,
        parameter_keys=parameter_keys,
        normalized=normalized,
        physical=space.physical(normalized),
        losses=losses,
        generations=np.concatenate(generation_parts),
        study_ids=np.concatenate(study_parts),
        basin_ids=np.concatenate(basin_parts),
        n_files=len(population_paths),
        n_candidates_available=available,
    )


def _parameter_group(name: str) -> str:
    if name in PASSIVE:
        return "passive"
    if name in KINETIC:
        return "kinetic"
    return "conductance"


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    x_order = np.argsort(x, kind="stable")
    y_order = np.argsort(y, kind="stable")
    x_rank = np.empty(len(x), dtype=float)
    y_rank = np.empty(len(y), dtype=float)
    x_rank[x_order] = np.arange(len(x), dtype=float)
    y_rank[y_order] = np.arange(len(y), dtype=float)
    correlation = np.corrcoef(x_rank, y_rank)[0, 1]
    return float(abs(correlation)) if np.isfinite(correlation) else 0.0


def _make_optuna_study(data: StageData, varying: list[str]) -> optuna.study.Study:
    study = optuna.create_study(direction="minimize", study_name=data.stage)
    distributions = {name: FloatDistribution(0.0, 1.0) for name in varying}
    index = {name: position for position, name in enumerate(data.parameter_keys)}
    for row, loss in zip(data.normalized, data.losses, strict=True):
        params = {
            name: float(np.clip(row[index[name]], 0.0, 1.0))
            for name in varying
        }
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=float(loss),
            )
        )
    return study


def _importance_values(
    study: optuna.study.Study, varying: list[str]
) -> tuple[dict[str, float], dict[str, float], str, list[str]]:
    warnings: list[str] = []
    fanova: dict[str, float] = {}
    pedanova: dict[str, float] = {}
    if len(study.trials) < 2 or not varying:
        warnings.append("Not enough varying parameters or candidates for Optuna importance.")
        return fanova, pedanova, "none", warnings
    try:
        fanova = get_param_importances(
            study,
            evaluator=FanovaImportanceEvaluator(seed=13),
            params=varying,
        )
    except Exception as exc:  # pragma: no cover - depends on estimator edge cases
        warnings.append(f"fANOVA failed: {exc}")
    try:
        pedanova = get_param_importances(
            study,
            evaluator=PedAnovaImportanceEvaluator(target_quantile=0.1),
            params=varying,
        )
    except Exception as exc:  # pragma: no cover - depends on estimator edge cases
        warnings.append(f"PED-ANOVA failed: {exc}")
    method = "Optuna fANOVA" if fanova else "Optuna PED-ANOVA" if pedanova else "absolute rank correlation"
    return fanova, pedanova, method, warnings


def _importance_rows(data: StageData) -> tuple[list[dict], str, list[str]]:
    varying = [
        name
        for index, name in enumerate(data.parameter_keys)
        if len(np.unique(data.normalized[:, index])) > 1
        and np.ptp(data.normalized[:, index]) > 1e-12
    ]
    study = _make_optuna_study(data, varying)
    fanova, pedanova, method, warnings = _importance_values(study, varying)
    best_index = int(np.argmin(data.losses))
    rows = []
    for index, name in enumerate(data.parameter_keys):
        values = data.physical[:, index]
        absolute_spearman = _rank_correlation(data.normalized[:, index], data.losses)
        if method == "Optuna fANOVA":
            importance = fanova.get(name, 0.0)
        elif method == "Optuna PED-ANOVA":
            importance = pedanova.get(name, 0.0)
        else:
            importance = absolute_spearman
        rows.append({
            "parameter": name,
            "group": _parameter_group(name),
            "importance": float(importance),
            "importance_method": method,
            "fanova_importance": float(fanova.get(name, 0.0)),
            "pedanova_importance": float(pedanova.get(name, 0.0)),
            "absolute_spearman": absolute_spearman,
            "physical_min": float(np.min(values)),
            "physical_max": float(np.max(values)),
            "physical_mean": float(np.mean(values)),
            "physical_std": float(np.std(values)),
            "best_candidate_value": float(values[best_index]),
            "unique_values": int(len(np.unique(data.normalized[:, index]))),
            "varies": name in varying,
        })
    rows.sort(key=lambda row: (-row["importance"], row["parameter"]))
    return rows, method, warnings


def _plot_importance(
    rows: list[dict], output: Path, stage: str, n_records: int, method: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = rows[: min(20, len(rows))]
    shown = list(reversed(shown))
    colors = {"passive": "#4C78A8", "conductance": "#F58518", "kinetic": "#54A24B"}
    fig, axis = plt.subplots(figsize=(9, max(4.5, 0.34 * len(shown) + 1.3)))
    axis.barh(
        [row["parameter"] for row in shown],
        [row["importance"] for row in shown],
        color=[colors[row["group"]] for row in shown],
    )
    axis.set_xlabel(f"{method} importance (normalized)")
    axis.set_title(f"{STAGE_NAMES[stage].title()} stage parameter importance (n={n_records:,})")
    axis.grid(axis="x", alpha=0.25)
    for position, row in enumerate(shown):
        axis.text(row["importance"], position, f" {row['importance']:.3f}", va="center")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[group], label=group)
        for group in ("passive", "conductance", "kinetic")
    ]
    axis.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_slices(data: StageData, rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rows[: min(9, len(rows))]
    columns = 3
    rows_count = max(1, math.ceil(len(top) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(13, 3.2 * rows_count), squeeze=False)
    delta = data.losses - np.min(data.losses)
    positive = delta[delta > 0]
    floor = max(float(np.min(positive)) if len(positive) else 1e-12, 1e-12)
    for axis, row in zip(axes.flat, top, strict=False):
        index = data.parameter_keys.index(row["parameter"])
        axis.scatter(data.physical[:, index], delta, s=10, alpha=0.45, linewidths=0)
        axis.set_yscale("symlog", linthresh=floor)
        axis.set_xlabel(row["parameter"])
        axis.set_ylabel("loss - best loss")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(top):]:
        axis.set_visible(False)
    fig.suptitle(f"Top {len(top)} parameter slices: {STAGE_NAMES[data.stage].title()}", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_history(data: StageData, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations = sorted(int(generation) for generation in np.unique(data.generations))
    best, median, low, high = [], [], [], []
    for generation in generations:
        values = data.losses[data.generations == generation]
        best.append(float(np.min(values)))
        median.append(float(np.median(values)))
        low.append(float(np.percentile(values, 10)))
        high.append(float(np.percentile(values, 90)))
    epsilon = max(1e-12, min((value for value in best + median if value > 0), default=1e-12) * 1e-3)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.fill_between(generations, np.log10(np.asarray(low) + epsilon),
                      np.log10(np.asarray(high) + epsilon), alpha=0.2, label="10–90 percentile")
    axis.plot(generations, np.log10(np.asarray(median) + epsilon), marker="o", label="median")
    axis.plot(generations, np.log10(np.asarray(best) + epsilon), marker="o", label="best")
    axis.set_xlabel("Saved generation")
    axis.set_ylabel(f"log10(loss + {epsilon:.2g})")
    axis.set_title(f"{STAGE_NAMES[data.stage].title()} optimization history")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_contour(
    data: StageData,
    rows: list[dict],
    output: Path,
    top_k: int = 4,
) -> list[str]:
    """Plot Optuna-style pairwise loss contours for the most important params."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    top_k = max(2, int(top_k))
    parameters = [row["parameter"] for row in rows if row["varies"]][:top_k]
    parameters = parameters[:top_k]
    if len(parameters) < 2:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.text(0.5, 0.5, "At least two varying parameters are required.",
                  ha="center", va="center")
        axis.set_axis_off()
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return parameters

    parameter_indices = {name: data.parameter_keys.index(name) for name in parameters}
    delta = data.losses - np.min(data.losses)
    positive = delta[delta > 0]
    epsilon = max(float(np.min(positive)) if len(positive) else 1e-12, 1e-12)
    score = np.log10(delta + epsilon)
    finite_score = score[np.isfinite(score)]
    vmin, vmax = float(np.min(finite_score)), float(np.max(finite_score))
    if vmin == vmax:
        vmax = vmin + 1.0
    levels = np.linspace(vmin, vmax, 9)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis_r"
    size = max(8.0, 3.0 * len(parameters))
    fig, axes = plt.subplots(
        len(parameters), len(parameters), figsize=(size, size), squeeze=False,
        constrained_layout=True,
    )
    contour_set = None
    for row_index, y_name in enumerate(parameters):
        for column_index, x_name in enumerate(parameters):
            axis = axes[row_index, column_index]
            if row_index == column_index:
                index = parameter_indices[x_name]
                axis.hist(data.physical[:, index], bins=15, color="#4C78A8", alpha=0.8)
                axis.set_xlabel(x_name)
                axis.set_ylabel("count")
                axis.grid(alpha=0.2)
                continue
            if row_index < column_index:
                axis.set_visible(False)
                continue
            x = data.physical[:, parameter_indices[x_name]]
            y = data.physical[:, parameter_indices[y_name]]
            try:
                if len(np.unique(x)) < 3 or len(np.unique(y)) < 3:
                    raise ValueError("insufficient unique values for a contour")
                contour_set = axis.tricontourf(x, y, score, levels=levels, cmap=cmap, norm=norm)
            except (ValueError, RuntimeError):
                axis.scatter(x, y, c=score, cmap=cmap, norm=norm, s=12, alpha=0.65)
            axis.scatter(x, y, c=score, cmap=cmap, norm=norm, s=5, alpha=0.25,
                         linewidths=0)
            axis.set_xlabel(x_name)
            axis.set_ylabel(y_name)
            axis.grid(alpha=0.2)
    if contour_set is None:
        contour_set = ScalarMappable(norm=norm, cmap=cmap)
        contour_set.set_array(score)
    fig.colorbar(
        contour_set, ax=axes.ravel().tolist(), shrink=0.75,
        label="log10(loss - best loss + epsilon)",
    )
    fig.suptitle(
        f"{STAGE_NAMES[data.stage].title()} parameter contours "
        f"(top {len(parameters)} by importance)", y=0.995,
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return parameters


def analyze_stage(
    data: StageData,
    output_dir: str | Path,
    *,
    contour_top_k: int = 4,
) -> dict:
    """Write reports and figures for a loaded stage and return the summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, method, warnings = _importance_rows(data)
    group_importance = {
        group: float(sum(row["importance"] for row in rows if row["group"] == group))
        for group in ("passive", "conductance", "kinetic")
    }
    summary = {
        "stage": data.stage,
        "stage_name": STAGE_NAMES[data.stage],
        "n_records_analyzed": int(len(data.losses)),
        "n_candidates_available": int(data.n_candidates_available),
        "n_population_files": data.n_files,
        "n_studies": int(len(np.unique(data.study_ids))),
        "n_generations": int(len(np.unique(data.generations))),
        "parameter_keys": list(data.parameter_keys),
        "loss_min": float(np.min(data.losses)),
        "loss_median": float(np.median(data.losses)),
        "loss_max": float(np.max(data.losses)),
        "importance_method": f"{method} on normalized [0, 1] parameter coordinates",
        "secondary_method": "Optuna PED-ANOVA and absolute rank correlation",
        "group_importance": group_importance,
        "contour_parameters": [],
        "contour_loss_scale": "log10(loss - best loss + epsilon)",
        "warnings": warnings,
        "parameters": rows,
    }
    (output_dir / f"{data.stage}_importance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / f"{data.stage}_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _plot_importance(
        rows, output_dir / f"{data.stage}_importance.png", data.stage,
        len(data.losses), method,
    )
    _plot_slices(data, rows, output_dir / f"{data.stage}_top_slices.png")
    _plot_history(data, output_dir / f"{data.stage}_history.png")
    contour_parameters = _plot_contour(
        data, rows, output_dir / f"{data.stage}_contour.png", top_k=contour_top_k,
    )
    summary["contour_parameters"] = contour_parameters
    (output_dir / f"{data.stage}_importance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_analysis(
    config_path: str | Path,
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    stages: Iterable[str] = ("stage0", "stage1", "stage2"),
    max_records: int = 50_000,
    seed: int = 13,
    contour_top_k: int = 4,
) -> dict:
    """Run all requested stage analyses that have saved population files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    skipped = {}
    for stage in stages:
        try:
            data = load_stage_data(
                run_dir,
                stage,
                config_path=config_path,
                max_records=max_records,
                seed=seed,
            )
            summaries[stage] = analyze_stage(
                data, output_dir, contour_top_k=contour_top_k,
            )
        except FileNotFoundError as exc:
            skipped[stage] = str(exc)
    combined = {
        "config": str(Path(config_path).resolve()),
        "run_dir": str(Path(run_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "max_records": max_records,
        "seed": seed,
        "stages": summaries,
        "skipped": skipped,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stages", nargs="+", choices=sorted(STAGE_DIRECTORIES), default=list(STAGE_DIRECTORIES))
    parser.add_argument(
        "--max-records", type=int, default=50_000,
        help="Maximum candidates to analyze; 0 keeps every finite candidate (default: 50000).",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--contour-top-k", type=int, default=4,
        help="Number of most important varying parameters in each contour grid (default: 4).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path("hyperparameter_analyses") / args.run_dir.name
    result = run_analysis(
        args.config,
        args.run_dir,
        output_dir,
        stages=args.stages,
        max_records=args.max_records,
        seed=args.seed,
        contour_top_k=args.contour_top_k,
    )
    for stage, summary in result["stages"].items():
        print(
            f"{stage}: analyzed {summary['n_records_analyzed']} candidates, "
            f"{summary['n_studies']} studies; wrote {output_dir}"
        )
    for stage, reason in result["skipped"].items():
        print(f"{stage}: skipped ({reason})")


if __name__ == "__main__":
    main()
