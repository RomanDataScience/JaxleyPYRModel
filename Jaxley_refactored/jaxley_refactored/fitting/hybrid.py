"""Serial CMA-ES -> Adam -> backtracking hybrid orchestration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from jaxley_refactored.parameters import ProjectedBoxSpace
from jaxley_refactored.reporting import plot_epoch_traces

from .checkpoints import CheckpointManager
from .global_search import CMAES
from .global_search.checkpoints import CMACheckpoint
from .initialization import initial_normalized_values
from .trainer import Trainer


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _stage_fit(base_fit, stage):
    line_search = replace(
        base_fit.optimizer.line_search,
        enabled=stage.backtracking,
    )
    optimizer = replace(
        base_fit.optimizer,
        epochs=stage.epochs,
        learning_rate=stage.learning_rate,
        gradient_clip_norm=stage.gradient_clip_norm,
        line_search=line_search,
    )
    return replace(base_fit, optimizer=optimizer)


def run_hybrid(
    *,
    model,
    training_buckets,
    validation_buckets,
    config,
    output,
    compatibility_hash: str,
) -> dict:
    search = config.search
    specs = model.parameterizer.specs
    names = tuple(spec.name for spec in specs)
    requested = search.global_search.parameter_names or names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"CMA parameter subset contains unknown/unfitted names: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("CMA parameter subset contains duplicate names.")
    subset = np.asarray([names.index(name) for name in requested], dtype=int)
    base = initial_normalized_values(model, config.fit, config.runtime)

    evaluator = Trainer(
        model, training_buckets, config.protocol, config.fit, config.runtime, None
    )
    evaluator.install_signal_handlers()
    cma_directory = output.path / "global"
    cma_checkpoint = CMACheckpoint(
        cma_directory / "checkpoints", compatibility_hash
    )
    cma = cma_checkpoint.load(seed=config.runtime.seed)
    if cma is None:
        cma = CMAES(
            base[subset],
            sigma=search.global_search.sigma0,
            seed=config.runtime.seed,
            population_size=search.global_search.population_size,
        )

    candidate_path = cma_directory / "candidates.jsonl"
    candidates: list[tuple[float, np.ndarray, str]] = []
    if candidate_path.is_file():
        with candidate_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") == "ok":
                    candidates.append(
                        (float(row["training_loss"]), np.asarray(row["normalized"]), row["candidate_id"])
                    )

    for generation in range(cma.state.generation, search.global_search.generations):
        population = cma.ask()
        losses = []
        generation_best = None
        for index, partial in enumerate(population):
            full = base.copy()
            full[subset] = partial
            candidate_id = f"g{generation:04d}-c{index:04d}"
            status = "ok"
            error_message = None
            component_losses = {}
            try:
                result = evaluator.evaluate(full, gradient=False)
                loss = float(result[0])
                component_losses = result[4]
                if not np.isfinite(loss):
                    raise FloatingPointError("nonfinite objective")
                if generation_best is None or loss < generation_best[0]:
                    generation_best = (loss, result[3])
            except Exception as error:
                loss = search.global_search.invalid_loss
                status = f"invalid:{type(error).__name__}"
                error_message = str(error)
            losses.append(loss)
            row = {
                "candidate_id": candidate_id,
                "generation": generation,
                "population_index": index,
                "training_loss": loss,
                "component_losses": component_losses,
                "status": status,
                "error": error_message,
                "normalized": full.tolist(),
                "physical": np.asarray(evaluator.space.physical(full)).tolist(),
            }
            _append_jsonl(candidate_path, row)
            if status == "ok":
                candidates.append((loss, full, candidate_id))
        cma.tell(population, losses)
        cma_plot_every = search.reporting.cma_plot_every_generations
        if (
            generation_best is not None
            and cma_plot_every
            and (generation + 1) % cma_plot_every == 0
        ):
            plot_epoch_traces(
                cma_directory / "plots",
                training_buckets,
                generation_best[1],
                epoch=generation,
                loss=generation_best[0],
                experimental_alpha=0.6,
            )
        _append_jsonl(
            cma_directory / "generations.jsonl",
            {
                "generation": generation,
                "best_loss": float(np.min(losses)),
                "median_loss": float(np.median(losses)),
                "sigma": cma.state.sigma,
                "evaluations": cma.state.evaluations,
            },
        )
        if (
            cma.state.generation
            % search.global_search.checkpoint_every_generations
            == 0
        ):
            cma_checkpoint.save(cma)
        if evaluator.stop_requested:
            cma_checkpoint.save(cma)
            raise InterruptedError("Hybrid search interrupted after CMA checkpoint.")

    if not candidates:
        raise RuntimeError("CMA-ES produced no valid candidates.")
    candidates.sort(key=lambda item: item[0])
    elites = candidates[: search.global_search.elites]

    exploration_fit = _stage_fit(config.fit, search.local_exploration)
    explored = []
    local_trainer = evaluator

    def stage_report(stage_dir):
        def report(metrics, predictions):
            _append_jsonl(stage_dir / "metrics.jsonl", metrics)
            every = search.reporting.adam_plot_every_epochs
            if every and (metrics["epoch"] + 1) % every == 0:
                plot_epoch_traces(
                    stage_dir / "plots",
                    training_buckets,
                    predictions,
                    epoch=metrics["epoch"],
                    loss=metrics["loss"],
                    experimental_alpha=0.6,
                )

        return report

    for _loss, initial, candidate_id in elites:
        stage_dir = output.path / "local_exploration" / candidate_id
        checkpoint = CheckpointManager(
            stage_dir / "checkpoints",
            f"{compatibility_hash}:exploration:{candidate_id}",
        )
        local_trainer.configure_optimizer(exploration_fit, checkpoint)
        result = local_trainer.train(
            stage_report(stage_dir),
            initial_normalized=initial,
        )
        if result.stopped_by_signal:
            raise InterruptedError("Hybrid search interrupted during Adam exploration.")
        best = np.asarray(local_trainer.space.normalize(result.best_parameters))
        explored.append((result.best_loss, best, candidate_id))
    explored.sort(key=lambda item: item[0])
    explored = explored[: search.keep_after_exploration]

    refinement_fit = _stage_fit(config.fit, search.local_refinement)
    refined = []
    for _loss, initial, candidate_id in explored:
        stage_dir = output.path / "local_refinement" / candidate_id
        checkpoint = CheckpointManager(
            stage_dir / "checkpoints",
            f"{compatibility_hash}:refinement:{candidate_id}",
        )
        local_trainer.configure_optimizer(refinement_fit, checkpoint)
        result = local_trainer.train(
            stage_report(stage_dir),
            initial_normalized=initial,
        )
        if result.stopped_by_signal:
            raise InterruptedError("Hybrid search interrupted during Adam refinement.")
        best = np.asarray(local_trainer.space.normalize(result.best_parameters))
        refined.append((result.best_loss, best, candidate_id))

    validation_evaluator = Trainer(
        model, validation_buckets, config.protocol, config.fit, config.runtime, None
    )
    comparison = []
    for training_loss, normalized, candidate_id in refined:
        training_evaluation = evaluator.evaluate(normalized, gradient=False)
        evaluation = validation_evaluator.evaluate(normalized, gradient=False)
        comparison.append(
            {
                "candidate_id": candidate_id,
                "training_loss": float(training_loss),
                "validation_loss": float(evaluation[0]),
                "validation_component_losses": evaluation[4],
                "validation_rmse_mV": float(evaluation[5]) ** 0.5,
                "normalized": normalized.tolist(),
            }
        )
        if search.reporting.plot_final_candidates:
            candidate_plot_dir = (
                output.path / "final_candidates" / candidate_id / "plots"
            )
            plot_epoch_traces(
                candidate_plot_dir,
                training_buckets,
                training_evaluation[3],
                epoch=0,
                loss=float(training_evaluation[0]),
                experimental_alpha=0.6,
                filename="training.png",
            )
            plot_epoch_traces(
                candidate_plot_dir,
                validation_buckets,
                evaluation[3],
                epoch=0,
                loss=float(evaluation[0]),
                experimental_alpha=0.6,
                filename="validation.png",
            )
    comparison.sort(key=lambda item: item["validation_loss"])
    selected = comparison[0]
    space = ProjectedBoxSpace.from_specs(specs)
    physical = np.asarray(space.physical(selected["normalized"]))
    output.write_json("hybrid_comparison.json", {"candidates": comparison})
    output.write_json(
        "selected_model.json",
        {key: value for key, value in selected.items() if key != "normalized"},
    )
    output.write_parameters("parameters_best.csv", specs, physical)
    selected_evaluation = validation_evaluator.evaluate(
        selected["normalized"], gradient=False
    )
    plot_epoch_traces(
        output.path / "validation" / "plots",
        validation_buckets,
        selected_evaluation[3],
        epoch=0,
        loss=selected["validation_loss"],
        experimental_alpha=0.6,
        filename="selected_validation.png",
    )
    return selected
