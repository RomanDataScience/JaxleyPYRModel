"""Full-dataset multi-bucket gradient training."""

from __future__ import annotations

from dataclasses import dataclass
import signal
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from jaxley_refactored.config.schema import FitSpec, ProtocolSpec, RuntimeSpec
from jaxley_refactored.data import TraceBucket
from jaxley_refactored.models import BuiltModel
from jaxley_refactored.parameters import ProjectedBoxSpace
from jaxley_refactored.simulation import InitialStateFactory, SimulationKernel

from .checkpoints import CheckpointManager
from .losses import (
    BucketObjective,
    component_denominators,
    default_loss_registry,
    weighted_bucket_loss,
)
from .initialization import initial_normalized_values
from .optimizer import Adam, BacktrackingLineSearch


@dataclass(frozen=True)
class PreparedBucket:
    bucket: TraceBucket
    kernel: SimulationKernel
    initial_states: object
    evaluate: Callable
    loss_and_gradient: Callable


@dataclass(frozen=True)
class FitResult:
    best_loss: float
    best_parameters: object
    final_parameters: object
    epochs_completed: int
    stopped_by_signal: bool


class Trainer:
    """Coordinate bucket gradients while keeping each compiled shape separate."""

    def __init__(
        self,
        model: BuiltModel,
        buckets: Iterable[TraceBucket],
        protocol: ProtocolSpec,
        fit: FitSpec,
        runtime: RuntimeSpec,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.model = model
        self.fit = fit
        self.runtime = runtime
        self.space = ProjectedBoxSpace.from_specs(model.parameterizer.specs)
        self.optimizer = Adam(fit.optimizer, self.space)
        self.line_search = BacktrackingLineSearch(
            fit.optimizer.line_search, self.optimizer.candidate
        )
        self.checkpoints = checkpoint_manager
        self._stop_requested = False
        buckets = tuple(buckets)
        denominators = component_denominators(fit.components, buckets)
        registry = default_loss_registry()
        self._prepared = tuple(
            self._prepare_bucket(
                bucket,
                protocol,
                BucketObjective(fit.components, bucket, denominators, registry),
            )
            for bucket in buckets
        )
        if not self._prepared:
            raise ValueError("Trainer requires at least one trace bucket.")

    def configure_optimizer(
        self, fit: FitSpec, checkpoint_manager: CheckpointManager | None
    ) -> None:
        """Reuse prepared simulation/loss kernels with a new Adam policy."""
        if fit.components != self.fit.components:
            raise ValueError("Optimizer reconfiguration cannot change loss components.")
        if fit.batching_strategy != self.fit.batching_strategy:
            raise ValueError("Optimizer reconfiguration cannot change batching strategy.")
        self.fit = fit
        self.optimizer = Adam(fit.optimizer, self.space)
        self.line_search = BacktrackingLineSearch(
            fit.optimizer.line_search, self.optimizer.candidate
        )
        self.checkpoints = checkpoint_manager
        self._stop_requested = False

    def _prepare_bucket(
        self,
        bucket: TraceBucket,
        protocol: ProtocolSpec,
        objective: BucketObjective,
    ) -> PreparedBucket:
        initial_voltages = (
            bucket.initial_voltage_mV
            if protocol.initial_state_mode == "observed_first_sample"
            else jnp.full(
                bucket.initial_voltage_mV.shape, protocol.fixed_voltage_mV
            )
        )
        initial_states = InitialStateFactory(
            self.model.cell, bucket.dt_ms
        ).build(initial_voltages)
        kernel = SimulationKernel(
            cell=self.model.cell,
            parameterizer=self.model.parameterizer,
            protocol=protocol,
            runtime=self.runtime,
            dt_ms=bucket.dt_ms,
            n_steps=bucket.n_steps,
        )
        currents = jnp.asarray(bucket.currents_nA)
        observed = jnp.asarray(bucket.observed_mV)
        score_masks = jnp.asarray(bucket.score_masks)
        trace_weights = jnp.asarray(bucket.weights)

        def loss(normalized):
            physical = self.space.physical(normalized)
            predicted = (
                kernel.simulate_serial(physical, currents, initial_states)
                if self.fit.batching_strategy == "serial"
                else kernel.simulate_batch(physical, currents, initial_states)
            )
            objective_loss, components = objective(predicted, observed)
            evaluation_mse = weighted_bucket_loss(
                predicted, observed, score_masks, trace_weights
            )
            return objective_loss, (predicted, components, evaluation_mse)

        evaluate = loss
        loss_and_gradient = jax.value_and_grad(loss, has_aux=True)
        if self.runtime.jit:
            # The simulation functions are already compiled per shape. Jitting
            # this small wrapper fuses loss reduction and reverse-mode setup.
            evaluate = jax.jit(evaluate)
            loss_and_gradient = jax.jit(loss_and_gradient)
        return PreparedBucket(
            bucket=bucket,
            kernel=kernel,
            initial_states=initial_states,
            evaluate=evaluate,
            loss_and_gradient=loss_and_gradient,
        )

    def evaluate(
        self, normalized, *, gradient: bool
    ) -> tuple[object, object | None, dict, dict, dict, float]:
        """Evaluate every shape bucket and optionally accumulate its gradient."""
        total_loss = jnp.asarray(0.0)
        total_gradient = jnp.zeros_like(normalized) if gradient else None
        bucket_losses = {}
        bucket_predictions = {}
        component_losses = {
            component.label: 0.0 for component in self.fit.components
        }
        evaluation_mse = 0.0
        for prepared in self._prepared:
            if gradient:
                (
                    (loss, (predicted, components, bucket_evaluation_mse)),
                    bucket_gradient,
                ) = prepared.loss_and_gradient(normalized)
                total_gradient = total_gradient + bucket_gradient
            else:
                loss, (
                    predicted,
                    components,
                    bucket_evaluation_mse,
                ) = prepared.evaluate(normalized)
            total_loss = total_loss + loss
            bucket_losses[str(prepared.bucket.key)] = float(loss)
            bucket_predictions[prepared.bucket.key] = np.asarray(predicted)
            evaluation_mse += float(bucket_evaluation_mse)
            for label, value in components.items():
                component_losses[label] += float(value)
        return (
            total_loss,
            total_gradient,
            bucket_losses,
            bucket_predictions,
            component_losses,
            evaluation_mse,
        )

    def install_signal_handlers(self) -> None:
        def request_stop(_signum, _frame):
            self._stop_requested = True

        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def train(
        self,
        on_epoch: Callable[[dict, dict[tuple[float, int], np.ndarray]], None]
        | None = None,
        *,
        initial_normalized=None,
    ) -> FitResult:
        normalized = jnp.asarray(
            initial_normalized
            if initial_normalized is not None
            else initial_normalized_values(self.model, self.fit, self.runtime)
        )
        optimizer_state = self.optimizer.initialize(normalized)
        best_normalized = normalized
        best_loss = float("inf")
        start_epoch = 0

        if self.checkpoints is not None:
            restored = self.checkpoints.load()
            if restored is not None:
                start_epoch = restored["epoch"] + 1
                normalized = jnp.asarray(restored["normalized"])
                optimizer_state = restored["optimizer"]
                best_normalized = jnp.asarray(restored["best_normalized"])
                best_loss = restored["best_loss"]

        completed = start_epoch
        for epoch in range(start_epoch, self.fit.optimizer.epochs):
            (
                total_loss,
                total_gradient,
                bucket_losses,
                bucket_predictions,
                component_losses,
                evaluation_mse,
            ) = self.evaluate(normalized, gradient=True)
            assert total_gradient is not None
            evaluated_normalized = normalized
            loss_before_step = float(total_loss)
            learning_rate = self.fit.optimizer.learning_rate
            line_search_trials = 0
            step_accepted = True

            if self.fit.optimizer.line_search.enabled:
                direction, proposed_state, gradient_norm = self.optimizer.direction(
                    total_gradient, optimizer_state
                )
                search = self.fit.optimizer.line_search
                learning_rate = float(
                    optimizer_state.current_learning_rate
                    if optimizer_state.current_learning_rate is not None
                    else self.fit.optimizer.learning_rate
                )
                learning_rate = min(
                    search.maximum_learning_rate,
                    max(search.minimum_learning_rate, learning_rate),
                )
                result = self.line_search.search(
                    normalized,
                    direction,
                    loss_before_step,
                    learning_rate,
                    lambda candidate: self.evaluate(candidate, gradient=False),
                )
                normalized = result.values
                learning_rate = result.learning_rate
                line_search_trials = result.trials
                step_accepted = result.accepted
                if result.accepted:
                    optimizer_state = self.optimizer.accept(
                        proposed_state, result.learning_rate
                    )
                    assert result.evaluation is not None
                    (
                        total_loss,
                        _,
                        bucket_losses,
                        bucket_predictions,
                        component_losses,
                        evaluation_mse,
                    ) = result.evaluation
                else:
                    optimizer_state = self.optimizer.reject(
                        optimizer_state, result.next_learning_rate
                    )
            else:
                normalized, optimizer_state, gradient_norm = self.optimizer.update(
                    normalized, total_gradient, optimizer_state
                )

            loss_value = float(total_loss)
            loss_parameters = (
                normalized
                if self.fit.optimizer.line_search.enabled
                else evaluated_normalized
            )
            is_best = loss_value < best_loss
            if is_best:
                best_loss = loss_value
                best_normalized = loss_parameters
            completed = epoch + 1
            metrics = {
                "epoch": epoch,
                "loss": loss_value,
                "loss_before_step": loss_before_step,
                "rmse_mV": evaluation_mse**0.5,
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "line_search_trials": line_search_trials,
                "step_accepted": step_accepted,
                "bucket_losses": bucket_losses,
                "component_losses": component_losses,
            }
            if on_epoch is not None:
                on_epoch(metrics, bucket_predictions)
            if self.checkpoints is not None and (
                completed % self.fit.checkpoint_every_epochs == 0
                or self._stop_requested
            ):
                self.checkpoints.save(
                    epoch=epoch,
                    normalized=normalized,
                    optimizer=optimizer_state,
                    best_normalized=best_normalized,
                    best_loss=best_loss,
                    is_best=is_best,
                )
            if self._stop_requested:
                break

        return FitResult(
            best_loss=best_loss,
            best_parameters=self.space.physical(best_normalized),
            final_parameters=self.space.physical(normalized),
            epochs_completed=completed,
            stopped_by_signal=self._stop_requested,
        )
