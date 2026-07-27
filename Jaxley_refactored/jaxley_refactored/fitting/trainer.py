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
from .optimizer import Adam


@dataclass(frozen=True)
class PreparedBucket:
    bucket: TraceBucket
    kernel: SimulationKernel
    initial_states: object
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

        loss_and_gradient = jax.value_and_grad(loss, has_aux=True)
        if self.runtime.jit:
            # The simulation functions are already compiled per shape. Jitting
            # this small wrapper fuses loss reduction and reverse-mode setup.
            loss_and_gradient = jax.jit(loss_and_gradient)
        return PreparedBucket(
            bucket=bucket,
            kernel=kernel,
            initial_states=initial_states,
            loss_and_gradient=loss_and_gradient,
        )

    def install_signal_handlers(self) -> None:
        def request_stop(_signum, _frame):
            self._stop_requested = True

        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def train(
        self,
        on_epoch: Callable[[dict, dict[tuple[float, int], np.ndarray]], None]
        | None = None,
    ) -> FitResult:
        physical = jnp.asarray(self.model.reference_values)
        normalized = self.space.normalize(physical)
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
            total_loss = jnp.asarray(0.0)
            total_gradient = jnp.zeros_like(normalized)
            bucket_losses = {}
            bucket_predictions = {}
            component_losses = {
                component.label: 0.0 for component in self.fit.components
            }
            evaluation_mse = 0.0
            for prepared in self._prepared:
                (
                    (loss, (predicted, components, bucket_evaluation_mse)),
                    gradient,
                ) = prepared.loss_and_gradient(normalized)
                total_loss = total_loss + loss
                total_gradient = total_gradient + gradient
                bucket_losses[str(prepared.bucket.key)] = float(loss)
                bucket_predictions[prepared.bucket.key] = np.asarray(predicted)
                evaluation_mse += float(bucket_evaluation_mse)
                for label, value in components.items():
                    component_losses[label] += float(value)

            loss_value = float(total_loss)
            is_best = loss_value < best_loss
            if is_best:
                best_loss = loss_value
                best_normalized = normalized
            normalized, optimizer_state, gradient_norm = self.optimizer.update(
                normalized, total_gradient, optimizer_state
            )
            completed = epoch + 1
            metrics = {
                "epoch": epoch,
                "loss": loss_value,
                "rmse_mV": evaluation_mse**0.5,
                "gradient_norm": float(gradient_norm),
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
