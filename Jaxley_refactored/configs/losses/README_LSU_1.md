# LSU_1 configuration reference

[`LSU_1.yaml`](LSU_1.yaml) defines a deliberately simple objective: one
point-by-point voltage MSE plus a differentiable penalty for spikes outside
the stimulus.

## Optimizer and traces

LSU_1 inherits Adam with backtracking from
`configs/optimizers/adam_backtracking.yaml`. The training set uses the first
and third traces mapped to each protocol, so trace names can differ across
cells.

The depolarizing simulation includes 500 ms after the pulse. Hyperpolarizing
traces use 100 ms after their 550 ms pulse offset and therefore end at 650 ms
for both cells. The score window starts 100 ms before stimulus onset and
continues through the simulation endpoint.

## Waveform MSE

The only additive component is:

```yaml
- kind: voltage_mse
  label: waveform_mse
  weight: 1.0
  protocols: [depolarizing_step, hyperpolarizing_pulse]
  window: score
  scale_mV: 5.0
```

For every selected trace it computes:

```text
mean(((V_simulated(t) - V_experimental(t)) / 5 mV)²)
```

This is a point-by-point comparison, not a comparison of the two traces'
overall mean voltages. It directly penalizes voltage differences throughout
the baseline, stimulus, and recovery portions included in the score window.

No separate firing-rate, DBLO, plateau, spike-shape, spike-height,
derivative, recovery, or AHP component is included. Those features influence
the optimization only insofar as they change the point-by-point waveform MSE.

## Aggregation

The objective uses `protocol_mean` with:

| Protocol | Protocol contribution | Contribution per selected trace |
|---|---:|---:|
| Depolarizing | `0.8` | `0.4` |
| Hyperpolarizing | `0.2` | `0.1` |

There are two selected traces per protocol. The unpenalized loss is therefore:

```text
base_loss =
    0.4 × MSE(depolarizing trace 1)
  + 0.4 × MSE(depolarizing trace 3)
  + 0.1 × MSE(hyperpolarizing trace 1)
  + 0.1 × MSE(hyperpolarizing trace 3)
```

## Outside-stimulus spike penalty

The retained penalty is:

```yaml
- kind: soft_outside_stimulus_spike_multiplier
  label: outside_step_spikes
  factor_per_spike: 1.1
  maximum_multiplier: 1.0e12
  protocols: [depolarizing_step, hyperpolarizing_pulse]
  threshold_mV: -20.0
  temperature_mV: 2.0
```

It counts continuous approximations of upward threshold crossings before and
after each current step. The final objective is:

```text
final_loss = base_loss × min(1.1 ^ N_outside, 1e12)
```

The soft count is not rounded, preserving gradients near the -20 mV
threshold. One complete outside spike approaches a multiplier of `1.1`; two
approach `1.21`.

## Differentiability

The MSE is differentiable in simulated voltage. The outside-spike count uses a
sigmoid threshold occupancy and positive occupancy changes rather than a hard
integer spike detector. This keeps the complete objective compatible with JAX
automatic differentiation.

## Related configurations

`LSU_1_wide_bounds.yaml`, `LSU_1_wide_bounds_adam.yaml`, and
`configs/search/LSU_1_cma_adam.yaml` inherit this same objective while changing
initialization or optimization strategy.
