# LSU_1 configuration reference

This file documents the complete configuration assembled from
[`LSU_1.yaml`](LSU_1.yaml) and its inherited configuration files. It describes
the current values in the YAML, not the historical values implied by some file
names.

## Inheritance and optimizer

`LSU_1.yaml` extends `configs/optimizers/adam_backtracking.yaml`, which in turn
extends the CPU, `dt = 0.1 ms` runtime and the main Combe fit configuration.
The resulting local optimizer is Adam with:

| Setting | Value |
|---|---:|
| Learning rate | `0.01` |
| Gradient clipping norm | `10.0` |
| Epochs | `50` |
| Backtracking | enabled |
| Backtracking reduction | `0.5` |
| Learning-rate growth after acceptance | `1.2` |
| Learning-rate limits | `0.0001`–`0.1` |
| Maximum line-search trials | `6` |
| Require loss decrease | `true` |

Adam proposes the update direction. Backtracking tries progressively smaller
steps until the complete objective decreases. The loss components described
below do not depend on whether this optimizer or a hybrid/global optimizer is
used.

## Training traces and time windows

The inherited dataset selection uses `trace_indices: [1, 3]` for both
`depolarizing_step` and `hyperpolarizing_pulse`. Thus, the training set contains
the first and third trace mapped to each protocol; it does not depend on the
recordings having identical names across cells.

For the timing currently configured for cell `m20260331b`:

| Protocol | Current step | Simulation end | Post-step interval |
|---|---:|---:|---:|
| Depolarizing | 200–500 ms | 1000 ms | 500 ms |
| Hyperpolarizing | 500–550 ms | 700 ms | 150 ms |

`LSU_1.yaml` requests 500 ms after a stimulus by default and overrides the
hyperpolarizing protocol to 150 ms, making its endpoint 700 ms. The `score`
window begins 100 ms before stimulus onset and runs to the simulation endpoint.
Other window names used below are:

- `stimulus`: current onset through current offset.
- `recovery`: after current offset through the simulation endpoint.
- `outside_stimulus`: the pre-step baseline and post-step recovery.

## Aggregation

The objective uses `aggregation: protocol_mean`, with:

| Protocol | Protocol total | Two-trace contribution |
|---|---:|---:|
| Depolarizing | `0.8` | `0.4` per trace |
| Hyperpolarizing | `0.2` | `0.1` per trace |

Because `renormalize_protocol_filtered_components` is `false`, a
depolarization-only component retains the `0.8` protocol allocation rather
than being renormalized to `1.0`.

For a component with raw weight \(w\), its contribution before the global
spike penalty is:

```text
w × sum(per-trace protocol allocation × normalized trace metric)
```

Raw weights are not percentages. The normalization scale and mathematical
form of each metric also determine its numerical influence.

## Components

### 1. Shared score-window waveform MSE

Label: `hyperpolarizing_waveform_mse`

| Field | Value |
|---|---|
| Kind | `voltage_mse` |
| Weight | `1.17` |
| Protocols | depolarizing and hyperpolarizing |
| Window | `score` |
| Scale | `5 mV` |

This compares the complete simulated and experimental voltage trajectories:

```text
mean(((V_sim - V_exp) / 5 mV)²)
```

The label is historical: despite saying “hyperpolarizing,” the component
currently applies to **both protocols**. Its effective coefficients are
`1.17 × 0.8 = 0.936` for the depolarizing protocol and
`1.17 × 0.2 = 0.234` for the hyperpolarizing protocol.

### 2. Depolarizing firing rate

Label: `depolarizing_firing_rate`

| Field | Value |
|---|---|
| Kind | `soft_firing_rate_error` |
| Weight | `4.0` |
| Protocol | depolarizing |
| Window | `stimulus` |
| Threshold | `-20 mV` |
| Temperature | `2 mV` |
| Rate scale | `5 Hz` |

A sigmoid around the threshold converts voltage to a smooth spike activation.
Positive activation changes approximate upward threshold crossings. The
squared difference between simulated and experimental firing rates is then
normalized by 5 Hz. This avoids the discontinuity of an integer spike count.

Its effective pre-metric coefficient is `4.0 × 0.8 = 3.2`, the largest in
LSU_1.

### 3. Additional depolarizing waveform MSE

Label: `depolarizing_waveform_mse`

| Field | Value |
|---|---|
| Kind | `voltage_mse` |
| Weight | `0.5` |
| Protocol | depolarizing |
| Window | `score` |
| Scale | `5 mV` |

This adds direct pointwise sensitivity to the whole scored depolarizing trace.
Its effective coefficient is `0.4`. Together with component 1, the
depolarizing score-window MSE coefficient is
`(1.17 + 0.5) × 0.8 = 1.336`.

### 4. Depolarizing voltage plateau

Label: `depolarizing_voltage_plateau`

| Field | Value |
|---|---|
| Kind | `subthreshold_mean_error` |
| Weight | `0.15` |
| Protocol | depolarizing |
| Window | `stimulus` |
| Threshold | `-20 mV` |
| Scale | `5 mV` |

This compares the mean simulated and experimental subthreshold voltage during
the pulse. The subthreshold mask is defined by the experimental trace, so its
membership does not change during optimization. It describes the broad
depolarized plateau and complements the more specific DBLO metric. Its
effective coefficient is `0.12`.

### 5. Depolarization baseline offset (DBLO)

Label: `depolarizing_dblo`

| Field | Value |
|---|---|
| Kind | `soft_dblo_error` |
| Weight | `2.0` |
| Protocol | depolarizing |
| Window | `stimulus` |
| Threshold | `-20 mV` |
| Temperature | `1 mV` |
| Scale | `5 mV` |

Experimental spikes define fixed intervals from each spike peak to the next
upward threshold crossing. A smooth minimum estimates the voltage floor in
each interval, and the pre-step resting voltage is subtracted. The loss
compares the mean simulated and experimental offsets. Fixing the intervals
from the experimental trace and using smooth minima keeps the simulated metric
differentiable.

If an experimental trace has fewer than two qualifying in-step spikes, no
inter-spike DBLO interval exists and that trace contributes zero to this
component. Its effective coefficient is `2.0 × 0.8 = 1.6`.

### 6. Spike waveform

Label: `depolarizing_spike_waveform`

| Field | Value |
|---|---|
| Kind | `pseudo_huber` |
| Weight | `0.25` |
| Protocol | depolarizing |
| Window | `stimulus` |
| Scale | `5 mV` |
| Delta | `1.0` |

Pseudo-Huber loss is pointwise over the entire stimulus window:

```text
delta² × (sqrt(1 + (normalized error / delta)²) - 1)
```

It is quadratic for small errors and approaches linear growth for large
errors, making it less dominated by temporally misaligned spikes than MSE.
It is not restricted to spike-only samples. Its pre-metric effective
coefficient is `0.20`.

### 7. Spike derivative

Label: `depolarizing_spike_derivative`

| Field | Value |
|---|---|
| Kind | `derivative_mse` |
| Weight | `0.25` |
| Protocol | depolarizing |
| Window | `stimulus` |
| Scale | `20 mV/ms` |

This compares `dV/dt` during the pulse, constraining spike rise, fall, and
symmetry. Its effective coefficient is `0.20`.

### 8. Spike height

Label: `depolarizing_spike_height`

| Field | Value |
|---|---|
| Kind | `soft_maximum_voltage_error` |
| Weight | `0.32` |
| Protocol | depolarizing |
| Window | `stimulus` |
| Temperature | `1 mV` |
| Scale | `5 mV` |

This compares smooth approximations of the maximum voltage during the
depolarizing step. It constrains the overall peak height, not every spike
height separately. Its effective coefficient is `0.256`.

### 9. Recovery waveform

Label: `depolarizing_recovery_waveform`

| Field | Value |
|---|---|
| Kind | `pseudo_huber` |
| Weight | `0.5` |
| Protocol | depolarizing |
| Window | `recovery` |
| Scale | `3 mV` |
| Delta | `1.0` |

This compares the full 500 ms post-step trajectory, capturing both the
after-hyperpolarization and slow return toward baseline. Its pre-metric
effective coefficient is `0.4`.

### 10. Recovery derivative

Label: `depolarizing_recovery_derivative`

| Field | Value |
|---|---|
| Kind | `derivative_mse` |
| Weight | `0.2` |
| Protocol | depolarizing |
| Window | `recovery` |
| Scale | `1 mV/ms` |

This compares recovery slopes, distinguishing a slow return from a trajectory
that reaches a similar voltage too quickly. Its effective coefficient is
`0.16`.

### 11. After-hyperpolarization depth

Label: `depolarizing_ahp_depth`

| Field | Value |
|---|---|
| Kind | `soft_minimum_voltage_error` |
| Weight | `0.10` |
| Protocol | depolarizing |
| Window | `recovery` |
| Temperature | `1 mV` |
| Scale | `3 mV` |

This compares smooth minimum voltages during recovery and directly constrains
the post-step AHP depth. Its effective coefficient is `0.08`.

## Outside-stimulus spike penalty

The penalty `outside_step_spikes` applies to both protocols:

| Field | Value |
|---|---:|
| Factor per soft spike | `1.1` |
| Threshold | `-20 mV` |
| Temperature | `2 mV` |
| Maximum multiplier | `1e12` |

The final objective is:

```text
final loss = base loss × min(1.1 ^ N_outside, 1e12)
```

`N_outside` is the sum of continuous upward-crossing proxies before and after
the current steps across all selected traces. It is deliberately not rounded:
fractional soft counts retain a gradient near threshold. One full extra spike
approaches a multiplier of `1.1`; two approach `1.21`.

## Differentiability

LSU_1 is designed for JAX differentiation:

- firing rate and outside-spike counts use smooth threshold activations;
- DBLO uses experimental intervals and smooth simulated minima;
- spike height and AHP depth use smooth maximum/minimum operations;
- waveform, derivative, and pseudo-Huber metrics are continuous.

The positive-part operation used for upward crossings is piecewise
differentiable, which remains compatible with automatic differentiation.

## How to interpret the weights

The pre-metric coefficients are a bookkeeping guide:

| Feature | Effective coefficient |
|---|---:|
| Firing rate | `3.200` |
| DBLO | `1.600` |
| Shared waveform MSE, both protocols combined | `1.170` |
| Additional depolarizing waveform MSE | `0.400` |
| Spike waveform | `0.200` |
| Spike derivative | `0.200` |
| Spike height | `0.256` |
| Recovery waveform | `0.400` |
| Recovery derivative | `0.160` |
| AHP depth | `0.080` |
| Plateau | `0.120` |

These are not expected loss magnitudes. MSE, pseudo-Huber, derivative,
soft-extremum, DBLO, and firing-rate metrics have different normalization and
distributions. Use the logged component losses to determine which terms
actually dominate a run.

## Output and diagnostics

`metrics.jsonl` reports:

- `loss`: the total objective after penalties;
- `component_losses`: every named weighted contribution;
- `penalty_metrics.base_loss`: the objective before multiplication;
- the outside-spike soft count and multiplier;
- `rmse_mV`: a common descriptive voltage error, not an additional LSU_1 term.

When the spike penalty is active, component contributions are reported with
the same multiplier so their sum remains consistent with the total loss.

## Related configurations

- `LSU_1.yaml`: local Adam with backtracking.
- `LSU_1_wide_bounds.yaml`: LSU_1 with jittered initialization. Its current
  bound expansion is `1.0`, so it presently uses canonical bounds despite its
  historical name.
- `LSU_1_wide_bounds_adam.yaml`: exploratory Adam without backtracking.
- `configs/search/LSU_1_cma_adam.yaml`: CMA-ES global search followed by Adam,
  while retaining this loss definition.

