# Loss-function library

Every objective consumes the same simulated traces and returns one scalar per
trace. Protocol-balanced trace weights and component weights reduce those
values to one scalar objective. Gradients from both natural-shape trace buckets
are summed before one update to the shared parameter vector.

## Registered losses

| Kind | Meaning | Typical scale |
| --- | --- | --- |
| `voltage_mse` | Mean squared voltage residual | `scale_mV: 1` |
| `voltage_mae` | Mean absolute voltage residual | `scale_mV: 1–5` |
| `pseudo_huber` | Smooth robust voltage residual | `scale_mV: 5`, `delta: 1` |
| `normalized_voltage_mse` | MSE normalized by observed voltage range | Minimum range via `scale_mV` |
| `derivative_mse` | MSE of `dV/dt` | `scale_mV_per_ms: 10` |
| `correlation_loss` | One minus masked waveform correlation | No scale required |
| `resting_voltage_error` | Squared error between window means | `baseline` window |
| `steady_state_error` | Squared error between window means | `stimulus_end` window |
| `soft_firing_rate_error` | Squared difference between positive smooth-threshold occupancy rates | `threshold_mV`, `temperature_mV`, `scale_hz` |
| `subthreshold_mean_error` | Squared inter-spike mean-voltage error using an experimental subthreshold mask | `threshold_mV`, `scale_mV` |
| `soft_dblo_error` | Squared error in depolarization baseline offset: mean interspike minimum minus pre-step rest | `threshold_mV`, `temperature_mV`, `scale_mV` |
| `soft_minimum_voltage_error` | Squared difference between smooth minimum voltages | `temperature_mV`, `scale_mV` |
| `soft_maximum_voltage_error` | Squared difference between smooth maximum voltages | `temperature_mV`, `scale_mV` |

`masked_voltage_mse` remains an alias for backward compatibility.

### Explicit DBLO metric

`soft_dblo_error` follows the paper-level definition
`DBLO = DBL - E_rest`, where DBL is the mean of the minimum voltage between
consecutive spikes and `E_rest` is the pre-step mean voltage. Upward threshold
crossings and spike peaks are detected once from the experimental trace. Each
fixed interval runs from one observed spike peak to immediately before the next
observed upward threshold crossing.

Within those fixed intervals, hard minima are replaced by normalized smooth
minima controlled by `temperature_mV`. The same smooth definition is applied
to simulated and experimental voltage. Consequently, the loss is continuous
and differentiable with respect to the simulation, while an exact trace match
still gives zero. A trace with fewer than two observed in-step spikes has no
defined DBLO and contributes finite zero to this component; the LSU_1
firing-rate term remains responsible for spike count. The LSU_1 recordings are
assumed to be liquid-junction-potential corrected before loading.

### Continuous firing-rate surrogate

The firing-rate term first maps voltage to a smooth threshold occupancy,
`p(t) = sigmoid((V(t) - threshold)/temperature)`, and sums only positive
increments, `relu(p(t+1) - p(t))`. A complete below-to-above excursion
therefore contributes approximately one event, while a stationary voltage near
threshold contributes zero. This avoids allowing a depolarized non-spiking
plateau to imitate repeated spikes. The construction is continuous, piecewise
differentiable, and compatible with JAX automatic differentiation.

## Named windows

- `score`: the previous fitter's stimulus window, from 100 ms before onset to
  800 ms after offset, clipped to the trace.
- `full_trace`: every real sample.
- `baseline`: samples before stimulus onset.
- `stimulus`: onset through offset.
- `outside_stimulus`: samples before onset and after offset; used by the
  outside-spike multiplier.
- `recovery`: samples after stimulus offset.
- `stimulus_end`: the final 5 ms of the stimulus.

LSU_1 simulates depolarizing records through 500 ms after stimulus offset.
Hyperpolarizing records use a protocol override of 150 ms after their
trace-local 550-ms offset, so they end at 700 ms. The loader fails rather than
silently shortening a requested window. Consequently, the current
`m20240527cd` hyperpolarizing segments—which end at about 650 ms—must be
re-extracted from longer source data before they can use the 700-ms endpoint.

## Multiplicative penalties

`soft_outside_stimulus_spike_multiplier` uses the same positive-occupancy-rise
surrogate as the firing-rate term, without duration normalization. A crossing
is assigned according to its destination sample, so a crossing at the first
recovery sample is outside while one at stimulus onset is inside.

Across all selected traces, the configured penalty is:

```text
penalized_loss = base_loss * factor_per_spike ** total_soft_outside_spikes
```

The LSU_1 factor is `1.1`. One isolated outside spike therefore approaches a
multiplier of `1.1`, and two approach `1.21`. Fractional counts preserve useful
gradients near threshold. A high `maximum_multiplier` is only a numerical guard
against pathological traces with very many threshold excursions.

## Configuration

```yaml
fit:
  objective:
    penalties:
      - kind: soft_outside_stimulus_spike_multiplier
        label: outside_step_spikes
        factor_per_spike: 1.1
        maximum_multiplier: 1.0e12
        protocols: [depolarizing_step, hyperpolarizing_pulse]
        threshold_mV: -20.0
        temperature_mV: 2.0

    components:
      - kind: pseudo_huber
        label: waveform
        weight: 1.0
        window: score
        scale_mV: 5.0
        delta: 1.0

      - kind: derivative_mse
        label: derivative
        weight: 0.2
        window: stimulus
        scale_mV_per_ms: 10.0

      - kind: steady_state_error
        label: hyper_steady
        weight: 0.25
        protocols: [hyperpolarizing_pulse]
        window: stimulus_end
        scale_mV: 5.0
```

Labels must be unique. Protocol-filtered components are normalized over the
selected traces, so their weights do not accidentally depend on trace count.
Set `renormalize_protocol_filtered_components: false` to retain the global
protocol weights for those components. LSU_1 uses this mode. With its current
`0.8/0.2` protocol allocation, the two depolarizing traces receive `0.4` each
and the two hyperpolarizing traces receive `0.1` each.

`metrics.jsonl` reports both the total objective under `loss` and each weighted
contribution under `component_losses`. When penalties are configured,
`penalty_metrics` reports the unpenalized `base_loss`, the global
`loss_multiplier`, and each continuous soft-spike count. `rmse_mV` always
remains the common score-window voltage RMSE, independent of the training
objective.

`configs/losses/LSU_1.yaml` combines shared score-window MSE with
smooth depolarizing firing rate, explicit DBLO, full score-window depolarizing
MSE, plateau, spike-shape, recovery-trajectory, and after-hyperpolarization
terms. Firing rate has raw weight `4.0` and DBLO has raw weight `2.0`, giving
pre-metric coefficients of `3.2` and `1.6` under the current depolarizing
allocation. The additional depolarizing waveform MSE has raw weight `0.5` and
retains direct sensitivity to conspicuous pointwise trace mismatches. Raw
weights differ because protocol allocations, normalization scales, and loss
primitives affect final contributions. LSU_1 inherits Adam with backtracking.

See
[`configs/losses/README_LSU_1.md`](../configs/losses/README_LSU_1.md)
for the complete component-by-component reference.

Use a 100-step smoke test with any loss configuration:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/losses/pseudo_huber.yaml \
  --cells m20240527cd --epochs 1 --max-steps 100
```

Run all three shipped objectives sequentially with identical cell, seed, and
training length:

```bash
bash scripts/run_loss_comparison.sh \
  --cell m20240527cd --epochs 1 --max-steps 100
```

Remove `--max-steps` and choose the full epoch count only after smoke testing.
