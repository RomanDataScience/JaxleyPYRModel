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
| `soft_firing_rate_error` | Squared difference between smooth upward-crossing rates | `threshold_mV`, `temperature_mV`, `scale_hz` |
| `subthreshold_mean_error` | Squared inter-spike mean-voltage error using an experimental subthreshold mask | `threshold_mV`, `scale_mV` |
| `soft_minimum_voltage_error` | Squared difference between smooth minimum voltages | `temperature_mV`, `scale_mV` |

`masked_voltage_mse` remains an alias for backward compatibility.

## Named windows

- `score`: the previous fitter's stimulus window, from 100 ms before onset to
  800 ms after offset, clipped to the trace.
- `full_trace`: every real sample.
- `baseline`: samples before stimulus onset.
- `stimulus`: onset through offset.
- `recovery`: samples after stimulus offset.
- `stimulus_end`: the final 5 ms of the stimulus.

## Configuration

```yaml
fit:
  objective:
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
`metrics.jsonl` reports both the total objective under `loss` and each weighted
contribution under `component_losses`. `rmse_mV` always remains the common
score-window voltage RMSE, independent of the training objective.

`configs/losses/LSU_1.yaml` combines hyperpolarizing score-window MSE with
smooth depolarizing firing-rate, plateau, spike-shape, recovery-trajectory, and
after-hyperpolarization terms. It inherits Adam with backtracking.

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
