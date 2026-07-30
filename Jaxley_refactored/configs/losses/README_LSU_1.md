# LSU_1 configuration reference

[`LSU_1.yaml`](LSU_1.yaml) uses different objectives for the two stimulation
protocols:

- Hyperpolarizing traces use point-by-point voltage MSE.
- Depolarizing traces use firing rate, DBLO, spike shape and height, and
  post-step recovery features. They do not use point-by-point voltage MSE.
- Both protocols receive a differentiable penalty for spikes outside the
  stimulus.

## Optimizer, traces, and windows

LSU_1 inherits Adam with backtracking from
`configs/optimizers/adam_backtracking.yaml`. The training set uses the first
and third trace mapped to each protocol, independently of cell-specific trace
names.

Depolarizing simulations include 500 ms after the pulse. Hyperpolarizing
simulations include 100 ms after their 550 ms pulse offset and end at 650 ms.

The relevant windows are:

- `score`: 100 ms before stimulus onset through the simulation endpoint.
- `stimulus`: current onset through current offset.
- `recovery`: after current offset through the simulation endpoint.

## Components

| Label | Purpose | Weight | Protocol/window | Scale |
|---|---|---:|---|---:|
| `hyperpolarizing_waveform_mse` | Point-by-point voltage MSE | `1.0` | Hyperpolarizing, `score` | `5 mV` |
| `depolarizing_firing_rate` | Smooth firing-frequency error | `4.0` | Depolarizing, `stimulus` | `5 Hz` |
| `depolarizing_dblo` | Inter-spike voltage floor relative to rest | `2.0` | Depolarizing, `stimulus` | `5 mV` |
| `depolarizing_spike_waveform` | Robust spike waveform comparison | `0.25` | Depolarizing, `stimulus` | `5 mV` |
| `depolarizing_spike_derivative` | Spike rise/fall and symmetry | `0.25` | Depolarizing, `stimulus` | `20 mV/ms` |
| `depolarizing_spike_height` | Smooth maximum-voltage error | `0.32` | Depolarizing, `stimulus` | `5 mV` |
| `depolarizing_recovery_waveform` | Post-step recovery trajectory | `0.5` | Depolarizing, `recovery` | `3 mV` |
| `depolarizing_recovery_derivative` | Speed of return toward baseline | `0.2` | Depolarizing, `recovery` | `1 mV/ms` |
| `depolarizing_ahp_depth` | Post-step minimum/AHP depth | `0.10` | Depolarizing, `recovery` | `3 mV` |

### Hyperpolarizing waveform MSE

This is the only `voltage_mse` component:

```text
mean(((V_simulated(t) - V_experimental(t)) / 5 mV)²)
```

It is restricted to `hyperpolarizing_pulse`; there is no depolarizing
point-by-point MSE.

### Firing rate

A sigmoid around -20 mV converts voltage into smooth threshold occupancy.
Positive occupancy changes approximate upward crossings. The squared
simulated-versus-experimental rate difference is normalized by 5 Hz.

### DBLO

Experimental spikes define fixed peak-to-next-threshold intervals. Smooth
simulated minima within those intervals estimate the inter-spike voltage floor
relative to the pre-step resting voltage. Experimental intervals and smooth
minima preserve differentiability. A trace with fewer than two qualifying
experimental spikes contributes zero to this term.

### Spike shape and height

Spike shape combines:

- pseudo-Huber voltage error across the stimulus, which is less sensitive than
  MSE to large temporally misaligned residuals;
- derivative MSE, which constrains rise and fall kinetics;
- smooth maximum-voltage error, which constrains overall spike height.

The waveform and derivative terms cover the complete stimulus interval, not
only samples classified as spikes.

### Decay to baseline

Recovery combines:

- pseudo-Huber voltage error over the complete post-step trajectory;
- derivative MSE to constrain the return speed;
- smooth minimum-voltage error to constrain AHP depth.

Together these terms represent both the hyperpolarized undershoot and the slow
return toward baseline.

## Aggregation

The objective uses `protocol_mean`:

| Protocol | Protocol total | Per selected trace |
|---|---:|---:|
| Depolarizing | `0.8` | `0.4` |
| Hyperpolarizing | `0.2` | `0.1` |

`renormalize_protocol_filtered_components: false` means depolarizing-only
components retain a total factor of `0.8`, while the hyperpolarizing MSE retains
a total factor of `0.2`.

## Outside-stimulus spike penalty

Both protocols use:

```text
final_loss = base_loss × min(1.1 ^ N_outside, 1e12)
```

`N_outside` is a continuous approximation of upward -20 mV crossings before
and after the stimulus. Its sigmoid temperature is 2 mV. Fractional counts
retain useful gradients near threshold.

## Differentiability

The loss avoids integer simulated spike counts and hard simulated extrema:
firing rate and the penalty use smooth threshold occupancy, while DBLO, spike
height, and AHP depth use smooth extrema. MSE, pseudo-Huber, and derivative
terms are continuous and compatible with JAX automatic differentiation.

## Related configurations

`LSU_1_wide_bounds.yaml`, `LSU_1_wide_bounds_adam.yaml`, and
`configs/search/LSU_1_cma_adam.yaml` inherit this same objective while changing
initialization or optimization strategy.
