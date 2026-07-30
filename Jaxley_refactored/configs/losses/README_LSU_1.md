# LSU_1 configuration reference

[`LSU_1.yaml`](LSU_1.yaml) uses different objectives for the two stimulation
protocols:

- Hyperpolarizing traces use point-by-point voltage MSE.
- Depolarizing traces use firing rate, absolute mean interspike-minimum
  voltage, spike shape and height, and post-step recovery features. They do
  not use point-by-point voltage MSE.
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
| `hyperpolarizing_waveform_mse` | Point-by-point voltage MSE | `1.0` | Hyperpolarizing, `score` | `1 mV` |
| `depolarizing_firing_rate` | Smooth firing-frequency error | `1.0` | Depolarizing, `stimulus` | `1 Hz` |
| `depolarizing_interspike_minimum_voltage` | Absolute mean interspike minimum voltage | `1.0` | Depolarizing, `stimulus` | `1 mV` |
| `depolarizing_spike_waveform` | Robust spike waveform comparison | `1.0` | Depolarizing, `stimulus` | `1 mV` |
| `depolarizing_spike_derivative` | Spike rise/fall and symmetry | `1.0` | Depolarizing, `stimulus` | `1 mV/ms` |
| `depolarizing_spike_height` | Smooth maximum-voltage error | `1.0` | Depolarizing, `stimulus` | `1 mV` |
| `depolarizing_recovery_waveform` | Post-step recovery trajectory | `1.0` | Depolarizing, `recovery` | `1 mV` |
| `depolarizing_recovery_derivative` | Speed of return toward baseline | `1.0` | Depolarizing, `recovery` | `1 mV/ms` |
| `depolarizing_ahp_depth` | Post-step minimum/AHP depth | `1.0` | Depolarizing, `recovery` | `1 mV` |
| `depolarizing_early_late_voltage_difference` | Difference between early and late mean voltage | `1.0` | Depolarizing, 100–200 vs 600–700 ms | `1 mV` |
| `depolarizing_minus50_minus40_voltage_mse` | Voltage error in the experimental −50 to −40 mV band | `1.0` | Depolarizing, `stimulus` | `1 mV` |

### Hyperpolarizing waveform MSE

This is the only `voltage_mse` component:

```text
mean(((V_simulated(t) - V_experimental(t)) / 1 mV)²)
```

It is restricted to `hyperpolarizing_pulse`; there is no depolarizing
point-by-point MSE.

### Firing rate

A sigmoid around -20 mV converts voltage into smooth threshold occupancy.
Positive occupancy changes approximate upward crossings. The squared
simulated-versus-experimental rate difference is normalized by 1 Hz.

### Mean interspike minimum voltage

Experimental spikes define fixed peak-to-next-threshold intervals. Smooth
simulated minima within those intervals estimate the inter-spike voltage floor
after the first spike and before the last spike. The metric averages the
minimum from every consecutive-spike interval and compares the simulated and
experimental absolute voltages. It does **not** subtract resting voltage and
is therefore not DBLO. Experimental intervals and smooth minima preserve
differentiability. A trace with fewer than two qualifying experimental spikes
contributes zero to this term.

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

### Early-to-late mean-voltage difference

This component computes:

```text
delta_sim = mean(V_sim[100:200 ms]) - mean(V_sim[600:700 ms])
delta_exp = mean(V_exp[100:200 ms]) - mean(V_exp[600:700 ms])
loss = ((delta_sim - delta_exp) / 1 mV)²
```

It applies only to depolarizing traces, whose simulations contain both complete
windows. It is invariant to a uniform voltage shift applied to the entire
trace and specifically constrains the change between the early baseline and
late recovery periods.

### −50 to −40 mV depolarizing voltage band

During the depolarizing stimulus, this component selects samples where:

```text
-50 mV <= V_experimental(t) <= -40 mV
```

It computes point-by-point simulated-versus-experimental voltage MSE at those
fixed samples, normalized by 1 mV. The experimental trace—not the simulated
trace—defines band membership, so the mask remains fixed during optimization
and the loss remains differentiable in simulated voltage.

## Aggregation

The objective uses `protocol_mean`:

| Protocol | Protocol total | Per selected trace |
|---|---:|---:|
| Depolarizing | `0.8` | `0.4` |
| Hyperpolarizing | `0.2` | `0.1` |

`renormalize_protocol_filtered_components: false` means depolarizing-only
components retain a total factor of `0.8`, while the hyperpolarizing MSE retains
a total factor of `0.2`.

All component weights and normalization scales are `1.0`, but this does not
make their realized contributions numerically identical. The loss primitives
have different units and distributions, and the protocol allocation still
multiplies depolarizing terms by `0.8` and the hyperpolarizing term by `0.2`.

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
firing rate and the penalty use smooth threshold occupancy, while the
interspike voltage, spike height, and AHP depth use smooth extrema. MSE,
pseudo-Huber, and derivative terms are continuous and compatible with JAX
automatic differentiation.

## Related configurations

`LSU_1_wide_bounds.yaml`, `LSU_1_wide_bounds_adam.yaml`, and
`configs/search/LSU_1_cma_adam.yaml` inherit this same objective while changing
initialization or optimization strategy.
