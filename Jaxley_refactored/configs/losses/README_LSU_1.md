# LSU_1 configuration reference

[`LSU_1.yaml`](LSU_1.yaml) is a differentiable, feature-aware objective for
matching the electrophysiological behavior of the selected experimental
traces. It deliberately does not use depolarizing whole-trace MSE as the sole
criterion: small spike-timing offsets can make pointwise errors large even
when the relevant physiology is similar.

For hyperpolarizing traces, the primary feature is the voltage level of the
trough relative to the stable pre-pulse baseline. The depolarizing priorities
remain:

1. depolarizing firing rate;
2. no depolarizing spikes outside the current step and no hyperpolarizing
   spikes anywhere;
3. interspike voltage floor and rounded/symmetric trough geometry;
4. a deep, long AHP followed by slow recovery to baseline from below;
5. spike height.

## Data and optimizer

LSU_1 inherits Adam with backtracking from
`configs/optimizers/adam_backtracking.yaml`. Local, exploratory Adam, and
CMA–Adam configurations inherit the same objective.

The training set uses the first and third trace mapped to each protocol,
independent of cell-specific trace names.

- Depolarizing traces run through 500 ms after current offset.
- Hyperpolarizing traces run through 100 ms after offset and end at 650 ms.
- `stimulus` is onset through offset.
- `outside_stimulus` is baseline plus recovery.
- `recovery` is after offset through the simulation endpoint.

## Complete objective

All additive components have raw weight `1.0` except the primary
hyperpolarizing trough-depth term, whose weight is `4.0`, and the dedicated
−50 to −40 mV plateau-band term, whose weight is `0.75`. Every normalization
scale remains `1.0` in its native units. The derivative scale is expressed in
mV/ms; voltage-feature scales are expressed in mV.

| Component | Biological role | Protocol/window |
|---|---|---|
| `hyperpolarizing_trough_depth` | Primary hyperpolarizing target: stable-baseline-to-smooth-trough depth; weight `4.0` | Hyperpolarizing, `stimulus` |
| `hyperpolarizing_waveform_mse` | Hyperpolarizing voltage trajectory | Hyperpolarizing, `score` |
| `hyperpolarizing_derivative_mse` | Hyperpolarizing dV/dt trajectory, including onset, sag, offset, and recovery kinetics | Hyperpolarizing, `score` |
| `depolarizing_firing_rate` | Firing frequency | Depolarizing, `stimulus` |
| `depolarizing_forbidden_spikes` | No spikes before or after depolarizing pulse | Depolarizing, `outside_stimulus` |
| `hyperpolarizing_forbidden_spikes` | No spikes anywhere in hyperpolarizing trace | Hyperpolarizing, `full_trace` |
| `depolarizing_interspike_minimum_voltage` | Absolute voltage floor between consecutive spikes | Depolarizing, `stimulus` |
| `depolarizing_interspike_trough_shape` | Trough position, rounded width, and asymmetry | Depolarizing, `stimulus` |
| `depolarizing_spike_waveform` | Robust stimulus-waveform regularizer | Depolarizing, `stimulus` |
| `depolarizing_spike_height` | Height of every experimental-aligned spike | Depolarizing, `stimulus` |
| `depolarizing_recovery_waveform` | Detailed post-step trajectory | Depolarizing, `recovery` |
| `depolarizing_ahp_depth` | AHP depth relative to pre-step baseline | Depolarizing, `recovery` |
| `depolarizing_ahp_duration` | Mean below-baseline deficit over recovery | Depolarizing, `recovery` |
| `depolarizing_early_late_voltage_difference` | Baseline versus early AHP, 100–200 and 600–700 ms | Depolarizing |
| `depolarizing_terminal_baseline_difference` | Baseline versus recovery tail, 100–200 and 900–1000 ms | Depolarizing |
| `depolarizing_minus50_minus40_voltage_mse` | Pointwise voltage error in the experimental −50 to −40 mV band; weight `0.75` | Depolarizing, `stimulus` |

## 1. Firing rate

`soft_firing_rate_error` maps voltage to smooth threshold occupancy:

```text
p(t) = sigmoid((V(t) - threshold) / temperature)
```

Positive changes in `p(t)` approximate upward spike crossings. Their sum is
divided by stimulus duration to obtain a differentiable firing-rate surrogate.
LSU_1 compares simulated and experimental rates during the depolarizing step
using threshold −20 mV and temperature 2 mV.

This directly targets firing frequency without differentiating through a hard
integer spike detector.

## 2. Forbidden spikes

Two additive soft-count losses make the allowed regions explicit:

- depolarizing: count crossings only in `outside_stimulus`;
- hyperpolarizing: count crossings over `full_trace`, including during the
  hyperpolarizing pulse.

Each additive loss is the squared soft count. Two multiplicative safeguards
then apply:

```text
final_loss =
    base_loss
    × 1.1 ^ N_depolarizing_outside
    × 1.1 ^ N_hyperpolarizing_anywhere
```

The additive terms remain nonzero even if other components become very small;
the multipliers increase the full objective for prohibited spikes.

## Hyperpolarizing trough depth

The principal hyperpolarizing feature is the voltage deflection from the
stable 400–500 ms pre-pulse baseline to the lowest voltage reached during the
500–550 ms current pulse. For each trace:

```text
depth(V) = mean(V[400:500 ms]) - softmin(V[500:550 ms])
loss = 4 × ((depth(V_sim) - depth(V_exp)) / 1 mV)²
```

The soft-minimum temperature is `0.5 mV`. This is narrow enough to represent
the rounded trough while distributing gradients over nearby low-voltage
samples instead of selecting one hard `argmin`. Restricting the baseline to
the scored interval explicitly excludes the startup transient before 400 ms.
The depth is invariant to a common voltage offset; waveform MSE separately
anchors absolute baseline and trajectory. Its raw weight `4.0` is four times
the waveform and derivative weights, making trough depth the dominant
hyperpolarizing feature.

## 3. Interspike minimum and rounded trough shape

Experimental upward crossings define fixed consecutive-spike intervals. Each
interval begins after the preceding observed spike peak and ends before the
next upward crossing. Fixing interval topology from the observation prevents
simulated hard spike detection from entering the differentiation graph.

### Absolute interspike minimum

A normalized smooth minimum is computed independently in every interval. LSU_1
averages the **per-interval squared errors**, so a trough that is 5 mV too high
cannot cancel another that is 5 mV too low. Resting voltage is not subtracted;
this is an absolute interspike-floor metric, not DBLO.

### Trough geometry

Within each interval, soft-minimum probabilities emphasize low-voltage
samples. They define three normalized phase features:

- center: where the trough occurs between consecutive spikes;
- width: whether the minimum is sharp or broad/rounded;
- asymmetry: whether descent and renewed depolarization have the same geometry.

The component compares all three simulated and experimental features per
interval. It does not force perfect mathematical symmetry; it reproduces the
symmetry actually present in the experimental trace.

### −50 to −40 mV band and robust waveform

The voltage-band MSE uses only stimulus samples where the **experimental**
voltage lies inclusively between −50 and −40 mV. The fixed observation-derived
mask preserves differentiability and adds local information around the
interspike voltage region.

A pseudo-Huber stimulus waveform term remains as a robust regularizer. Unlike
MSE, pseudo-Huber growth becomes approximately linear for large residuals, so
a small spike-time shift is less dominant.

## 4. Slow AHP and return to baseline

The recovery objective uses complementary summaries:

- `depolarizing_recovery_waveform`: robust pointwise recovery shape;
- `depolarizing_ahp_depth`: pre-step baseline minus smooth recovery minimum;
- `depolarizing_ahp_duration`: mean smooth voltage deficit below baseline
  across the entire 500 ms recovery;
- `depolarizing_early_late_voltage_difference`: compares
  `mean(V[100:200]) - mean(V[600:700])`;
- `depolarizing_terminal_baseline_difference`: compares
  `mean(V[100:200]) - mean(V[900:1000])`.

Depth alone cannot distinguish a brief AHP from a long one. The mean
below-baseline deficit encodes both depth and duration, while the terminal
window checks whether the trace approaches the experimental baseline from
below after several hundred milliseconds.

The raw recovery derivative MSE was removed. At scale 1 it was several orders
of magnitude more sensitive than the voltage features to sub-millisecond
timing shifts and did not specifically encode slow AHP physiology.

For hyperpolarizing traces, by contrast, LSU_1 retains first-derivative MSE
over the scored 400–650 ms interval. These traces contain no action potentials,
so dV/dt directly complements voltage MSE by constraining pulse onset, sag,
offset, and passive return kinetics without spike-time-shift amplification:

```text
mean(((diff(V_sim) / dt - diff(V_exp) / dt) / 1 mV/ms)²)
```

Only adjacent sample pairs fully inside the score window contribute.

## 5. Spike height

Observed spike peaks define fixed ±3 ms windows. A smooth maximum is calculated
inside every window, and LSU_1 averages the per-spike squared peak-voltage
errors. This prevents one correctly tall spike from hiding several short
spikes, which could happen with a single maximum over the entire current step.

## Aggregation

The objective uses `protocol_mean`:

| Protocol | Protocol total | Per selected trace |
|---|---:|---:|
| Depolarizing | `0.7` | `0.35` |
| Hyperpolarizing | `0.3` | `0.15` |

Protocol-filtered components are not renormalized. Thus raw component weights
do not directly equal final influence: depolarizing-only components retain the
`0.7` allocation and hyperpolarizing-only components retain `0.3`. The plateau
band has a pre-metric coefficient of `0.75 × 0.7 = 0.525`.

Likewise, setting every native-unit scale to `1.0` does not statistically
equalize the metrics. Firing-rate error is measured in Hz, voltage features in
mV, phase features are dimensionless, and their empirical variances differ.
For publication-quality calibration, these values should eventually be
normalized by prespecified biological tolerances or experimental variability,
and component losses and gradient norms should be reported.

## Differentiability and limitations

All simulated-voltage operations are continuous or piecewise differentiable:
sigmoid threshold occupancy, smooth extrema, pseudo-Huber loss, fixed masks,
and ordinary means.

Experimental spike timing supplies the interval and peak-window topology. This
is necessary for stable gradients but means a simulated trace with a severely
wrong spike count can place the wrong waveform inside a fixed interval.
Firing-rate and forbidden-spike terms therefore remain essential. Final
solutions should also be checked with discrete, non-differentiable spike
counts and held-out traces.

## Methodological precedents

- [Druckmann et al. (2007)](https://doi.org/10.3389/neuro.01.1.1.001.2007)
  used multiple electrophysiological objectives—including firing rate, spike
  overshoot, AHP depth, and interspike trough minima—and discussed combining
  feature objectives with direct trace error.
- [Gouwens et al. (2018)](https://doi.org/10.1038/s41467-017-02718-3)
  fitted firing frequency, AP peak, fast/slow trough depth, and trough timing
  within the interspike interval. This motivates per-spike peak and trough
  phase/shape features.
- [Van Geit et al. (2016), BluePyOpt](https://doi.org/10.3389/fninf.2016.00017)
  formalized feature-based multi-objective optimization of conductance-based
  neuron models.
- [Deistler et al. (2025), Jaxley](https://doi.org/10.1038/s41592-025-02895-w)
  emphasized differentiable summaries for spike-related objectives and used
  timing-tolerant trace comparison for spiking responses.
- [Gulledge et al. (2013)](https://doi.org/10.1523/JNEUROSCI.0220-13.2013)
  documented long-lasting activity-dependent AHPs in pyramidal neurons,
  supporting explicit depth, duration, and late-recovery constraints.
