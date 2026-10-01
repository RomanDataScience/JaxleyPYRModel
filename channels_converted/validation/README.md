# Channel Validation

These scripts compare each converted Jaxley channel against the compiled NEURON
MOD mechanism under the same voltage-clamp protocol.

Run from the repository root with the `Jaxley` conda environment active:

```bash
python channels_converted/validation/compare_channel.py --compile --channel nax --plot
```

Then inspect:

```text
channels_converted/validation/results/nax_traces.npz
channels_converted/validation/results/nax_overlay.png
channels_converted/validation/results/metrics.csv
```

Validate all non-skipped channels:

```bash
python channels_converted/validation/compare_channel.py --all --plot
```

Validate and plot every registered mechanism, including `cal4` and `d3`:

```bash
python channels_converted/validation/compare_channel.py --all --include-skipped --plot
```

The `--plot` flag writes one overlay image per channel and, for `--all`, also
writes:

```text
channels_converted/validation/results/all_channels_metric_summary.png
```

List available channels:

```bash
python channels_converted/validation/compare_channel.py --list
```

Override parameters with either full Jaxley names or unprefixed MOD names:

```bash
python channels_converted/validation/compare_channel.py --channel nax --param gbar=0.02
python channels_converted/validation/compare_channel.py --channel nax --param Nax_gbar=0.02
```

`kir` has `gbar = 0` in both isolated mechanism defaults. Use the Combe setup
value to generate a nonzero KIR comparison:

```bash
python channels_converted/validation/compare_channel.py --channel kir --param gbar=0.00101535 --plot
```

Notes:

- `NEURON_MODULE_OPTIONS=-nogui` is set by the runner so NEURON can run headless.
- The default protocol is a voltage-step family: hold at `-80 mV`, step through
  `[-90, -70, -50, -30, -10, 10, 30] mV`, then return to `-80 mV`.
- `cal4` is skipped by `--all` unless `--include-skipped` is passed: without a
  calcium current its concentration stays at rest. Channels that read calcium
  (`cal`, `cat`, `kca`, `mykca`) are checked here with `cai` held at the
  common default, so this script cannot see how they work together; use
  `compare_calcium.py` for that (below).
- Channel conductances default to their MOD values, which are 0 for several
  channels (e.g. `cal`, `cat`); their current rows then compare 0 with 0. Pass
  `--param gbar=...` (or the channel's conductance name) for a nonzero check.

## Coupled calcium validation

```bash
python channels_converted/validation/compare_calcium.py --plot
```

One compartment with `cal` + `cat` (calcium entry) + `cal4` (calcium pool) +
`kca` + `mykca`, under the same voltage clamp in NEURON and in Jaxley's
`jx.integrate` (the path the fitted model uses), at 20, 2 and 0.6 um diameter.
Protocol: hold -70 mV, 20 pulses to 0 mV (2 ms, 50 Hz), 200 ms at -10 mV,
600 ms recovery. It compares free calcium in all four annuli, the calcium
current and both calcium-activated K currents, and writes
`results/calcium_metrics.csv`, `results/calcium_diam*.npz` and
`results/calcium_overlay.png`.

Pass criterion: max |NEURON - Jaxley| within 5% of the signal range. Jaxley
records a step's `i_Ca` at the start of the step (one sample behind NEURON), and
NEURON's inner-annulus variables are only set after the first step; both are
aligned before comparing.

Current result (dt 0.025 ms): 20 of 21 signals pass (0.01-2.5% of range). The
calcium current in the 0.6 um compartment reaches 5.0% on a single sample
0.05 ms after a voltage jump. It does not shrink with dt: NEURON advances
`cai` with the current of the same step, while Jaxley updates channel and pump
states before it computes that step's currents, so `cal4` sees each step's
calcium current one step later. In a 0.6 um compartment calcium rises fast
enough for this to show on the tail current for ~0.2 ms; free calcium itself
stays within 1.6%.
- `na16a` uses an implicit sparse-style update for the four-state Markov chain,
  matching NEURON's `KINETIC ... METHOD sparse` stepping for fixed voltage steps.
