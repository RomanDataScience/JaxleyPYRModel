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
- `cal4` is skipped by `--all` unless `--include-skipped` is passed because the
  current Jaxley version is reduced; the MOD mechanism uses radial annuli,
  buffering, KINETIC reactions, and longitudinal diffusion.
- `na16a` uses an implicit sparse-style update for the four-state Markov chain,
  matching NEURON's `KINETIC ... METHOD sparse` stepping for fixed voltage steps.
