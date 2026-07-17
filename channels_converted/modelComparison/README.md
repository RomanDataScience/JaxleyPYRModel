# Combe2023 Model Comparison

This folder compares:

- `Combe2023/cell_setup_pc2b_CCh_driven.hoc` loaded directly in NEURON.
- The one-to-one Jaxley Combe channel-placement port from
  `JaxleyModel/model/model_Combe.py`, using `JaxleyModel/model/CELL.SWC`.

Default protocol:

- `0.3 nA` somatic current step
- `300 ms` duration
- `100 ms` delay
- `500 ms` total simulation time
- `0.025 ms` integration step

Run:

```bash
bash channels_converted/modelComparison/run_comparison.sh
```

Outputs are written to `channels_converted/modelComparison/results/`:

- `neuron_step.npz`
- `jaxley_step.npz`
- `combe_vs_jaxley_step.png`
- `metrics.csv`

For a quick smoke test:

```bash
bash channels_converted/modelComparison/run_comparison.sh --tstop 20 --delay 5 --duration 5
```

For a faster full-protocol check, use a coarser Jaxley compartment density:

```bash
bash channels_converted/modelComparison/run_comparison.sh --jaxley-d-lambda 0.5 --output-dir channels_converted/modelComparison/results_combe
```

For the stronger current-step check used during model alignment:

```bash
bash channels_converted/modelComparison/run_comparison.sh --jaxley-d-lambda 0.3 --amp 0.9 --output-dir channels_converted/modelComparison/results_combe_lambda03_amp09
```

The Jaxley Combe port uses path distance from `soma(0)`, matching the HOC
`soma distance()` convention for distance-dependent passive and channel
placement.

If you change `--dt`, the Jaxley stimulus is rebuilt at the same `dt`. The n-th
entry of the Jaxley stimulus is applied at the n-th simulation step.
