# Optimization stages and solution handoff

The pipeline has three numbered optimization stages:

```text
Stage 0: passive pre-calibration
    best passive solution
            |
            v
Stage 1: full-model hyperpolarizing optimization
    all-generation candidates -> retained basins
            |
            v
Stage 2: full-model depolarizing optimization
    AP-core parameter subset; one study per basin and Stage 2 seed
            |
            v
Stage 3: expanded active-current depolarizing optimization
    all active parameters; Stage 2 subset constrained locally
```

Stage 3 is the final stage. It starts from the best Stage 2 result for each
basin and does not produce starting points for another stage.

The parameter dimensions are:

| Stage | Free parameters | What is free |
|---|---:|---|
| Stage 0 | 12 | Passive membrane and cable parameters |
| Stage 1 | 44 | All 12 passive parameters plus all 32 active parameters |
| Stage 2 | 22 | AP-core sodium, potassium, and kinetic parameters |
| Stage 3 | 32 active parameters | The Stage 2 set plus the remaining calcium, H, Kir, CAN, and dendritic K parameters |

The optimizer uses the same parameter catalog for NEURON and Jaxley. A
parameter name describes a shared model control, although the corresponding
conductance can be distributed differently across soma, axon, basal dendrite,
and apical dendrite.

## Stage 0: passive pre-calibration

Stage 0 first checks that the raw current waveform is replayed correctly. It
then fits only the passive parameter subset, with active conductances disabled.
The configured passive seeds run independent CMA-ES studies. The study with
the lowest passive loss is selected.

The selected physical passive values are saved in:

```text
stage0_passive/passive_best.json
```

This is a single selected solution, not the complete final population.

### Stage 0 parameter roles

| Parameters | Mechanism or model role | Main voltage features affected |
|---|---|---|
| `RmSoma`, `RmTuft` | Specific membrane resistance at the soma and tuft ends of the distance-dependent leak profile | Input resistance, steady-state current-step amplitude, and the passive component of the membrane time constant |
| `RaSoma`, `RaTuft` | Axial resistivity at the soma and tuft ends of the cable profile | Electrotonic coupling, dendritic filtering, and how quickly voltage changes reach the soma |
| `DistHalfRm`, `SlopeRm` | Half-distance and transition slope for the soma-to-tuft `Rm` profile | Where the cell changes from somatic to dendritic leak behavior and how gradual that transition is |
| `DistHalfRa`, `SlopeRa` | Half-distance and transition slope for the `Ra` profile | Spatial cable filtering and dendritic voltage attenuation |
| `Epas` | Leak reversal potential | Resting voltage and the direction of the passive current response |
| `CmSoma` | Specific membrane capacitance; dendritic capacitance is additionally scaled by spine factors | Passive charging speed, spike width, and the size/timing of fast voltage transients |
| `SpineFactorBasal`, `SpineFactorTuft` | Effective area, leak, and capacitance multipliers for basal and apical compartments | Dendritic load, effective input resistance, charging transients, and propagation into the soma |

During Stage 0 every active conductance is set to zero. This makes the stage
the cleanest place to identify the passive amplitude and charging behavior.

## Stage 1: hyperpolarizing optimization

For each Stage 1 seed, the code builds a full-model starting vector by:

1. taking the Stage 0 passive solution;
2. inserting it into the full parameter space;
3. leaving active parameters at their configured default values;
4. applying a deterministic ±15% perturbation only to passive coordinates.

Each Stage 1 seed then runs an independent full-model CMA-ES search against the
hyperpolarizing objective. The Stage 1 studies do not start from the same exact
point because their passive perturbations are seed-dependent.

Stage 1 uses the complete 44-dimensional parameter space. The passive
parameters listed above remain free, and all 32 active parameters are also
free, although they begin at their configured defaults. Consequently, Stage 1
can use active conductances to improve a hyperpolarizing trace; its active
parameter estimates should therefore be treated as provisional rather than as
independent measurements of depolarizing physiology.

Stage 1 evaluates the configured representative hyperpolarizing recording
(`data.hyperpolarizing_trace_index: 0` in the checked-in configs). Its objective matches the voltage
trajectory across pre-step, stimulus, and recovery regions, adds a deflection
term, and penalizes spikes during a hyperpolarizing trial. The four configured
trace names remain available for Stage 2's depolarizing objective.

At the end of Stage 1, all saved generations from every Stage 1 seed are used
to create basin records. The code first reserves up to
`stage1.basin_quota_per_seed` candidates per seed, defaulting to ten, removes
duplicate normalized vectors, and then backfills with the globally best
remaining candidates across all seeds and generations. Selection stops at
`min(100, number of available candidates)`.

Each basin stores both normalized and physical parameter vectors, along with
its source seed, source generation, rank, and loss. The records are written to:

```text
stage1_hyper/basins.jsonl
```

Stage 2 does not use only the single best Stage 1 candidate. It uses every
retained basin as a separate starting solution.

## Stage 2: AP-core depolarizing optimization

For every basin and every configured Stage 2 seed, the code creates a separate
initial mean:

```text
initial_mean = clip(
    basin_normalized_parameters + Uniform(-0.15, +0.15),
    0, 1,
)
```

The perturbation is deterministic from the pair `(stage2_seed, basin_id)`, so
the same run can be reproduced. The perturbed vector is used as the initial
mean of a full-model CMA-ES study for the depolarizing objective.

The resulting directory structure is:

```text
stage2_depolarizing/
  b000/seed_000/
  b000/seed_001/
  b001/seed_000/
  ...
```

The Stage 2 seed therefore controls both the CMA-ES random sequence and the
deterministic perturbation of the selected coordinates. Different basins and
seeds represent separate starting solutions.

Stage 2 evaluates all four configured depolarizing recordings. Its loss is the
weighted mean of these seven components:

| Component | Weight |
|---|---:|
| Voltage trajectory | 12/120 = 10% |
| Spike shape | 10/120 = 8.33% |
| Spike symmetry | 10/120 = 8.33% |
| Spike height | 10/120 = 8.33% |
| Plateau | 24/120 = 20% |
| Return to baseline | 24/120 = 20% |
| Firing rate during the current step | 30/120 = 25% |

The firing-rate term compares the experimental and simulated spike counts within
the depolarizing current step, converted to Hz using the step duration. Spike
symmetry and height are evaluated for matched experimental and simulated spikes.
Extra spikes outside the allowed timing window can trigger the configured large
penalty.

### Stage 2 calibrated parameters and roles

Stage 2 frees the 22 parameters most directly connected to action-potential
generation, spike shape, sustained depolarization, and firing rate. The
passive values and the ten active parameters reserved for Stage 3 are fixed to
the Stage 1 basin values during this stage.

| Parameters | Mechanism or model role | Expected influence on the depolarizing response |
|---|---|---|
| `gna` | Main somatic Nav1.6 conductance and the base axonal sodium conductance | Spike threshold, upstroke strength, spike amplitude, and excitability |
| `gnadend` | Dendritic Nav1.6/Na3 conductance scale | Dendritic recruitment, back-propagation, and support for prolonged depolarization |
| `AXNa` | Axonal sodium conductance multiplier | Axonal spike initiation and reliable propagation into the soma |
| `scale_Na_conduct` | Shared multiplier for the soma/apical Nav1.6 conductances | Strongly changes sodium drive; interacts multiplicatively with `gna` and `gnadend` |
| `persist` | Nav1.6 `I1 → O1` recovery rate, producing persistent channel availability | Sustained inward sodium current, plateau height, extra spikes, and depolarization-block risk; bounded to 0–0.02 |
| `nat_fast_inactivation_tau_scale` | Nav1.6 and related sodium fast-inactivation time scale | Spike width, sodium current termination, and recovery between spikes |
| `nat_slow_recovery_tau_scale` | Nav1.6 slow `I1 ↔ I2` pathway time scale | Slow recovery/adaptation and the ability to sustain or terminate repetitive firing |
| `gkv2soma`, `gkv2`, `gkv2axon`, `gkv2scale` | Kv2-like delayed-rectifier potassium conductances and their dendritic scale | Repolarization, interspike voltage, firing rate, and suppression of an excessive plateau |
| `soma_kap`, `axon_kap`, `basal_kap` | A-type potassium conductances in soma/apical, axon, and basal dendrite | Early outward current, spike initiation timing, spike width, and adaptation |
| `soma_kad` | Apical A-type potassium conductance profile | Dendritic excitability and the shape of the voltage trajectory around spike onset |
| `gkdrsoma`, `axongkdr`, `gkdrapical` | Delayed-rectifier K conductance in soma, axon, and apical dendrite | Late repolarization, spike frequency, afterdepolarization, and plateau control |
| `kd_deactivation_tau_scale` | Delayed-rectifier K deactivation kinetics | Duration of outward current, interspike recovery, and late action-potential shape |
| `soma_kca`, `mykca_init` | Calcium-activated K conductances | Afterhyperpolarization, spike-frequency adaptation, and termination of sustained firing |
| `soma_km` | Slowly activating, non-inactivating M-type K conductance | Subthreshold adaptation, firing-rate control, and slow return from depolarization |

`persist` and `scale_Na_conduct` should be interpreted jointly. A modest
`persist` value can have a large effect when the fitted Nav1.6 conductance is
also large, because the persistent current scales with the underlying sodium
conductance.

## Stage 3: expanded active-current optimization

Stage 3 selects the best Stage 2 result within each basin, then adds the
remaining active-current parameters from `stage3.parameter_names`. The Stage 2
parameters remain trainable, but their bounds are restricted to
`±stage3.local_normalized_half_width` around their Stage 2 values in normalized
coordinates. The default half-width is `0.15`. Passive parameters remain fixed
to the Stage 1 basin values.

Stage 3 uses the same depolarizing objective and weights as Stage 2. Its output
is written under `stage3_depolarizing/`.

### Stage 3 additions and roles

Stage 3 makes the remaining ten active parameters free. These parameters were
held at their Stage 1 basin values in Stage 2, then initialized at those same
values in Stage 3. The newly added parameters use their normal catalog bounds;
the Stage 2 parameters use the local ±15% normalized bounds described above.

| Parameters | Mechanism or model role | Expected influence on the depolarizing response |
|---|---|---|
| `gkdrdend` | Basal dendritic delayed-rectifier K conductance | Dendritic repolarization and attenuation of sustained dendritic depolarization |
| `soma_caL`, `soma_car` | L- and R-type calcium conductance controls | Depolarizing inward current, calcium entry, plateau support, and spike-afterpotential shape |
| `gsomacar` | Somatic R-type calcium conductance | Somatic inward support during the plateau and calcium-dependent adaptation |
| `soma_caLH` | High-voltage-activated dendritic/somatic calcium conductance profile | Late inward current and prolonged depolarization |
| `soma_caT` | T-type calcium conductance | Low-threshold activation, early depolarization, bursting, and spike timing |
| `icangbar` | Calcium-activated nonspecific cation conductance (`I_CAN`) | Slow inward depolarization, plateau maintenance, and post-spike depolarization |
| `soma_hbar` | H-current conductance | Resting potential, sag, subthreshold excitability, and rebound behavior |
| `KirGbar` | Inward-rectifier potassium conductance | Resting stabilization, suppression of depolarized states, and recovery toward baseline |
| `h_tau_scale` | H-current kinetics | Timing of sag and rebound, and slow recovery after a current step |

Stage 3 is therefore intended to refine plateau duration, return to baseline,
and slow firing behavior after Stage 2 has established a plausible spike
shape. It does not change the loss function.

## Active parameters that remain fixed

Several Nav1.6 controls exist in the mechanism or original HOC setup but are
not free optimizer parameters. They preserve the established spatial model
profiles while the staged calibration adjusts the conductances and selected
kinetic scales:

| Fixed control | Current implementation | Why it remains fixed |
|---|---|---|
| `dist` | Soma uses the configured Nav1.6 distance factor; apical values follow the distance-dependent profile | Controls entry into the slow-inactivated `I2` state and encodes spatial channel specialization |
| `slowdown` | Soma and apical compartments use fixed profile values (`slowsoma` and `slownotsoma`) | Sets the time scale of the slow-inactivation pathway; adding it as a free parameter would introduce another strong plateau/adaptation lever |
| `C1O1v2` / `proximalv` / `distalv` | Fixed voltage-dependence profile | Changes Nav1.6 activation and would be strongly confounded with sodium conductance and `persist` |
| `sinfsoma` | Fixed soma Nav1.6 distance/profile control | Preserves the original channel distribution convention |
| `psoma` | Legacy HOC variable; the optimizer's active parameter is `persist` | Avoids calibrating two names for the same persistent Nav1.6 mechanism |
| `icand_can`, `gip3` | Fixed at their model defaults in the current catalog | Keeps the calcium-dependent signaling/background model outside the AP-core search |
| `nap` | Not calibrated; removed from the converted comparison model where appropriate | It is not an independently active current in the calibrated model configuration |

In particular, the `dist = 0` and `slowdown = 0.2` values shown in the raw
MOD-file defaults should not be read as the values used everywhere in the
cell. The model builder overwrites them with soma and distance-dependent
profiles before simulation.

## Counts for the smoke configuration

The smoke configuration has one Stage 1 seed, five generations, and a
population of 16. Therefore it has up to `5 × 16 = 80` basin candidates and
can produce up to 80 basins. It has one Stage 2 seed, so it creates one Stage 2
study per basin.

The normal checked-in hyperpolarizing configuration has three Stage 1 seeds,
100 generations, and 20 candidates per generation, giving up to 6,000 basin
candidates. The basin cap still limits selection to 100, so ten Stage 2 seeds
produce up to 1,000 Stage 2 studies. The default Stage 3 configuration then
creates one study per basin using the best Stage 2 seed for that basin.

The larger 100-basin design reaches the cap as soon as the Stage 1 history
contains at least 100 distinct candidates.

## Important resume rule

Each study has its own checkpoint and initial mean. A resumed study continues
from its own checkpoint; it does not recompute its initial point. Changing the
objective or its weights changes the configuration hash, so old checkpoints
are intentionally treated as incompatible with the new loss definition.
