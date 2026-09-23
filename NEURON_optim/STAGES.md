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
    one perturbed CMA-ES study per basin and Stage 2 seed
```

Stage 2 is the final stage. It does not produce starting points for another
stage.

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

## Stage 1: hyperpolarizing optimization

For each Stage 1 seed, the code builds a full-model starting vector by:

1. taking the Stage 0 passive solution;
2. inserting it into the full parameter space;
3. leaving active parameters at their configured default values;
4. applying a deterministic ±15% perturbation only to passive coordinates.

Each Stage 1 seed then runs an independent full-model CMA-ES search against the
hyperpolarizing objective. The Stage 1 studies do not start from the same exact
point because their passive perturbations are seed-dependent.

At the end of Stage 1, all saved generations from every Stage 1 seed are used
to create basin records. The code first reserves up to ten of the best
candidates per seed, removes duplicate normalized vectors, and then backfills
with the globally best remaining candidates across all seeds and generations.
Selection stops at `min(100, number of available candidates)`.

Each basin stores both normalized and physical parameter vectors, along with
its source seed, source generation, rank, and loss. The records are written to:

```text
stage1_hyper/basins.jsonl
```

Stage 2 does not use only the single best Stage 1 candidate. It uses every
retained basin as a separate starting solution.

## Stage 2: depolarizing optimization

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
deterministic perturbation of the basin starting point. Different basins and
seeds represent separate starting solutions.

## Counts for the smoke configuration

The smoke configuration has one Stage 1 seed, five generations, and a
population of 16. Therefore it has up to `5 × 16 = 80` basin candidates and
can produce up to 80 basins. It has one Stage 2 seed, so it creates one Stage 2
study per basin.

The normal checked-in hyperpolarizing configuration has three Stage 1 seeds,
100 generations, and 20 candidates per generation, giving up to 6,000 basin
candidates. The basin cap still limits selection to 100, so ten Stage 2 seeds
produce up to 1,000 Stage 2 studies.

The larger 100-basin design reaches the cap as soon as the Stage 1 history
contains at least 100 distinct candidates.

## Important resume rule

Each study has its own checkpoint and initial mean. A resumed study continues
from its own checkpoint; it does not recompute its initial point. Changing the
objective or its weights changes the configuration hash, so old checkpoints
are intentionally treated as incompatible with the new loss definition.
