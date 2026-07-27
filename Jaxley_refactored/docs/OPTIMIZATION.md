# Optimization and adaptive step sizes

The default configuration uses projected Adam with a fixed learning rate.
Parameters are optimized in normalized `[0, 1]` coordinates and projected back
into that interval after every proposal.

The optional backtracking method separates two responsibilities:

1. Adam computes a direction from the loss gradient accumulated over every
   trace bucket.
2. A line search tries that direction with a large learning rate, evaluates the
   configured objective over all traces, and halves the rate until the loss is
   strictly lower.

Only an accepted proposal commits Adam's moment estimates. After acceptance,
the next epoch starts with a slightly larger rate. If no proposal is accepted,
the parameters and moments remain unchanged and the next epoch starts with the
smaller rate reached by the search.

Use the supplied configuration:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/optimizers/adam_backtracking.yaml \
  --cells m20240527cd \
  --epochs 50
```

For a quick compilation and behavior check:

```bash
bash scripts/run_full_fitting.sh \
  --config configs/optimizers/adam_backtracking.yaml \
  --cells m20240527cd \
  --epochs 3 \
  --max-steps 100
```

Each epoch log reports `lr`, `accepted`, and `trials`. The matching
`metrics.jsonl` entry also contains `loss_before_step`. With backtracking,
`loss` and the epoch plot describe the accepted post-step parameters. A
rejected epoch has `accepted=false`, unchanged parameters, and equal
`loss_before_step` and `loss`.

Backtracking can require several forward simulations per epoch. It does not
repeat the backward pass: the gradient and Adam direction are computed once.
All traces in each shape bucket still run through the same `vmap` kernel, and
both natural trace buckets contribute to every candidate's acceptance test.

The main controls are:

- `learning_rate`: first trial rate at a fresh run.
- `maximum_learning_rate`: ceiling after successful growth.
- `minimum_learning_rate`: floor when shrinking.
- `reduction_factor`: multiplier after rejection; `0.5` halves the step.
- `growth_factor`: multiplier for the next epoch after acceptance.
- `maximum_trials`: maximum forward evaluations per epoch.

Because the acceptance test uses the configured objective, it guarantees a
strict decrease in that objective for every accepted update. It does not imply
that every individual trace loss decreases.
