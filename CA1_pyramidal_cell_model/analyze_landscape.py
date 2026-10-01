#!/usr/bin/env python3
"""Loss landscape and parameter degeneracy of CA1 model fits to one cell.

Collects every evaluated parameter vector from one or more CMA-ES runs of the
same cell (``evaluations.jsonl``; runs made before that file existed fall back
to the top candidates saved per generation) and writes, to
``plots/landscape_<cell>/``:

- ``loss_vs_parameters.png``: loss against each parameter within its bounds.
- ``embedding.png``: UMAP and PCA of the bound-normalized parameter vectors,
  coloured by loss; good fits are outlined and clustered in UMAP space.
- ``good_fits_parameters.png``: every good fit as a line across the
  normalized parameters, coloured by cluster: which parameters are pinned and
  which are free, and how clusters differ.
- ``good_fits_correlation.png``: Spearman correlations between parameters
  within the good fits; strong negative pairs are candidate compensations.
- ``param_importance.png``: Optuna's fANOVA hyperparameter importance, the
  share of the (log) loss variance each parameter explains on its own, for the
  joint loss and for each trace's loss (values in ``param_importance.csv``).
- ``landscape.csv``: all points with loss, good-fit flag, embedding and cluster.

A good fit is a loss at most ``--good-factor`` times the best loss.

CMA-ES samples follow the optimizer's own search distribution, so densities
and correlations are biased towards the path it took. Several seeds per cell
make the picture more trustworthy.

Usage::

    python analyze_landscape.py runs/cma_m20240527cd
    python analyze_landscape.py runs/cma_m20240527cd runs/cma_m20240527cd_seed2 --good-factor 1.3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ca1_parameters import DEFAULTS  # noqa: E402
FAILED_LOSS = 1.0e5


def load_run(run_dir: Path) -> tuple[list[dict], str, dict[str, tuple[float, float]]]:
    """Return records ({run, generation, loss, parameters}), cell name and bounds."""
    config = json.loads((run_dir / "run_metadata.json").read_text())
    catalog = json.loads((run_dir / "parameter_catalog.json").read_text())
    bounds = {key: (spec["lower"], spec["upper"]) for key, spec in catalog["parameters"].items()}
    cell = yaml.safe_load((run_dir / "config.yaml").read_text())["cell"]
    evaluations = run_dir / "evaluations.jsonl"
    records = []
    if evaluations.exists():
        latest = {}
        for line in evaluations.read_text().splitlines():
            item = json.loads(line)
            # A generation re-run after an interrupted resume overwrites itself.
            latest[(item["generation"], item["index"])] = item
        source = "all evaluations"
        records = list(latest.values())
    else:
        source = "top candidates per generation only"
        for path in sorted((run_dir / "plots").glob("generation_*/top_candidates.json")):
            generation = int(path.parent.name.split("_")[1])
            for item in json.loads(path.read_text()):
                records.append({"generation": generation, "loss": item["loss"],
                                "parameters": item["parameters"],
                                "traces": item.get("details", {}).get("traces", [])})
    for item in records:
        item["run"] = run_dir.name
        # Per-trace losses; the pulse can reuse a step's trace name, so key by kind.
        item["trace_losses"] = {
            f"{'hyper' if trace.get('kind', '').startswith('hyper') else 'step'} {trace['trace']}":
            float(trace["loss"])
            for trace in item.get("traces") or [] if trace.get("loss") is not None
        }
    print(f"{run_dir.name}: {len(records)} points ({source}); "
          f"{len(config['parameters'])} free parameters")
    return records, cell, bounds


def param_importances(physical: np.ndarray, target: np.ndarray, keys: list[str],
                      bounds: dict[str, tuple[float, float]], seed: int) -> dict[str, float]:
    """Optuna's fANOVA importances of each parameter for ``target`` (lower is better)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    distributions = {key: optuna.distributions.FloatDistribution(*bounds[key]) for key in keys}
    study = optuna.create_study(direction="minimize")
    study.add_trials([
        optuna.trial.create_trial(params=dict(zip(keys, map(float, row), strict=True)),
                                  distributions=distributions, value=float(value))
        for row, value in zip(physical, target, strict=True)
    ])
    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=seed)
    return optuna.importance.get_param_importances(study, evaluator=evaluator)


def _clusters(points: np.ndarray) -> np.ndarray:
    # Too few points for density clusters to mean anything: one group.
    if len(points) < 30:
        return np.zeros(len(points), dtype=int)
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=max(5, len(points) // 10)).fit_predict(points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="run directories of the same cell")
    parser.add_argument("--good-factor", type=float, default=1.25,
                        help="good fit: loss <= factor * best loss (default 1.25)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0, help="UMAP random seed")
    args = parser.parse_args()

    records, cells, bounds = [], set(), {}
    for run in args.runs:
        run_records, cell, run_bounds = load_run(run.resolve())
        records += run_records
        cells.add(cell)
        for key, value in run_bounds.items():
            if bounds.setdefault(key, value) != value:
                raise ValueError(f"{key} has different bounds across runs: {bounds[key]} vs {value}")
    if len(cells) != 1:
        raise ValueError(f"Runs mix cells {sorted(cells)}; losses are only comparable within a cell")
    cell = cells.pop()
    keys = list(bounds)
    output_dir = (args.output_dir or HERE / "plots" / f"landscape_{cell}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [item for item in records if np.isfinite(item["loss"]) and item["loss"] < FAILED_LOSS]
    loss = np.asarray([item["loss"] for item in records])
    physical = np.asarray([[item["parameters"][key] for key in keys] for item in records])
    lower = np.asarray([bounds[key][0] for key in keys])
    upper = np.asarray([bounds[key][1] for key in keys])
    normalized = (physical - lower) / (upper - lower)
    runs = np.asarray([item["run"] for item in records])
    best = float(loss.min())
    good = loss <= args.good_factor * best
    order = np.argsort(-loss)  # draw the best points last
    norm = LogNorm(vmin=best, vmax=float(np.percentile(loss, 98)))
    cmap = plt.get_cmap("viridis_r")
    print(f"{cell}: {len(loss)} points, best loss {best:.4g}, "
          f"{good.sum()} good fits (loss <= {args.good_factor:g} x best)")
    # Shown on every figure so plots from different settings cannot be mixed up.
    criterion = (f"{len(loss)} evaluations; good fit = loss <= {args.good_factor:g} x best "
                 f"({best:.3g}): {good.sum()} points")

    # 1. Loss against each parameter.
    ncols = 6
    nrows = int(np.ceil(len(keys) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows),
                                squeeze=False, constrained_layout=True)
    for ax, column, key in zip(axes.flat, range(len(keys)), keys):
        ax.scatter(physical[order, column], loss[order], c=loss[order], cmap=cmap, norm=norm,
                   s=9, linewidths=0)
        ax.scatter(physical[good, column], loss[good], facecolors="none", edgecolors="#d62728",
                   s=16, linewidths=0.6)
        ax.axvline(DEFAULTS[key], color="0.5", ls="--", lw=0.8)
        ax.axhline(args.good_factor * best, color="#d62728", ls=":", lw=0.7)
        ax.set_xlim(lower[column], upper[column])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    figure.suptitle(f"{cell}: loss vs parameter across the full bounds\n{criterion}; "
                    "red rings = good fits (dotted line), dashed = default", fontsize=10)
    figure.savefig(output_dir / "loss_vs_parameters.png", dpi=150)
    plt.close(figure)

    # 2. UMAP and PCA embeddings of the normalized parameter vectors.
    import umap
    embedding = umap.UMAP(n_neighbors=min(15, len(loss) - 1), min_dist=0.1,
                          random_state=args.seed).fit_transform(normalized)
    pca = PCA(n_components=2).fit(normalized)
    projected = pca.transform(normalized)
    cluster = np.full(len(loss), -2, dtype=int)  # -2 = not a good fit
    cluster[good] = _clusters(embedding[good])
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, coords, title in (
        (axes[0], embedding, "UMAP"),
        (axes[1], projected, f"PCA (PC1 {pca.explained_variance_ratio_[0]:.0%}, "
                             f"PC2 {pca.explained_variance_ratio_[1]:.0%})"),
    ):
        scatter = ax.scatter(coords[order, 0], coords[order, 1], c=loss[order], cmap=cmap,
                             norm=norm, s=14, linewidths=0)
        ax.scatter(coords[good, 0], coords[good, 1], facecolors="none", edgecolors="#d62728",
                   s=40, linewidths=0.8)
        winner = int(np.argmin(loss))
        ax.scatter(*coords[winner], marker="*", s=220, color="#d62728", edgecolors="black",
                   zorder=5)
        found = sorted(set(cluster[good]) - {-1})
        if title == "UMAP" and len(found) > 1:
            for label in found:
                centre = coords[cluster == label].mean(axis=0)
                ax.text(*centre, f"C{label}", fontsize=11, weight="bold", ha="center",
                        bbox={"facecolor": "white", "alpha": 0.7, "lw": 0})
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    figure.colorbar(scatter, ax=axes, label="joint loss (log)", shrink=0.8)
    figure.suptitle(f"{cell}: parameter space coloured by loss\n{criterion}; "
                    "red rings = good fits, star = best, C# = good-fit clusters", fontsize=10)
    figure.savefig(output_dir / "embedding.png", dpi=150)
    plt.close(figure)

    # 3. Good fits across the normalized parameters.
    labels = sorted(set(cluster[good]))
    colors = {label: ("0.6" if label == -1 else plt.get_cmap("tab10")(index % 10))
              for index, label in enumerate(labels)}
    figure, ax = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    x = np.arange(len(keys))
    for row in np.flatnonzero(good)[np.argsort(-loss[good])]:
        ax.plot(x, normalized[row], color=colors[cluster[row]], alpha=0.5, lw=0.9)
    default_norm = (np.asarray([DEFAULTS[key] for key in keys]) - lower) / (upper - lower)
    ax.plot(x, default_norm, color="black", ls="--", lw=1.4, marker="o", ms=3, label="default")
    ax.plot(x, normalized[int(np.argmin(loss))], color="#d62728", lw=2.0, label="best")
    for label in labels:
        ax.plot([], [], color=colors[label], lw=2,
                label=("unclustered" if label == -1 else f"C{label}")
                + f" ({int(np.sum(cluster == label))})")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("position within bounds (0 = lower, 1 = upper)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, axis="x", color="0.9")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    spread = normalized[good].std(axis=0)
    for column, value in enumerate(spread):
        ax.text(column, 1.05, f"{value:.2f}", ha="center", fontsize=6, color="0.35",
                transform=ax.get_xaxis_transform())
    ax.set_title(f"{cell}: good fits across the parameters\n{criterion}; "
                 "numbers on top = SD within the good fits (small = constrained)",
                 fontsize=10, pad=24)
    figure.savefig(output_dir / "good_fits_parameters.png", dpi=150)
    plt.close(figure)

    # 4. Spearman correlations between parameter pairs across the good fits.
    # Drawn even when there are too few good fits, so every folder has the
    # same set of figures.
    enough = good.sum() >= 5
    rho = (np.atleast_2d(spearmanr(normalized[good]).statistic) if enough
           else np.full((len(keys), len(keys)), np.nan))
    figure, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    # Strict lower triangle; its first row and last column are empty.
    lower_triangle = np.tril(np.ones_like(rho, dtype=bool), k=-1)
    masked_rho = np.ma.masked_where(~lower_triangle | ~np.isfinite(rho), rho)[1:, :-1]
    corr_cmap = plt.get_cmap("RdBu_r").copy()
    corr_cmap.set_bad("white")
    image = ax.imshow(masked_rho, cmap=corr_cmap, vmin=-1, vmax=1)
    ax.set_xticks(x[:-1])
    ax.set_yticks(x[:-1])
    ax.set_xticklabels(keys[:-1], rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(keys[1:], fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if not enough:
        ax.text(0.5, 0.5, f"only {good.sum()} good fit(s): need at least 5\nfor correlations",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="0.4")
    figure.colorbar(image, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_title(f"{cell}: parameter correlations across the good fits\n{criterion}\n"
                 "negative = possible compensation (one up, the other down)", fontsize=10)
    figure.savefig(output_dir / "good_fits_correlation.png", dpi=150)
    plt.close(figure)
    if enough:
        pairs = sorted(((rho[i, j], keys[i], keys[j]) for i in range(len(keys))
                        for j in range(i + 1, len(keys)) if np.isfinite(rho[i, j])),
                       key=lambda item: -abs(item[0]))
        print("Strongest correlations within good fits:")
        for value, first, second in pairs[:8]:
            print(f"  {value:+.2f}  {first} ~ {second}")

    # 5. fANOVA parameter importance (Optuna), on log10 loss; the random forest
    # is refit with several seeds and the spread is shown as error bars.
    seeds = range(args.seed, args.seed + 5)
    runs_imp = [param_importances(physical, np.log10(loss), keys, bounds, seed) for seed in seeds]
    joint = np.asarray([[imp[key] for key in keys] for imp in runs_imp])
    mean, spread = joint.mean(axis=0), joint.std(axis=0)
    rank = np.argsort(mean)
    trace_names = sorted(set().union(*(item["trace_losses"] for item in records)))
    complete = [name for name in trace_names
                if all(name in item["trace_losses"] for item in records)]
    per_trace = np.asarray([
        [param_importances(physical,
                           np.log10(np.maximum([item["trace_losses"][name] for item in records],
                                               1e-9)),
                           keys, bounds, args.seed)[key] for key in keys]
        for name in complete
    ]) if complete else np.zeros((0, len(keys)))
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].barh(np.arange(len(keys)), mean[rank], xerr=spread[rank], color="#4c72b0",
                 error_kw={"lw": 0.8})
    axes[0].set_yticks(np.arange(len(keys)))
    axes[0].set_yticklabels([keys[i] for i in rank], fontsize=8)
    axes[0].set_xlabel("importance for log10 joint loss (share of variance)")
    axes[0].set_title("Joint loss (mean +- SD over 5 forest seeds)", fontsize=10)
    axes[0].grid(True, axis="x", color="0.9")
    if len(complete):
        order_desc = rank[::-1]
        image = axes[1].imshow(np.vstack([mean[order_desc], per_trace[:, order_desc]]),
                               cmap="Blues", vmin=0, aspect="auto")
        axes[1].set_yticks(np.arange(len(complete) + 1))
        axes[1].set_yticklabels(["joint loss", *complete], fontsize=8)
        axes[1].axhline(0.5, color="black", lw=0.8)
        axes[1].set_xticks(np.arange(len(keys)))
        axes[1].set_xticklabels([keys[i] for i in order_desc], rotation=60, ha="right",
                                fontsize=8)
        figure.colorbar(image, ax=axes[1], label="importance (each row sums to 1)", shrink=0.8)
        axes[1].set_title("Importance for each trace's loss", fontsize=10)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "no per-trace losses saved for these runs",
                     ha="center", va="center", color="0.4")
    figure.suptitle(f"{cell}: fANOVA parameter importance (Optuna)\n{criterion}; "
                    "importance = share of loss variance a parameter explains on its own, "
                    "over the sampled region only", fontsize=10)
    figure.savefig(output_dir / "param_importance.png", dpi=150)
    plt.close(figure)
    with (output_dir / "param_importance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "joint_mean", "joint_sd", *complete])
        for column, key in enumerate(keys):
            writer.writerow([key, mean[column], spread[column], *per_trace[:, column]])
    print("Most important parameters for the joint loss: "
          + ", ".join(f"{keys[i]} {mean[i]:.2f}" for i in rank[::-1][:6]))

    (output_dir / "analysis_settings.json").write_text(json.dumps({
        "cell": cell, "runs": sorted(set(runs.tolist())), "evaluations": int(len(loss)),
        "best_loss": best, "good_factor": args.good_factor, "good_fits": int(good.sum()),
        "umap_seed": args.seed,
    }, indent=2) + "\n")
    with (output_dir / "landscape.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "loss", "good", "cluster", "umap_1", "umap_2", "pc_1", "pc_2", *keys])
        for row in np.argsort(loss):
            writer.writerow([runs[row], loss[row], bool(good[row]), cluster[row],
                             *embedding[row], *projected[row], *physical[row]])
    print(f"Saved plots and landscape.csv to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
