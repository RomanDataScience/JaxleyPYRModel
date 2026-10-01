#!/usr/bin/env python3
"""Which parameters have the same effect on the dynamics? Local sensitivity at a fit.

Each free parameter is moved by +-``--step`` of its bound range around a
parameter set (by default a run's best fit) and the change in a vector of
electrophysiological features is measured with central differences, giving a
sensitivity matrix S (features x parameters). Features are divided by a
tolerance (``FEATURE_SCALES``) so that one unit means a meaningful change,
and S is expressed per 10% of each parameter's bound range.

Two parameters whose columns of S point along the same line (|cosine| near 1)
change the cell in the same way, so one can stand in for the other: a positive
cosine means they substitute (raise one, lower the other), a negative cosine
means they cancel (raise both). Parameters are clustered by 1 - |cosine| with
average linkage. The SVD of S gives the stiff parameter combinations the
features pin down and the sloppy ones they barely see.

Outputs, in ``plots/sensitivity_<cell>/``:

- ``parameter_clusters.png``: dendrogram over the parameters, with S below in
  the same order and each parameter's overall effect size.
- ``cosine_similarity.png``: signed cosine between parameter effects.
- ``parameter_relevance.png`` / ``relevance.csv``: each parameter's effect
  when moved alone against what is left after the best one or two partners
  compensate (within ``--max-ratio``), splitting core, compensable (with
  partners) and irrelevant parameters.
- ``stiff_sloppy.png``: singular values and the parameter loadings of each
  direction, stiff to sloppy.
- ``sensitivity.npz`` / ``sensitivity.csv``: features of every simulation and S.

Usage::

    python sensitivity.py runs/cma_m20240527cd
    python sensitivity.py runs/cma_m20240527cd --replot   # reuse sensitivity.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_cma  # noqa: E402
from ca1_parameters import DEFAULTS  # noqa: E402
from fitting.data import load_dataset  # noqa: E402
from fitting.objective import _spikes  # noqa: E402

# Feature -> change that counts as one unit (a meaningful difference).
FEATURE_SCALES = {
    "freq_hz": 2.0,
    "latency_ms": 2.0,
    "first_peak_mV": 2.0,
    "peak_drop_mV": 2.0,
    "adaptation": 0.2,
    "trough_mV": 2.0,
    "rest_mV": 1.0,
    "peak_deflection_mV": 0.5,
    "steady_deflection_mV": 0.5,
    "sag_mV": 0.2,
    "rebound_mV": 0.2,
}
CLUSTER_DISTANCE = 0.3  # cut the dendrogram where |cosine| drops below 0.7
# Effect sizes are |feature change| in tolerance units per 10% of the bound
# range. Below NOISE_FLOOR the direction of an effect is not trusted for
# clustering; RELEVANCE is the default line between relevant and irrelevant.
NOISE_FLOOR = 0.25
RELEVANCE = 1.0
CLASS_COLORS = {"core": "#2ca02c", "compensable": "#ff7f0e", "irrelevant": "0.6"}


def features(depolarizing, hyperpolarizing, simulations) -> dict[str, float]:
    """Feature vector of one simulation (depolarizing traces first, then the pulse)."""
    out: dict[str, float] = {}
    for trace, (time, voltage) in zip(depolarizing, simulations, strict=False):
        start, stop = trace.epoch_start_ms, trace.epoch_stop_ms
        spikes = _spikes(time, voltage, start, stop)
        times = np.asarray([spike[0] for spike in spikes])
        peaks = np.asarray([spike[1] for spike in spikes])
        step = (time >= start) & (time <= stop)
        isis = np.diff(times)
        if len(spikes) >= 2:
            troughs = [voltage[(time > a) & (time < b)].min() for a, b in zip(times[:-1], times[1:])]
        else:
            troughs = [voltage[step & (time >= 0.5 * (start + stop))].mean()]
        name = trace.name
        # Mean frequency from the ISIs, not the spike count: a count moves in
        # whole spikes and would make the differences jump.
        out[f"{name}:freq_hz"] = (1000.0 / isis.mean() if len(isis)
                                  else 1000.0 * len(spikes) / (stop - start))
        out[f"{name}:latency_ms"] = times[0] - start if len(spikes) else stop - start
        out[f"{name}:first_peak_mV"] = peaks[0] if len(spikes) else voltage[step].max()
        out[f"{name}:peak_drop_mV"] = peaks[0] - peaks[-1] if len(spikes) >= 2 else 0.0
        out[f"{name}:adaptation"] = isis[-1] / isis[0] if len(isis) >= 2 else 1.0
        out[f"{name}:trough_mV"] = float(np.mean(troughs))
    time, voltage = simulations[len(depolarizing)]
    start, stop = hyperpolarizing.epoch_start_ms, hyperpolarizing.epoch_stop_ms
    rest = voltage[(time >= start - 100.0) & (time < start)].mean()
    pulse = voltage[(time >= start) & (time <= stop)]
    steady = pulse[int(0.8 * pulse.size):].mean() - rest
    peak = pulse.min() - rest
    out["hyper:rest_mV"] = rest
    out["hyper:peak_deflection_mV"] = peak
    out["hyper:steady_deflection_mV"] = steady
    out["hyper:sag_mV"] = steady - peak
    out["hyper:rebound_mV"] = voltage[(time > stop) & (time <= stop + 150.0)].max() - rest
    return {key: float(value) for key, value in out.items()}


def _simulate_features(values: dict[str, float]) -> dict[str, float]:
    evaluator = run_cma._WORKER
    simulations = evaluator.simulate(values)
    return features(evaluator.traces[:-1], evaluator.traces[-1], simulations)


def _scale(name: str) -> float:
    return FEATURE_SCALES[name.split(":", 1)[1]]


def run_simulations(run_dir: Path, base: dict[str, float], keys: list[str],
                    bounds: dict[str, tuple[float, float]], step: float, workers: int):
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    _, fixed = run_cma._resolve_parameters(dict(config))
    depolarizing, hyperpolarizing = load_dataset(
        (HERE / "configs" / config["data_root"]).resolve(), config["cell"],
        list(config["depolarizing_traces"]), config["hyperpolarizing_trace"])
    evaluator = run_cma.Evaluator(
        [*depolarizing, hyperpolarizing], {**DEFAULTS, **fixed}, config.get("simulation", {}) or {},
        float(config.get("hyper_weight", 1.0)), dict(config.get("objective", {}) or {}))
    run_cma.build_mechanisms()

    # Positions in [0, 1] within the bounds; clipped at the bounds, so a
    # parameter sitting on a bound gets a one-sided difference.
    lower = np.asarray([bounds[key][0] for key in keys])
    upper = np.asarray([bounds[key][1] for key in keys])
    x0 = (np.asarray([base[key] for key in keys]) - lower) / (upper - lower)
    plus = np.clip(x0 + step, 0.0, 1.0)
    minus = np.clip(x0 - step, 0.0, 1.0)
    candidates = [dict(base)]
    for column, key in enumerate(keys):
        for position in (plus[column], minus[column]):
            candidates.append({**base, key: float(lower[column] + position * (upper[column] - lower[column]))})
    print(f"Simulating {len(candidates)} parameter sets with {workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=run_cma._initialize_worker, initargs=(evaluator,)) as pool:
        results = list(pool.map(_simulate_features, candidates))
    names = list(results[0])
    values = np.asarray([[result[name] for name in names] for result in results])
    return names, values, x0, plus, minus, config["cell"]


def plot(output_dir: Path, cell: str, keys: list[str], names: list[str], S: np.ndarray,
         source: str, step: float) -> None:
    effect = np.linalg.norm(S, axis=0)
    sensitive = effect >= NOISE_FLOOR
    sens_keys = [key for key, keep in zip(keys, sensitive) if keep]
    unit = S[:, sensitive] / effect[sensitive]
    cosine = np.clip(unit.T @ unit, -1.0, 1.0)
    distance = 1.0 - np.abs(cosine)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    groups = fcluster(tree, t=CLUSTER_DISTANCE, criterion="distance")

    # Figure 1, parameters as rows: dendrogram | names + effect size | S.
    # Rows go bottom to top: parameters below the noise floor, then the
    # dendrogram leaves. Leaves sit at 5, 15, 25, ... in dendrogram units.
    tree_plot = dendrogram(tree, labels=sens_keys, no_plot=True,
                           color_threshold=CLUSTER_DISTANCE, above_threshold_color="0.6")
    weak = [index for index, keep in enumerate(sensitive) if not keep]
    order = weak + [keys.index(label) for label in tree_plot["ivl"]]
    ordered_keys = [keys[index] for index in order]
    leaf_colors = dict(zip(tree_plot["ivl"], tree_plot["leaves_color_list"]))
    offset = 10.0 * len(weak)
    centres = 5.0 + 10.0 * np.arange(len(order))
    height = 10.0 * len(order)

    figure = plt.figure(figsize=(18, 3.0 + 0.42 * len(order)), constrained_layout=True)
    grid = figure.add_gridspec(1, 4, width_ratios=[4, 2.2, 14, 0.4])
    ax_tree = figure.add_subplot(grid[0, 0])
    for xs, ys, color in zip(tree_plot["icoord"], tree_plot["dcoord"], tree_plot["color_list"]):
        ax_tree.plot(ys, np.asarray(xs) + offset, color=color, lw=1.3)
    ax_tree.axvline(CLUSTER_DISTANCE, color="#d62728", ls=":", lw=0.8)
    ax_tree.set_ylim(0, height)
    ax_tree.set_xlim(max(1.0, max(max(d) for d in tree_plot["dcoord"]) * 1.05), 0)
    ax_tree.set_xlabel("1 - |cosine|")
    ax_tree.set_yticks([])
    if weak:
        ax_tree.axhline(offset, color="black", lw=0.8)
        ax_tree.text(0.5 * ax_tree.get_xlim()[0], offset / 2, "effect below\nnoise floor:\nnot clustered",
                     ha="center", va="center", fontsize=8, color="0.45")
    for spine in ("top", "right", "left"):
        ax_tree.spines[spine].set_visible(False)

    ax_effect = figure.add_subplot(grid[0, 1], sharey=ax_tree)
    ax_effect.barh(centres, effect[order], height=8,
                   color=["#4c72b0" if sensitive[index] else "0.75" for index in order])
    ax_effect.set_xscale("log")
    ax_effect.axvline(RELEVANCE, color="black", lw=0.8, ls=":")
    ax_effect.set_xlabel("effect size |S column|\n(dotted = relevance line)", fontsize=8)
    ax_effect.set_yticks(centres)
    ax_effect.set_yticklabels(ordered_keys, fontsize=9)
    ax_effect.tick_params(axis="y", length=0, labelleft=True)
    ax_tree.tick_params(axis="y", left=False, labelleft=False)  # shared y: names only once
    for label, key in zip(ax_effect.get_yticklabels(), ordered_keys):
        color = leaf_colors.get(key, "0.5")
        label.set_color("0.25" if color == "0.6" else color)
        if key in leaf_colors and color != "0.6":
            label.set_fontweight("bold")
    if weak:
        ax_effect.axhline(offset, color="black", lw=0.8)

    ax_heat = figure.add_subplot(grid[0, 2], sharey=ax_tree)
    limit = float(np.percentile(np.abs(S), 98)) or 1.0
    image = ax_heat.imshow(S[:, order].T, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto",
                           origin="lower", extent=(0, len(names), 0, height),
                           interpolation="nearest")
    ax_heat.set_xticks(np.arange(len(names)) + 0.5)
    ax_heat.set_xticklabels(names, rotation=90, fontsize=7)
    ax_heat.tick_params(axis="y", labelleft=False, length=0)
    previous = None
    for column, name in enumerate(names):
        prefix = name.split(":")[0]
        if previous is not None and prefix != previous:
            ax_heat.axvline(column, color="black", lw=0.5)
        previous = prefix
    if weak:
        ax_heat.axhline(offset, color="black", lw=0.8)
    # Box each cluster: its rows across the tree, the names and the heatmap.
    # scipy colours the leaves of one cluster alike and keeps them adjacent.
    leaves = tree_plot["ivl"]
    start = 0
    for stop in range(1, len(leaves) + 1):
        if stop < len(leaves) and leaf_colors[leaves[stop]] == leaf_colors[leaves[start]]:
            continue
        color = leaf_colors[leaves[start]]
        if color != "0.6" and stop - start > 1:
            low, high = offset + 10.0 * start + 0.8, offset + 10.0 * stop - 0.8
            merge = max(max(d) for d, c in zip(tree_plot["dcoord"], tree_plot["color_list"])
                        if c == color)
            ax_tree.add_patch(matplotlib.patches.Rectangle(
                (0, low), merge * 1.15 + 0.02, high - low, facecolor=color, alpha=0.08,
                edgecolor=color, lw=1.4, clip_on=False))
            ax_effect.axhspan(low, high, facecolor=color, alpha=0.10, edgecolor=color, lw=1.4)
            ax_heat.add_patch(matplotlib.patches.Rectangle(
                (0, low), len(names), high - low, facecolor="none", edgecolor=color, lw=1.6))
        start = stop

    figure.colorbar(image, cax=figure.add_subplot(grid[0, 3]),
                    label="feature change (tolerance units) per +10% of the bound range")
    figure.suptitle(f"{cell}: parameters clustered by the direction of their effect on the features; "
                    f"coloured names = clusters at |cosine| >= {1 - CLUSTER_DISTANCE:.1f} (dotted)\n"
                    f"local sensitivity at {source}; +-{step:g} of each bound range", fontsize=10)
    figure.savefig(output_dir / "parameter_clusters.png", dpi=150)
    plt.close(figure)

    # Figure 2: signed cosine, strict lower triangle, dendrogram order.
    sens_order = [sens_keys.index(key) for key in tree_plot["ivl"]]
    ordered_cos = cosine[np.ix_(sens_order, sens_order)]
    lower = np.tril(np.ones_like(ordered_cos, dtype=bool), k=-1)
    masked = np.ma.masked_where(~lower, ordered_cos)[1:, :-1]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    figure, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    image = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    labels = tree_plot["ivl"]
    ax.set_xticks(np.arange(len(labels) - 1))
    ax.set_yticks(np.arange(len(labels) - 1))
    ax.set_xticklabels(labels[:-1], rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels[1:], fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    figure.colorbar(image, ax=ax, label="cosine between effects on the features", shrink=0.8)
    ax.set_title(f"{cell}: similarity of parameter effects (dendrogram order)\n"
                 "+1 = same effect (raise one, lower the other to compensate); "
                 "-1 = opposite (raise both)", fontsize=10)
    figure.savefig(output_dir / "cosine_similarity.png", dpi=150)
    plt.close(figure)

    # Figure 3: stiff and sloppy directions.
    _, singular, vt = np.linalg.svd(S, full_matrices=False)
    figure, (ax_s, ax_v) = plt.subplots(1, 2, figsize=(17, 7), constrained_layout=True,
                                        gridspec_kw={"width_ratios": [1, 3]})
    ax_s.semilogy(np.arange(1, len(singular) + 1), singular, "o-", color="#4c72b0")
    ax_s.set_xlabel("direction (stiff -> sloppy)")
    ax_s.set_ylabel("singular value (feature change per unit move)")
    ax_s.set_title("Spectrum: decades between stiff and sloppy", fontsize=10)
    ax_s.grid(True, color="0.9")
    loadings = vt[:, order]
    image = ax_v.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_v.set_yticks(np.arange(len(singular)))
    ax_v.set_yticklabels([f"{i + 1} ({value:.2g})" for i, value in enumerate(singular)], fontsize=7)
    ax_v.set_xticks(np.arange(len(order)))
    ax_v.set_xticklabels(ordered_keys, rotation=60, ha="right", fontsize=8)
    ax_v.set_ylabel("direction (singular value)")
    ax_v.set_title("Parameter loadings: top rows = combinations the features pin down; "
                   "bottom rows = combinations that barely change them (degenerate)", fontsize=10)
    figure.colorbar(image, ax=ax_v, label="loading", shrink=0.8)
    figure.suptitle(f"{cell}: stiff and sloppy parameter combinations at {source}")
    figure.savefig(output_dir / "stiff_sloppy.png", dpi=150)
    plt.close(figure)

    print("Parameter clusters (|cosine| >= %.1f):" % (1 - CLUSTER_DISTANCE))
    for group in sorted(set(groups)):
        members = [key for key, label in zip(sens_keys, groups) if label == group]
        if len(members) > 1:
            print("  " + ", ".join(members))
    weak = [key for key, keep in zip(keys, sensitive) if not keep]
    if weak:
        print("Little measurable effect: " + ", ".join(weak))


def _compensate(target: np.ndarray, columns: np.ndarray, max_ratio: float):
    """Best bounded linear compensation: residual norm and coefficients."""
    from scipy.optimize import lsq_linear
    fit = lsq_linear(columns, target, bounds=(-max_ratio, max_ratio))
    return float(np.linalg.norm(target - columns @ fit.x)), fit.x


def relevance(output_dir: Path, cell: str, keys: list[str], S: np.ndarray, threshold: float,
              source: str, max_ratio: float = 3.0) -> None:
    """Classify parameters as core, compensable or irrelevant under these protocols.

    individual: feature change when only this parameter moves (|column of S|).
    after 1 / after 2: what is left after the best compensation by one partner
    or by the best pair of partners. A partner may move at most ``max_ratio``
    times as far (as a fraction of its own range) as the parameter it
    compensates, so compensation has to stay plausible within the bounds.

    irrelevant: individual < threshold. compensable: individual >= threshold
    but at most two partners bring it below threshold. core: no pair can.
    """
    from itertools import combinations
    effect = np.linalg.norm(S, axis=0)
    after_one = np.zeros(len(keys))
    residual = np.zeros(len(keys))
    partners: list[str] = [""] * len(keys)
    for column in range(len(keys)):
        target = S[:, column]
        others = [index for index in range(len(keys)) if index != column]
        singles = {k: _compensate(target, S[:, [k]], max_ratio) for k in others}
        best_single = min(singles, key=lambda k: singles[k][0])
        after_one[column] = singles[best_single][0]
        best_pair, best = None, (np.inf, None)
        for pair in combinations(others, 2):
            result = _compensate(target, S[:, list(pair)], max_ratio)
            if result[0] < best[0]:
                best_pair, best = pair, result
        residual[column] = best[0]
        # Raising this parameter is undone by moving partner k by -coef_k.
        if after_one[column] < threshold:
            chosen = [(best_single, singles[best_single][1][0])]
        else:
            chosen = list(zip(best_pair, best[1]))
        partners[column] = ", ".join(f"{'down' if coef > 0 else 'up'} {keys[k]} x{abs(coef):.1f}"
                                     for k, coef in chosen if abs(coef) > 0.05)
    classes = np.where(effect < threshold, "irrelevant",
                       np.where(residual >= threshold, "core", "compensable"))

    importance_path = HERE / "plots" / f"landscape_{cell}" / "param_importance.csv"
    importance = None
    if importance_path.exists():
        with importance_path.open() as handle:
            rows = {row["parameter"]: float(row["joint_mean"]) for row in csv.DictReader(handle)}
        if all(key in rows for key in keys):
            importance = np.asarray([rows[key] for key in keys])

    order = np.lexsort((-effect, [list(CLASS_COLORS).index(c) for c in classes]))
    figure, (ax_scatter, ax_bars) = plt.subplots(
        1, 2, figsize=(19, 8.5), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})

    floor = 0.5 * min(effect[effect > 0].min(), residual[residual > 0].min(), threshold)
    x = np.maximum(effect, floor)
    y = np.maximum(residual, floor)
    top = 2.0 * max(x.max(), y.max(), threshold)
    ax_scatter.fill_between([floor, threshold], floor, top, color="0.93", lw=0)
    ax_scatter.fill_between([threshold, top], floor, threshold, color="#ffe9d1", lw=0)
    ax_scatter.fill_between([threshold, top], threshold, top, color="#dff2df", lw=0)
    ax_scatter.plot([floor, top], [floor, top], color="0.5", ls="--", lw=0.8)
    sizes = 40 + 900 * (importance if importance is not None else np.full(len(keys), 0.05))
    ax_scatter.scatter(x, y, s=sizes, c=[CLASS_COLORS[c] for c in classes],
                       edgecolors="black", linewidths=0.6, zorder=3)
    for index, key in enumerate(keys):
        ax_scatter.annotate(key, (x[index], y[index]), xytext=(5, 4), textcoords="offset points",
                            fontsize=7)
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlim(floor, top)
    ax_scatter.set_ylim(floor, top)
    ax_scatter.set_xlabel("individual effect: moved alone (does the model care?)")
    ax_scatter.set_ylabel("effect left after the best pair of partners compensates\n"
                          "(can the data pin it down?)")
    ax_scatter.text(1.15 * floor, 0.6 * top, "irrelevant", fontsize=11, color="0.4", va="top")
    ax_scatter.text(1.15 * threshold, 0.6 * top, "core", fontsize=11, color="#2ca02c", va="top")
    ax_scatter.text(1.15 * threshold, 0.8 * threshold, "compensable (degenerate)", fontsize=11,
                    color="#d9822b", va="top")
    ax_scatter.set_title("feature change in tolerance units per 10% of the bound range; dashed: no "
                         f"pair can compensate at all; partners move <= {max_ratio:g}x as far\n" + ("marker size = fANOVA importance from the fits"
                                                    if importance is not None
                                                    else "no fANOVA importances found"),
                         fontsize=9)

    rows = np.arange(len(keys))
    ax_bars.barh(rows, effect[order], color=[CLASS_COLORS[c] for c in classes[order]], alpha=0.35,
                 label="individual")
    ax_bars.barh(rows, after_one[order], height=0.7, color=[CLASS_COLORS[c] for c in classes[order]],
                 alpha=0.6, label="after best single partner")
    ax_bars.barh(rows, residual[order], height=0.4, color=[CLASS_COLORS[c] for c in classes[order]],
                 label="after best pair of partners")
    ax_bars.axvline(threshold, color="black", lw=0.8, ls=":")
    ax_bars.set_xscale("log")
    ax_bars.set_xlim(floor, top * 30)  # room for the partner text
    ax_bars.set_yticks(rows)
    ax_bars.set_yticklabels([keys[i] for i in order], fontsize=8)
    for label, index in zip(ax_bars.get_yticklabels(), order):
        label.set_color("#d9822b" if classes[index] == "compensable" else
                        "#2ca02c" if classes[index] == "core" else "0.45")
    ax_bars.invert_yaxis()
    for row, index in enumerate(order):
        if classes[index] == "compensable":
            ax_bars.text(1.3 * effect[index], row, f"<- {partners[index]}", va="center",
                         fontsize=7, color="#8c4a00")
    ax_bars.set_xlabel("effect (tolerance units per 10% of the bound range); dotted = relevance line")
    ax_bars.set_title("light = moved alone, mid = after best partner, dark = after best pair;\n"
                      "compensable rows: partners that undo a rise, with how far they move "
                      "relative to it", fontsize=9)
    ax_bars.legend(loc="lower right", fontsize=8)
    figure.suptitle(f"{cell}: core vs compensable vs irrelevant parameters under these protocols "
                    f"(local, at {source})", fontsize=11)
    figure.savefig(output_dir / "parameter_relevance.png", dpi=150)
    plt.close(figure)

    with (output_dir / "relevance.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "class", "individual_effect", "after_best_partner",
                         "after_best_pair", "compensating_partners", "fanova_importance"])
        for index in order:
            writer.writerow([keys[index], classes[index], effect[index], after_one[index],
                             residual[index],
                             partners[index] if classes[index] == "compensable" else "",
                             "" if importance is None else importance[index]])
    for name in CLASS_COLORS:
        members = [keys[i] for i in order if classes[i] == name]
        print(f"{name} ({len(members)}): " + ", ".join(members))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory; its best fit is the base point")
    parser.add_argument("--parameters", type=Path, default=None,
                        help="JSON with a 'parameters' mapping to use instead of the run's best fit")
    parser.add_argument("--step", type=float, default=0.05,
                        help="half step as a fraction of each bound range (default 0.05)")
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--threshold", type=float, default=RELEVANCE,
                        help="relevance line in tolerance units per 10%% of the bound range")
    parser.add_argument("--max-ratio", type=float, default=3.0,
                        help="largest partner move relative to the compensated parameter's move")
    parser.add_argument("--replot", action="store_true", help="reuse sensitivity.npz")
    args = parser.parse_args()

    run_dir = args.run.resolve()
    parameter_file = (args.parameters or run_dir / "best_parameters.json").resolve()
    catalog = json.loads((run_dir / "parameter_catalog.json").read_text())
    keys = list(catalog["parameters"])
    bounds = {key: (spec["lower"], spec["upper"]) for key, spec in catalog["parameters"].items()}
    cell = yaml.safe_load((run_dir / "config.yaml").read_text())["cell"]
    output_dir = HERE / "plots" / f"sensitivity_{cell}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = (str(parameter_file.relative_to(HERE)) if parameter_file.is_relative_to(HERE)
              else str(parameter_file))

    if args.replot:
        data = np.load(output_dir / "sensitivity.npz", allow_pickle=False)
        names, values = list(data["names"]), data["values"]
        plus, minus, step = data["plus"], data["minus"], float(data["step"])
    else:
        base = {**DEFAULTS, **json.loads(parameter_file.read_text())["parameters"]}
        names, values, _, plus, minus, cell = run_simulations(
            run_dir, base, keys, bounds, args.step, args.workers)
        step = args.step
    scales = np.asarray([_scale(name) for name in names])
    deltas = values[1::2] - values[2::2]               # (+) minus (-) per parameter
    S = (deltas / scales).T / ((plus - minus) / 0.1)   # per +10% of the bound range
    np.savez(output_dir / "sensitivity.npz", names=np.asarray(names), keys=np.asarray(keys),
             values=values, plus=plus, minus=minus, step=step, S=S)
    with (output_dir / "sensitivity.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "tolerance", "value_at_base", *keys])
        for row, name in enumerate(names):
            writer.writerow([name, _scale(name), values[0, row], *S[row]])
    plot(output_dir, cell, keys, names, S, source, step)
    relevance(output_dir, cell, keys, S, args.threshold, source, args.max_ratio)
    print(f"Saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
