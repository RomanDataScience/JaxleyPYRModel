#!/usr/bin/env python3
"""Plot the CA1 model's channel densities across the morphology.

Mirrors ``OptimizationNEURON_Vitale/scripts/plot_mechanisms.py``: one
morphology panel per mechanism, coloured per segment by its density divided by
that mechanism's maximum, and a section-by-mechanism heatmap below.

Values are read from the built cell after ``apply_parameters``, so they include
the distance-dependent profiles and the F-factor spine correction exactly as
simulated. Calcium channels are GHK permeabilities (cm/s), not conductances.
The single spine (``shead``/``sneck``) is outside ``h.all``, carries no 3D
points and is left out.

Usage::

    python plot_mechanisms.py
    python plot_mechanisms.py --parameters runs/cma_m20260331b/best_parameters.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ca1_model  # noqa: E402
from ca1_parameters import DEFAULTS  # noqa: E402

# HOC section list -> plot group; dendrite lists are disjoint in the loader.
GROUP_LISTS = (
    ("soma", "soma"),
    ("axon", "all_axon"),
    ("basal", "basal_dendrites"),
    ("trunk", "trunk"),
    ("oblique", "oblique_dendrites"),
    ("tuft", "tuft"),
)
GROUP_ORDER = tuple(group for group, _ in GROUP_LISTS)
GROUP_COLORS = {
    "soma": "#d62728",
    "axon": "#9467bd",
    "basal": "#2ca02c",
    "trunk": "#8c564b",
    "oblique": "#ff7f0e",
    "tuft": "#1f77b4",
}

# mechanism -> (density field, unit). Mechanisms not listed (calcium pools,
# vmax) have no density field and are drawn in gray.
DENSITY_FIELDS = {
    "Leak_pyr": ("gmax", "S/cm²"),
    "Na_BG_soma": ("gmax", "S/cm²"),
    "Na_BG_dend": ("gmax", "S/cm²"),
    "Na_BG_axon": ("gmax", "S/cm²"),
    "K_DRS4_params_voltage_dep": ("gmax", "S/cm²"),
    "kd_params3": ("gkd", "S/cm²"),
    "km_q10_2": ("gbar", "S/cm²"),
    "K_A_prox": ("gmax", "S/cm²"),
    "K_A_dist": ("gmax", "S/cm²"),
    "H_CA1pyr_prox": ("gmax", "S/cm²"),
    "H_CA1pyr_dist": ("gmax", "S/cm²"),
    "bk_ch_pool": ("gbar", "S/cm²"),
    "K_AHP3_lpool": ("gmax", "S/cm²"),
    "sKCa": ("gk", "S/cm²"),
    "CaN_BG_pool2_ghk": ("pbar", "cm/s"),
    "CaL_pool2_ghk": ("pbar", "cm/s"),
    "cat3": ("pbar", "cm/s"),
    "car": ("pbar", "cm/s"),
}
SKIP_MECHANISMS = {"vmax"}


def _xy(sec) -> tuple[np.ndarray, np.ndarray]:
    """3D points rotated 90 degrees so the apical tuft (+x in the HOC) points up."""
    n3d = int(sec.n3d())
    x = np.asarray([sec.x3d(i) for i in range(n3d)], dtype=float)
    y = np.asarray([sec.y3d(i) for i in range(n3d)], dtype=float)
    return -y, x


def _segment_paths(sec) -> list[np.ndarray]:
    nseg = int(sec.nseg)
    n3d = int(sec.n3d())
    arc = np.asarray([sec.arc3d(i) for i in range(n3d)], dtype=float)
    x3d, y3d = _xy(sec)
    position = np.linspace(0.0, float(sec.L), nseg + 1)
    x = np.interp(position, arc, x3d)
    y = np.interp(position, arc, y3d)
    return [np.asarray([[x[i], y[i]], [x[i + 1], y[i + 1]]]) for i in range(nseg)]


def _sections(h):
    groups = {}
    for group, list_name in GROUP_LISTS:
        members = getattr(h, list_name)
        # ``h.soma`` is a single Section; iterating it would yield segments.
        for sec in ([members] if isinstance(members, type(h.soma)) else members):
            groups.setdefault(sec, group)
    sections = [sec for sec in h.all if sec in groups and int(sec.n3d()) >= 2]
    sections.sort(key=lambda sec: (GROUP_ORDER.index(groups[sec]), sec.name()))
    mechanisms = {
        sec: set(sec.psection()["density_mechs"]) - SKIP_MECHANISMS
        for sec in sections
    }
    return sections, groups, mechanisms


def _values(sec, mechanism: str) -> np.ndarray | None:
    if mechanism not in DENSITY_FIELDS:
        return None
    field = DENSITY_FIELDS[mechanism][0]
    values = np.asarray(sec.psection()["density_mechs"][mechanism][field],
                        dtype=float).ravel()
    if values.size == 1:
        values = np.repeat(values, int(sec.nseg))
    return values


def _draw(ax, sections, section_mechanisms, mechanism, normalized, norm, cmap):
    for sec in sections:
        x, y = _xy(sec)
        ax.plot(x, y, color="#d9d9d9", alpha=0.45, linewidth=0.55,
                solid_capstyle="round")
    for sec in sections:
        if mechanism not in section_mechanisms[sec]:
            continue
        values = normalized[sec]
        paths = _segment_paths(sec)
        colors = (["#777777"] * len(paths) if values is None else
                  ["#777777" if value <= 0.0 else cmap(norm(value)) for value in values])
        ax.add_collection(LineCollection(paths, colors=colors, linewidths=1.35,
                                         alpha=0.95))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_plot(output: Path, parameters: Path | None = None) -> dict[str, dict[str, int]]:
    values = dict(DEFAULTS)
    source = "ca1_parameters defaults"
    if parameters is not None:
        payload = json.loads(parameters.read_text(encoding="utf-8"))
        values.update({key: float(value) for key, value in
                       payload.get("parameters", payload).items()})
        source = (str(parameters.relative_to(HERE)) if parameters.is_relative_to(HERE)
                  else str(parameters))
    h = ca1_model.build_cell()
    ca1_model.apply_parameters(h, values)

    sections, groups, section_mechanisms = _sections(h)
    mechanisms = [mech for mech in DENSITY_FIELDS
                  if any(mech in section_mechanisms[sec] for sec in sections)]
    mechanisms += sorted(set().union(*section_mechanisms.values()) - set(mechanisms))

    raw = {mech: {sec: (_values(sec, mech) if mech in section_mechanisms[sec] else None)
                  for sec in sections}
           for mech in mechanisms}
    maxima = {
        mech: max((float(v) for vals in raw[mech].values() if vals is not None
                   for v in vals if v > 0.0 and np.isfinite(v)), default=0.0)
        for mech in mechanisms
    }
    normalized = {
        mech: {sec: (None if vals is None else
                     vals / maxima[mech] if maxima[mech] > 0.0 else np.zeros_like(vals))
               for sec, vals in raw[mech].items()}
        for mech in mechanisms
    }

    norm = LogNorm(vmin=1.0e-3, vmax=1.0)
    cmap = plt.get_cmap("viridis")
    ncols = 6
    nrows = int(np.ceil(len(mechanisms) / ncols))
    figure = plt.figure(figsize=(17, 3.6 * nrows + 4.4), constrained_layout=True)
    grid = figure.add_gridspec(nrows + 1, ncols, height_ratios=[1.0] * nrows + [0.85])

    for index, mech in enumerate(mechanisms):
        ax = figure.add_subplot(grid[index // ncols, index % ncols])
        _draw(ax, sections, section_mechanisms, mech, normalized[mech], norm, cmap)
        present = sorted({groups[sec] for sec in sections
                          if mech in section_mechanisms[sec]}, key=GROUP_ORDER.index)
        count = sum(mech in section_mechanisms[sec] for sec in sections)
        if mech in DENSITY_FIELDS:
            field, unit = DENSITY_FIELDS[mech]
            title = f"{mech}  ({count})\n{field} max={maxima[mech]:.3g} {unit}"
        else:
            title = f"{mech}  ({count})\nno density field"
        ax.set_title(title, fontsize=8)
        ax.text(0.02, 0.0, ", ".join(present), transform=ax.transAxes,
                fontsize=7, color="#444444", va="bottom")
    for index in range(len(mechanisms), nrows * ncols):
        figure.add_subplot(grid[index // ncols, index % ncols]).axis("off")

    heatmap_ax = figure.add_subplot(grid[nrows, :])
    matrix = np.full((len(mechanisms), len(sections)), np.nan)
    for row, mech in enumerate(mechanisms):
        for column, sec in enumerate(sections):
            vals = normalized[mech][sec]
            if vals is not None:
                matrix[row, column] = float(np.mean(vals))
    heatmap_ax.imshow(np.ma.masked_invalid(matrix), aspect="auto",
                      interpolation="nearest", cmap=cmap, norm=norm)
    heatmap_ax.set_yticks(np.arange(len(mechanisms)))
    heatmap_ax.set_yticklabels(mechanisms, fontsize=7)
    heatmap_ax.set_xticks([])
    heatmap_ax.set_xlabel("Sections ordered by soma, axon, basal, trunk, oblique and tuft",
                          labelpad=16)
    heatmap_ax.set_title("Mean density by section", fontsize=10, pad=8)
    offset = 0
    for group in GROUP_ORDER:
        count = sum(groups[sec] == group for sec in sections)
        heatmap_ax.axvline(offset - 0.5, color="white", linewidth=1.2)
        # Stagger the tiny soma/axon labels so they do not overlap.
        y = len(mechanisms) + (0.3 if count > 5 else
                               0.3 + 1.1 * GROUP_ORDER.index(group))
        heatmap_ax.text(offset + max(count - 1, 0) / 2.0, y, f"{group} ({count})",
                        ha="center", va="top", fontsize=8, color=GROUP_COLORS[group],
                        clip_on=False)
        offset += count
    heatmap_ax.set_ylim(len(mechanisms) - 0.5, -0.5)

    scalar_map = ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=figure.axes, location="right",
                               shrink=0.75, pad=0.02)
    colorbar.set_label("Density / mechanism maximum (log scale)")
    figure.suptitle(
        "CA1 pyramidal cell model mechanism distributions (after F-factor spine correction)\n"
        "normalized to each mechanism's maximum; gray = zero/no density field; "
        f"values: {source}",
        fontsize=12, y=1.03,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    inventory: dict[str, dict[str, int]] = {}
    for group in GROUP_ORDER:
        group_sections = [sec for sec in sections if groups[sec] == group]
        inventory[group] = {
            mech: count for mech in mechanisms
            if (count := sum(mech in section_mechanisms[sec] for sec in group_sections))
        }
    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parameters", type=Path, default=None,
                        help="JSON with a 'parameters' mapping (e.g. best_parameters.json); "
                             "otherwise the catalog defaults.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (HERE / "plots" / (
        "ca1_mechanism_distribution.png" if args.parameters is None else
        f"ca1_mechanism_distribution_{args.parameters.resolve().parent.name}.png"))
    inventory = make_plot(output.resolve(),
                          args.parameters.resolve() if args.parameters else None)
    print(f"Saved {output.resolve()}")
    for group, entries in inventory.items():
        print(f"{group}: " + ", ".join(f"{name} ({count})" for name, count in entries.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
