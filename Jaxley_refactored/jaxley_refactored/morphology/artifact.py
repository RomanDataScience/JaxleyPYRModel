"""Portable, non-pickle exact-HOC artifacts.

The exporter runs where NEURON and compiled MOD files are available. The loader
reconstructs the same Jaxley cell from JSON + NPZ and never imports NEURON,
which makes it suitable for GPU compute nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from jaxley_refactored.compatibility import LegacyCombeBackend
from jaxley_refactored.config.hashing import file_sha256, stable_hash
from jaxley_refactored.config.schema import ModelSpec

from .features import extract_features
from .records import MorphologyResult


ARTIFACT_SCHEMA_VERSION = 1
_STRUCTURAL_COLUMNS = {
    "local_cell_index",
    "local_branch_index",
    "local_comp_index",
    "global_cell_index",
    "global_branch_index",
    "global_comp_index",
    "controlled_by_param",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def export_hoc_artifact(cell, destination: Path, *, provenance=None) -> dict[str, Any]:
    """Serialize a fully built exact-HOC Jaxley cell."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    ncomp = (
        cell.nodes.groupby("global_branch_index", sort=True).size().to_numpy(dtype=int)
    )
    parents = np.asarray(cell.comb_parents, dtype=int)
    xyzr_offsets = np.concatenate(
        ([0], np.cumsum([len(branch_xyzr) for branch_xyzr in cell.xyzr]))
    )
    arrays: dict[str, np.ndarray] = {
        "ncomp": ncomp,
        "parents": parents,
        "xyzr_offsets": xyzr_offsets,
        "xyzr": np.concatenate(cell.xyzr, axis=0),
    }
    columns: dict[str, str] = {}
    for index, column in enumerate(cell.nodes.columns):
        values = cell.nodes[column].to_numpy()
        if column in _STRUCTURAL_COLUMNS or values.dtype.kind not in "biuf":
            continue
        key = f"node_{index:04d}"
        arrays[key] = values
        columns[key] = column

    arrays_path = destination / "arrays.npz"
    with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays)
    temporary.replace(arrays_path)

    channels = [channel._name for channel in cell.channels]
    core = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "columns": columns,
        "channels": channels,
        "diffusion_states": list(cell.diffusion_states),
        "n_branch": int(ncomp.size),
        "n_compartment": int(ncomp.sum()),
        "arrays_sha256": file_sha256(arrays_path),
        "reference_parameters": getattr(cell, "_combe_reference_parameters", {}),
        "parameter_update_mode": getattr(
            cell, "_combe_parameter_update_mode", "exact_hoc_frozen_grid"
        ),
        "provenance": dict(provenance or {}),
    }
    core["fingerprint"] = stable_hash(core)
    _atomic_json(destination / "manifest.json", core)
    return core


@dataclass
class HocArtifactProvider:
    backend: LegacyCombeBackend
    key: str = "hoc_artifact"

    def build(self, spec: ModelSpec) -> MorphologyResult:
        if spec.morphology.path is None:
            raise ValueError("hoc_artifact requires model.morphology.path.")
        root = spec.morphology.path
        manifest_path = root / "manifest.json"
        arrays_path = root / "arrays.npz"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported HOC artifact schema: {manifest.get('schema_version')}"
            )
        actual_hash = file_sha256(arrays_path)
        if actual_hash != manifest.get("arrays_sha256"):
            raise ValueError("HOC artifact arrays checksum does not match manifest.")

        import jaxley as jx

        with np.load(arrays_path, allow_pickle=False) as arrays:
            ncomp = arrays["ncomp"].astype(int)
            parents = arrays["parents"].astype(int).tolist()
            offsets = arrays["xyzr_offsets"].astype(int)
            xyzr_all = arrays["xyzr"]
            xyzr = [
                xyzr_all[offsets[index] : offsets[index + 1]]
                for index in range(len(ncomp))
            ]
            branches = [jx.Branch(ncomp=int(value)) for value in ncomp]
            cell = jx.Cell(branches, parents=parents, xyzr=xyzr)
            cell.initialize()

            saved = {
                column: np.asarray(arrays[key])
                for key, column in manifest["columns"].items()
            }
            for group in spec.morphology.required_groups:
                if group not in saved:
                    raise ValueError(f"HOC artifact is missing group column {group}.")
                indices = np.flatnonzero(saved[group].astype(bool))
                cell.select(indices).add_to_group(group)

            self.backend.insert_channels(cell)
            if spec.mechanisms.calcium_diffusion:
                self.backend.enable_diffusion(
                    cell, spec.mechanisms.calcium_axial_diffusion
                )
            for column, values in saved.items():
                if column in _STRUCTURAL_COLUMNS:
                    continue
                cell.nodes[column] = values

        cell._combe_reference_parameters = {
            **self.backend.reference_parameters,
            **manifest.get("reference_parameters", {}),
        }
        cell._combe_parameter_update_mode = manifest.get(
            "parameter_update_mode", "exact_hoc_frozen_grid"
        )
        features = extract_features(cell, spec.morphology.required_groups)
        return MorphologyResult(
            cell=cell,
            features=features,
            fingerprint=manifest["fingerprint"],
            provenance={
                "provider": self.key,
                "artifact": str(root),
                "arrays_sha256": actual_hash,
                "neuron_required": False,
            },
        )
