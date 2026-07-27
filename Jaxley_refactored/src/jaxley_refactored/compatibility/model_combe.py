"""Migration adapter around ``JaxleyModel/model/model_Combe.py``.

All dependencies on the monolithic implementation are confined here. The rest
of the refactored package depends on the small backend methods below and can be
migrated to native builders/rules one component at a time.
"""

from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from typing import Any, Sequence


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def legacy_module():
    root = _repository_root()
    if str(root) not in sys.path:
        # Transitional boundary only: the historical project is not packaged.
        sys.path.insert(0, str(root))
    return importlib.import_module("JaxleyModel.model.model_Combe")


class LegacyCombeBackend:
    """Concrete backend used until every Combe recipe is native to this package."""

    def build_cell(
        self,
        *,
        morphology_source: str,
        d_lambda: float,
        calcium_diffusion: bool,
    ):
        return legacy_module().Combe2023(
            morphology_source=morphology_source,
            d_lambda=d_lambda,
            enable_calcium_diffusion=calcium_diffusion,
        )

    def build_swc_cell(
        self,
        path: Path,
        *,
        d_lambda: float,
        frequency_hz: float,
        calcium_diffusion: bool,
        calcium_axial_diffusion: float,
    ):
        """Build the Combe recipe around any compatible grouped SWC file.

        This adapter deliberately owns the migration-era calls into the legacy
        recipe. The morphology provider therefore depends on a stable, small
        interface instead of importing functions from ``model_Combe``.
        """
        from dataclasses import asdict

        legacy = legacy_module()
        cell = legacy.jx.read_swc(str(path), ncomp=1, assign_groups=True)
        legacy.set_distances_from_soma(cell)
        legacy.set_passive_properties(cell, legacy.COMBE_PARAMS)
        legacy.update_number_compartments(
            cell, d_lambda=d_lambda, frequency=frequency_hz
        )
        cell.initialize()
        legacy.set_distances_from_soma(cell)
        legacy.insert_combe_channels(cell)
        legacy.set_passive_properties(cell, legacy.COMBE_PARAMS)
        legacy.set_combe_channels(cell, legacy.COMBE_PARAMS)
        if calcium_diffusion:
            legacy.enable_cal4_diffusion(
                cell, axial_diffusion=calcium_axial_diffusion
            )
        cell.set("v", legacy.COMBE_PARAMS.Epas)
        cell._combe_reference_parameters = asdict(legacy.COMBE_PARAMS)
        cell._combe_parameter_update_mode = legacy.RULE_UPDATE_MODE
        return cell

    def parameter_state(
        self,
        cell: Any,
        keys: Sequence[str],
        values: Any,
        state: Any = None,
    ):
        return legacy_module().set_fitted_parameters(cell, keys, values, state)

    def insert_channels(self, cell):
        return legacy_module().insert_combe_channels(cell)

    def enable_diffusion(self, cell, value: float = 0.22):
        return legacy_module().enable_cal4_diffusion(cell, axial_diffusion=value)

    @property
    def reference_parameters(self) -> dict[str, float]:
        from dataclasses import asdict

        return asdict(legacy_module().COMBE_PARAMS)

    @property
    def default_swc_path(self) -> Path:
        """Return the historical morphology as an explicit provider default."""
        return legacy_module().morphology_path()
