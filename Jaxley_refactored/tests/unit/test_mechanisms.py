from types import SimpleNamespace

from jaxley_refactored.mechanisms.registry import MechanismInfo, MechanismRegistry


class _Cell:
    def __init__(self):
        channel = SimpleNamespace(_name="channel", current_name="i_channel")
        removed = SimpleNamespace(_name="removed", current_name="i_removed")
        pump = SimpleNamespace(_name="pump", current_name="i_pump")
        self.base = SimpleNamespace(
            channels=[channel, removed],
            pumps=[pump],
            membrane_current_names=["i_channel", "i_removed", "i_pump"],
        )

    @property
    def channels(self):
        return self.base.channels

    @property
    def pumps(self):
        return self.base.pumps

    def delete(self, mechanism):
        self.base.channels.remove(mechanism)
        # Reproduce Jaxley 0.13's channel-only recomputation.
        self.base.membrane_current_names = [
            item.current_name for item in self.base.channels
        ]


def test_filtering_restores_retained_pump_current_names():
    registry = MechanismRegistry(
        (
            MechanismInfo("channel", "retained channel"),
            MechanismInfo("removed", "removed channel"),
            MechanismInfo("pump", "retained pump"),
        )
    )
    cell = _Cell()

    selected = registry.apply(cell, include=("channel", "pump"), exclude=())

    assert selected == {"channel", "pump"}
    assert cell.base.membrane_current_names == ["i_channel", "i_pump"]
