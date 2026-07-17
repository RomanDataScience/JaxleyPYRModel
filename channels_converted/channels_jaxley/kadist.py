from .kaprox import Kap


class Kad(Kap):
    """Jaxley translation of `kadist.mod`."""

    def __init__(self, name=None):
        super().__init__(name)
        prefix = self._name
        self.channel_params.update(
            {
                f"{prefix}_vhalfn": -1.0,
                f"{prefix}_a0n": 0.1,
                f"{prefix}_zetan": -1.8,
                f"{prefix}_gmn": 0.39,
                f"{prefix}_nmin": 0.2,
            }
        )

