"""Parameter catalog for the CA1 pyramidal cell model.

The 21 fitted parameters are the vector ``v`` of
``loader_script_opt_Luca_model_soma_and_axon.py``; names and order follow the
header of ``last.csv``. Defaults are the loader's values. ``cm`` and ``Ra``,
which the loader sets uniformly to 1 uF/cm2 and 50 ohm cm, are optional.
"""

from __future__ import annotations

# name: (default, lower, upper)
CATALOG: dict[str, tuple[float, float, float]] = {
    "Na_inact": (-47.7233744163709, -60.0, -35.0),
    "gmax_K_DRS4_dend": (0.011401720134011, 0.002, 0.05),
    "gmax_Na_BG_soma": (0.17866165872572, 0.05, 0.5),
    "gmax_K_DRS4_soma": (0.110533546361937, 0.02, 0.4),
    "gmax_KD_axon": (0.018195328014435, 0.0, 0.06),
    "gmax_Leak_pyr": (4.13376819751696e-06, 1.0e-06, 2.0e-05),
    "H_ratio": (0.558280689990683, 0.1, 3.0),
    "e_Leak_pyr": (-55.1977064419009, -75.0, -45.0),
    "gbar_km_soma_dend": (0.000102007454842, 0.0, 0.0005),
    "KA_ratio": (2.35151579716906, 0.3, 8.0),
    "gbar_kd": (0.00018773285843, 0.0, 0.002),
    "pbar_CaN": (0.008177991377281, 0.0, 0.03),
    "gmax_bk": (0.082608363126436, 0.0, 0.3),
    "CaL_ratio": (3.69975735209366, 0.0, 10.0),
    "gmax_K_AHP": (0.00058114570326, 0.0, 0.003),
    "cat_ratio": (3.69310750687799, 0.0, 10.0),
    "gk_SK": (5.1266274824495e-05, 0.0, 0.0003),
    "pbar_car": (1.15317561526235e-05, 0.0, 0.0001),
    "Na_act": (-25.8009959332103, -40.0, -15.0),
    "KA_kinetic_shift": (9.72470557972213, 0.0, 20.0),
    "KA_slope": (9.75671751943343, 4.0, 16.0),
}

VECTOR_KEYS: tuple[str, ...] = tuple(CATALOG)

OPTIONAL: dict[str, tuple[float, float, float]] = {
    "cm": (1.0, 0.6, 1.6),
    "Ra": (50.0, 30.0, 200.0),
}

ALL_KEYS: tuple[str, ...] = VECTOR_KEYS
CATALOG_KEYS: tuple[str, ...] = VECTOR_KEYS + tuple(OPTIONAL)
DEFAULTS: dict[str, float] = {key: value[0] for key, value in {**CATALOG, **OPTIONAL}.items()}
BOUNDS: dict[str, tuple[float, float]] = {
    key: (value[1], value[2]) for key, value in {**CATALOG, **OPTIONAL}.items()
}
