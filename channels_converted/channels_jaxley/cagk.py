from .common import Channel, FARADAY_KC, R, channel_prefix, gate_update, safe_exp, state_or_param


class MyKca(Channel):
    """Jaxley translation of `cagk.mod` (`SUFFIX mykca`)."""

    def __init__(self, name=None):
        self.current_is_in_mA_per_cm2 = True
        super().__init__(name)
        prefix = channel_prefix(self)
        self.channel_params = {
            f"{prefix}_gkbar": 0.01,
            f"{prefix}_cainit": 100e-6,
            f"{prefix}_d1": 1.0,
            f"{prefix}_d2": 1.5,
            f"{prefix}_k1": 0.18,
            f"{prefix}_k2": 0.011,
            f"{prefix}_bbar": 0.28,
            f"{prefix}_abar": 0.48,
            f"{prefix}_st": 1.0,
            "eK": -85.0,
            "celsius": 20.0,
        }
        self.channel_states = {f"{prefix}_o": 0.0}
        self.current_name = "i_K"

    def update_states(self, states, dt, v, params):
        prefix = channel_prefix(self)
        cai = state_or_param(states, params, "CaCon_i", params[f"{prefix}_cainit"])
        oinf, tau = self.rates(v, cai, params)
        return {f"{prefix}_o": gate_update(states[f"{prefix}_o"], dt, oinf, tau)}

    def compute_current(self, states, v, params):
        prefix = channel_prefix(self)
        return params[f"{prefix}_gkbar"] * states[f"{prefix}_o"] ** params[f"{prefix}_st"] * (v - params["eK"])

    def init_state(self, states, v, params, delta_t):
        prefix = channel_prefix(self)
        oinf, _ = self.rates(v, params[f"{prefix}_cainit"], params)
        return {f"{prefix}_o": oinf}

    def exp1(self, k, d, v, params):
        return k * safe_exp(-2.0 * d * FARADAY_KC * v / R / (273.15 + params["celsius"]))

    def rates(self, v, cai, params):
        prefix = channel_prefix(self)
        alpha = cai * params[f"{prefix}_abar"] / (
            cai + self.exp1(params[f"{prefix}_k1"], params[f"{prefix}_d1"], v, params)
        )
        beta = params[f"{prefix}_bbar"] / (
            1.0 + cai / self.exp1(params[f"{prefix}_k2"], params[f"{prefix}_d2"], v, params)
        )
        tau = 1.0 / (alpha + beta)
        return alpha * tau, tau

