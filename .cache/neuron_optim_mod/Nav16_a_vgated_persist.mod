TITLE Nav1.6 ionic voltage-gated channel with kinetic scheme (voltage-gated persistent variant)

COMMENT
A six-state markovian kinetic model of ionic channel.
Part of a study on kinetic models.
Author: Piero Balbi, August 2016
With Changes by Dr. Christopher Knowlton and Carol Upchurch

MODIFICATION (this file):
The original 'persist' parameter scaled the I1->O1 leak-back rate
(I1O1_a = persist*O1I1_a) uniformly at all voltages. This let a
fixed fraction of inactivated channels recycle into the open state
even at hyperpolarized/resting potentials, where no persistent Na
current should exist physiologically. That produced depolarization
block and spontaneous (stimulus-independent) oscillations when
persist was tuned to fit the depolarized plateau.

Here the leak-back rate is additionally multiplied by a sigmoidal
voltage gate, persist_gate(v), parameterized by p_vhalf and p_k.
The gate is close to 0 at rest/hyperpolarized potentials and close
to 1 near and above spike threshold, so the persistent component
only engages in the voltage range where it is meant to contribute
(e.g. during a depolarizing step or the interspike plateau), and
switches off automatically as the membrane repolarizes -- without
requiring any explicit stimulus-timing information.

See accompanying references for the biophysical motivation:
persistent Na current in real neurons is itself voltage-gated,
typically activating over roughly -65 to -40 mV, so this brings the
mechanism in line with experimentally characterized I_NaP rather
than treating 'persist' as a constant offset.
ENDCOMMENT

NEURON {
    SUFFIX na16a_vgp
    USEION na READ ena WRITE ina
    RANGE gbar, ina, g, dist, persist, slowdown, C1O1v2, I2init
    RANGE p_vhalf, p_k, fast_inactivation_tau_scale, slow_recovery_tau_scale
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    v (mV)
    ena (mV)
    celsius (degC)
    gbar  = 0.1  (mho/cm2)
    I2init=0

    C1O1b2    = 14
    C1O1v2    = 0       :0-18
    C1O1k2    = -6      :-10
    O1C1b1    = 4
    O1C1v1    = -48
    O1C1k1    = 9
    O1C1b2    = 0        :14
    O1C1v2    = 0         :-18
    O1C1k2    = -5.1      :-10
    O1I1b1    = 1         :6
    O1I1v1    = -42        :-40
    O1I1k1    = 12         :13
    O1I1b2    = 5      :10
    O1I1v2    = 10      :15
    O1I1k2    = -12     :-18
    I1C1b1    = 0.2     :0.1
    I1C1v1    = -65     :-86
    I1C1k1    = 10      :9

    C1I1b2    = 0.2     :0.08
    C1I1v2    = -65     :-55
    C1I1k2    = -11     :-12
    I1I2b2    = 0.022   :0.00022
    I1I2v2    = -25     :-25
    I1I2k2    = -5      :-5 -2

    I2I1b1    = 0.0018
    I2I1v1    = -50         :-50    -40
    I2I1k1    = 12          :12 1
    dist = 0
    slowdown = 0.2

    persist = 0          : max fraction of O1I1_a leaked back to O1
    p_vhalf = -50  (mV)  : half-activation voltage of the persistent gate
    p_k     = 5    (mV)  : slope of the persistent gate (mV); larger = softer onset
    p_vhalf2 = 0   (mV)
    p_k2     = 15  (mV)
    fast_inactivation_tau_scale = 1
    slow_recovery_tau_scale = 1
}

ASSIGNED {
    ina  (mA/cm2)
    g   (mho/cm2)

    C1O1_a (/ms)
    O1C1_a (/ms)
    O1I1_a (/ms)
    I1O1_a (/ms)
    I1I2_a (/ms)
    I2I1_a (/ms)
    I1C1_a (/ms)
    C1I1_a (/ms)

    Q10 (1)
    p_gate (1)
}

STATE {
    C1
    O1
    I1
    I2
}


INITIAL {
    Q10 = 3^((celsius-20(degC))/10 (degC))
    SOLVE kin
    STEADYSTATE sparse
}

BREAKPOINT {
    SOLVE kin METHOD sparse
    g = gbar * (O1) : (mho/cm2)
    ina = g * (v - ena)     : (mA/cm2)
}

KINETIC kin {
    rates(v)

    ~ C1 <->  O1 (C1O1_a, O1C1_a)
    ~ O1 <->  I1 (O1I1_a, I1O1_a)
    ~ I1 <->  C1 (I1C1_a, C1I1_a)
    ~ I1 <->  I2 (I1I2_a, I2I1_a)

    CONSERVE O1 + C1 + I1 + I2= 1
}

FUNCTION rates2(v, b, vv, k) {
    LOCAL Arg
    Arg=(v-vv)/k
    if (Arg<-50) {rates2=b}
    else if (Arg>50) {rates2=0}
    else {rates2 = (b/(1+exp(Arg)))}
}

FUNCTION persist_gate(v(mV)) {
    LOCAL rising, falling
    rising  = rates2(v, 1, p_vhalf,  -p_k)   : ~0 below p_vhalf, ~1 above
    falling = rates2(v, 1, p_vhalf2, p_k2)   : ~1 below p_vhalf2, ~0 above
    persist_gate = rising*falling
}

PROCEDURE rates(v(mV)) {
UNITSOFF

    C1O1_a = Q10*(rates2(v, C1O1b2, C1O1v2, C1O1k2))
    O1C1_a = Q10*(rates2(v, O1C1b1, O1C1v1, O1C1k1) + rates2(v, O1C1b2, O1C1v2, O1C1k2))
    O1I1_a = 0.5*Q10*(rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2)) / fast_inactivation_tau_scale

    p_gate = persist_gate(v)
    I1O1_a = persist*p_gate*O1I1_a

    I1C1_a = Q10*(rates2(v, I1C1b1, I1C1v1, I1C1k1)) / fast_inactivation_tau_scale
    C1I1_a = Q10*(rates2(v, C1I1b2, C1I1v2, C1I1k2)) / fast_inactivation_tau_scale
    I1I2_a = slowdown*dist*Q10*(rates2(v, I1I2b2, I1I2v2, I1I2k2)) / slow_recovery_tau_scale
    I2I1_a = slowdown*Q10*(rates2(v, I2I1b1, I2I1v1, I2I1k1)) / slow_recovery_tau_scale
UNITSON
}
