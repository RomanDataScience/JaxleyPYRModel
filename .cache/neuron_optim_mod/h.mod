TITLE  H-current
: Corrected and cleaned up by Val G. Rousseau on 07/17/20.

NEURON {
    SUFFIX h
    NONSPECIFIC_CURRENT i
    RANGE gbar,vhalf,K,taun,ninf,g,i,tau_scale
}

UNITS {
    (um)=(micrometer)
    (mA)=(milliamp)
    (uA)=(microamp)
    (mV)=(millivolt)
    (pmho)=(picomho)
    (mmho)=(millimho)
}

PARAMETER {
    dt (ms)
    v (mV)
    ena=50 (mV)
    eh=-10 (mV)
    K=8.5 (mV)
    gbar=0 (mho/cm2)                                            : initialize conductance to zero
    vhalf=-81 (mV)
    tau_scale=1                                              : half potential
}

STATE {                                                         : the unknown parameters to be solved in the DEs
    n
}

ASSIGNED {                                                      : parameters needed to solve DE
    i (mA/cm2)
    ninf
    taun (ms)
    g
}

INITIAL {                                                       : initialize the following parameter using states()
    states()
    n=ninf
    g=gbar*n
    i=g*(v-eh)
}

BREAKPOINT {
    SOLVE h METHOD derivimplicit
    g=gbar*n
    i=g*(v-eh)
}

DERIVATIVE h {
    states()
    n'=(ninf-n)/taun
}

FUNCTION MyExp(x) {
    if (x<-50) {MyExp=0}
    else if (x>50) {MyExp=exp(50)}
    else {MyExp=exp(x)}
}

PROCEDURE states() {
    taun=2*(1/(MyExp((v+186.32)/-29.91)+MyExp((v+21.84)/13.77)))    : Carol's equation

    taun = taun * tau_scale

    if (taun<5) {
        taun=5
    }

    ninf=1-(1/(1+MyExp((vhalf-v)/K)))                               : steady state value
}
