TITLE Nav1.6 ionic voltage-gated channel with kinetic scheme

COMMENT
A six-state markovian kinetic model of ionic channel.
Part of a study on kinetic models.
Author: Piero Balbi, August 2016
With Changes by Dr. Christopher Knowlton and Carol Upchurch

ENDCOMMENT

NEURON {
	SUFFIX na16a
	USEION na READ ena WRITE ina
	RANGE gbar, ina, g, dist, persist, slowdown, C1O1v2, I2init, fast_inactivation_tau_scale, slow_recovery_tau_scale
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
	ena (mV)
	celsius (degC)
	gbar  = 0.1
	fast_inactivation_tau_scale = 1
	slow_recovery_tau_scale = 1	 (mho/cm2)
	
	I2init=0
	

	C1O1b2	  = 14
	C1O1v2    = 0       :0-18
	C1O1k2	  = -6      :-10
	
	O1C1b1	  = 4
	O1C1v1    = -48
	O1C1k1	  = 9
	O1C1b2	  = 0        :14
	O1C1v2    = 0         :-18
	O1C1k2	  = -5.1      :-10
	
	O1I1b1	  = 1         :6
	O1I1v1	  = -42        :-40
	O1I1k1	  = 12         :13
	O1I1b2	  = 5      :10
	O1I1v2	  = 10		:15
	O1I1k2	  = -12		:-18
	
	I1C1b1	  = 0.2		:0.1
	I1C1v1	  = -65		:-86
	I1C1k1	  = 10		:9

	
	C1I1b2	  = 0.2		:0.08
	C1I1v2	  = -65		:-55
	C1I1k2	  = -11		:-12
	
	I1I2b2	  = 0.022	:0.00022
	I1I2v2	  = -25		:-25
	I1I2k2	  = -5		:-5	-2

	I2I1b1	  = 0.0018
	I2I1v1	  = -50			:-50	-40
	I2I1k1	  = 12			:12	1
	
	dist = 0
	slowdown = 0.2
	persist = 0
	
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
	g = gbar * (O1)	: (mho/cm2)
	ina = g * (v - ena)   	: (mA/cm2)
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

PROCEDURE rates(v(mV)) {
UNITSOFF

	
	C1O1_a = Q10*(rates2(v, C1O1b2, C1O1v2, C1O1k2))
	
	O1C1_a = Q10*(rates2(v, O1C1b1, O1C1v1, O1C1k1) + rates2(v, O1C1b2, O1C1v2, O1C1k2))
	
	O1I1_a = 0.5*Q10*(rates2(v, O1I1b1, O1I1v1, O1I1k1) + rates2(v, O1I1b2, O1I1v2, O1I1k2)) / fast_inactivation_tau_scale
	
	I1O1_a = persist*O1I1_a
	
	I1C1_a = Q10*(rates2(v, I1C1b1, I1C1v1, I1C1k1)) / fast_inactivation_tau_scale
	C1I1_a = Q10*(rates2(v, C1I1b2, C1I1v2, C1I1k2)) / fast_inactivation_tau_scale
	
	
	I1I2_a = slowdown*dist*Q10*(rates2(v, I1I2b2, I1I2v2, I1I2k2)) / slow_recovery_tau_scale
	I2I1_a = slowdown*Q10*(rates2(v, I2I1b1, I2I1v1, I2I1k1)) / slow_recovery_tau_scale
UNITSON
}
