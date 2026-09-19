TITLE K-D
: K-D current for prefrontal cortical neuron ------Yuguo Yu  2007

NEURON {
	SUFFIX kd
	USEION k READ ek WRITE ik
	RANGE gk, gbar, i, deactivation_tau_scale
	GLOBAL minf, mtau, hinf, htau
}

PARAMETER {
	gbar = 0.1
	deactivation_tau_scale = 1   	(S/cm2)	
								
	celsius
	ek = -100	(mV)          
	v 		(mV)
	vhalfm=-43  (mV)
	km=8
	vhalfh=-67  (mV) 
      kh=7.3
	q10=2.3
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	i  (mA/cm2)
	ik 		(mA/cm2)
	minf
	mtau (ms)	 	
	hinf
	htau (ms)	 	
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
       i = gbar*m*h*(v-ek)
	   ik=i
} 

INITIAL {
	trates(v)
	m=minf  
	h=hinf  
}

DERIVATIVE states {   
        trates(v)      
        m' = (minf-m)/mtau
        h' = (hinf-h)/htau
}

PROCEDURE trates(v) {
	LOCAL qt, Arg1, Arg2
        qt=q10^((celsius-22)/10)

		Arg1=(v-vhalfm)/km
		if (Arg1<-50) {minf=0}
		else if (Arg1>50) {minf=1}
		else {minf=1-1/(1+exp(Arg1))}

		Arg2=(v-vhalfh)/kh
		if (Arg2<-50) {hinf=1}
		else if (Arg2>50) {hinf=0}
		else {hinf=1/(1+exp(Arg2))}

  	 mtau = 0.6 * deactivation_tau_scale
	 htau = 1500
}





