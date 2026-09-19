TITLE CA1 KM channel from Mala Shah
: M. Migliore June 2006

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {
	v 		(mV)
	ek
	celsius 	(degC)
	gbar=.0001 	(mho/cm2)
	gk       	(mho/cm2)
      : vhalfl=-40  	(mV)
     vhalfl=-42	(mV)
	kl=-4
       : vhalft=-42   	(mV)
       vhalft=-42 (mV)
        a0t=0.04  	(/ms)
      : a0t=0.009      	(/ms)

      : zetat=7    	(1)
      zetat=4
        gmt=.7	(1)
       	q10=5
	b0=60
	st=1
}


NEURON {
	SUFFIX km
	USEION k READ ek WRITE ik
        RANGE  gk,gbar,ik
      GLOBAL inf, tau
}

STATE {
        m
}

ASSIGNED {
	ik (mA/cm2)
        inf
	tau
        taua
	taub
}

INITIAL {
	rate(v)
	m=inf
}


BREAKPOINT {
	SOLVE state METHOD cnexp
	ik = gbar*m^st*(v-ek)
	gk = gbar*m^st
}


FUNCTION alpt(v(mV)) {
    LOCAL Arg
    Arg=0.0378*zetat*(v-vhalft)
    
    if (Arg<-50) {alpt=0}
    else if (Arg>50) {alpt=exp(50)}
    else {alpt=exp(Arg)}
}

FUNCTION bett(v(mV)) {
    LOCAL Arg
    Arg=0.0378*zetat*gmt*(v-vhalft)
    
    if (Arg<-50) {bett=0}
    else if (Arg>50) {bett=exp(50)}
    else {bett=exp(Arg)}
}

DERIVATIVE state {
        rate(v)
:        if (m<inf) {tau=taua} else {tau=taub}
	m' = (inf - m)/tau
}

PROCEDURE rate(v (mV)) { :callable from hoc
        LOCAL a,qt, Arg
        qt=q10^((celsius-35)/10)
        Arg=(v-vhalfl)/kl    
		if (Arg<-50) {inf=1}
		else if (Arg>50) {inf=0}
		else {inf=1/(1+exp(Arg))}
        a = alpt(v)
        tau = b0 + bett(v)/(a0t*(1+a))
:        taua = 50
:        taub = 300
}














