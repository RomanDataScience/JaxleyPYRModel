TITLE na3
: Na current for axon. No slow inact.
: M.Migliore Jul. 1997

NEURON {
	SUFFIX nax
	USEION na READ ena WRITE ina
	RANGE  gbar, fast_inactivation_tau_scale
	GLOBAL minf, hinf, mtau, htau,thinf, qinf
}

PARAMETER {
	gbar = 0.010
	fast_inactivation_tau_scale = 1   	(mho/cm2)	
								
	:tha  =  -30	(mV)		: v 1/2 for act	
       tha  =  -30	(mV)		: v 1/2 for act	
	qa   = 7.2	(mV)		: act slope (4.5)		
	Ra   = 0.4	(/ms)		: open (v)		
	Rb   = 0.124 	(/ms)		: close (v)		

	:thi1  = -45	(mV)		: v 1/2 for inact 	
	:thi2  = -45 	(mV)		: v 1/2 for inact 
      thi1  = -45(mV)		: v 1/2 for inact 	
	thi2  = -45 	(mV)		: v 1/2 for inact	
	qd   = 1.5	(mV)	        : inact tau slope
	qg   = 1.5      (mV)
	mmin=0.02	
	hmin=0.5			
	q10=2
	Rg   = 0.01 	(/ms)		: inact recov (v) 	
	Rd   = .03 	(/ms)		: inact (v)	

	thinf  = -50	(mV)		: inact inf slope	
	:qinf  = 4 	(mV)		: inact inf slope 
       qinf  =1 	(mV)		: inact inf slope 


	ena		(mV)            : must be explicitly def. in hoc
	celsius
	v 		(mV)
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	thegna		(mho/cm2)
	minf 		hinf 		
	mtau (ms)	htau (ms) 	
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
        thegna = gbar*m*m*m*h
	ina = thegna * (v - ena)
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

PROCEDURE trates(vm) {  
        LOCAL  a, b, qt, Arg
        qt=q10^((celsius-24)/10)
	a = trap0(vm,tha,Ra,qa)
	b = trap0(-vm,-tha,Rb,qa)
	mtau = 1/(a+b)/qt
        if (mtau<mmin) {mtau=mmin}
	minf = a/(a+b)

	a = trap0(vm,thi1,Rd,qd)
	b = trap0(-vm,-thi2,Rg,qg)
	htau =  1/(a+b)/qt
        htau = htau * fast_inactivation_tau_scale
        if (htau<hmin) {htau=hmin}
	
	Arg=(vm-thinf)/qinf
    
    if (Arg<-50) {hinf=1}
    else if (Arg>50) {hinf=0}    
    else {hinf=1/(1+exp(Arg))}
}

FUNCTION trap0(v,th,a,q) {
    LOCAL Arg
    
    if (fabs(v-th)>1e-6) {
        Arg=-(v-th)/q
        
        if (Arg<-50) {trap0=a*(v-th)}
        else if (Arg>50) {trap0=0}
        else {trap0=a*(v-th)/(1-exp(Arg))}
        
    }
    
    else {trap0=a*q}
}	

        

