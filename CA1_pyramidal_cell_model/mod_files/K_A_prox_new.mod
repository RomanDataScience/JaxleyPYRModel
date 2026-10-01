COMMENT

Based on Hoffmann et al. 1997. and Martina et al. 1998. 
Created by Luca Tar, 2024.

ENDCOMMENT


?  This is a NEURON mod file generated from a ChannelML file

?  Unit system of original ChannelML file: SI Units


TITLE Channel: K_A_prox



UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
    (um) = (micrometer)
    (molar) = (1/liter)
    (mM) = (millimolar)
    (l) = (liter)
}


    
NEURON {
      

    SUFFIX K_A_prox
    USEION k READ ek WRITE ik VALENCE 1  ? reversal potential of ion is read, outgoing current is written
           
        
    RANGE gmax, gion, shift, X_k0
    
    RANGE Xinf, Xtau
    
    RANGE Yinf, Ytau
    
    RANGE ik_ka_prox
    
}

PARAMETER { 
      

    gmax = 0.0090 (S/cm2)  ? default value, should be overwritten when conductance placed on cell
    X_k0 = 22
	X_vhalf = -9 (mV)
	X_tau0 = 1 (ms)
	q10 = 3
	shift = 0
	shift2 = 0
}



ASSIGNED {
      

    v (mV)
    
    celsius (degC)
          

    ? Reversal potential of k
    ek (mV)
    ? The outward flow of ion: k calculated by rate equations...
    ik (mA/cm2)
    
    ik_ka_prox (mA/cm2)
    gion (S/cm2)
    Xinf
    Xtau (ms)
    Yinf
    Ytau (ms)
    
}

BREAKPOINT { 
                        
    SOLVE states METHOD cnexp
         

    gion = gmax*((X)^2)*((Y)^1)      

    ik = gion*(v - ek)
    ik_ka_prox = ik
            

}



INITIAL {
    
    ek = -80
        
    rates(v)
    X = Xinf
        Y = Yinf
        
    
}
    
STATE {
    X
    Y
    
}

DERIVATIVE states {
    rates(v)
    X' = (Xinf - X)/Xtau
    Y' = (Yinf - Y)/Ytau
    
}

PROCEDURE rates(v(mV)) {  
    
    ? Note: not all of these may be used, depending on the form of rate equations
    LOCAL  alpha, beta, tau, inf, gamma, zeta, temp_adj_X, A_alpha_X, B_alpha_X, Vhalf_alpha_X, A_beta_X, B_beta_X, Vhalf_beta_X, temp_adj_Y, A_tau_Y, B_tau_Y, Vhalf_tau_Y, A_inf_Y, B_inf_Y, Vhalf_inf_Y
        
    TABLE Xinf, Xtau,Yinf, Ytau
 DEPEND celsius, X_vhalf, X_k0, X_tau0, q10, shift, shift2
 FROM -100 TO 50 WITH 3000
    
    
    UNITSOFF
    temp_adj_X = q10^((celsius-22)/10)
    temp_adj_Y = q10^((celsius-22)/10)
    
            
                
           

        
    ?      ***  Adding rate equations for gate: X  ***
        
    Xtau = X_tau0/temp_adj_X
    
	Xinf = 1/(1+exp(-(v-X_vhalf-shift)/X_k0))
       
    
    ?     *** Finished rate equations for gate: X ***
    

    
            
                
           

        
    ?      ***  Adding rate equations for gate: Y  ***
         
    ? Found a generic form of the rate equation for tau, using expression: v > -0.030 ? 0.26*(v + 0.030) + 0.005 : 0.005
    
    ? Note: Equation (and all ChannelML file values) in SI Units so need to convert v first...
    
    v = v * 0.0010   ? temporarily set v to units of equation...
            
    
    
    if (v > -0.030 ) {
        tau =  0.26*(v + 0.030) + 0.005 
    } else {
        tau =  0.005
    }
    ? Set correct units of tau for NEURON
    tau = tau * 1000 
    
    v = v * 1000   ? reset v
        
    Ytau = tau/temp_adj_Y
     
    ? Found a generic form of the rate equation for inf, using expression: 1 / (1 + (exp (125*(v + 0.051))))
    
    ? Note: Equation (and all ChannelML file values) in SI Units so need to convert v first...
    
    v = v * 0.0010   ? temporarily set v to units of equation...
            
    inf = 1 / (1 + (exp (125*(v + 0.056 - (shift/1000)))))
         
    
    v = v * 1000   ? reset v
        
    Yinf = inf
          
       
    
    ?     *** Finished rate equations for gate: Y ***
    

         

}


UNITSON


