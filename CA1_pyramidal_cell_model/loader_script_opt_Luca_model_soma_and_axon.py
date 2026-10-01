import matplotlib.pyplot as plt
import matplotlib
import numpy
from importlib import reload
import sys
import os
import importlib
import types
# print(sys.modules.keys())
import multiprocessing
import multiprocessing.pool
original_packs = sys.modules.copy()
try:
    import cPickle as pickle
except:
    import pickle
import gzip

from neuron import h
import neuron
from scipy import integrate
import pandas as pd


neuron.load_mechanisms('/home/tarluca/Modellke_story/Luca_modell_rework_2/prep_and_run_model_rework_new_K_stuff/nsg_csomag_model_rework_new/mod_files/')

h.load_file('/home/tarluca/Modellke_story/Luca_modell_rework_2/prep_and_run_model_rework_new_K_stuff/nsg_csomag_model_rework_new/load_model_na_inhomo_minimal_model_full_soma_f_fact_true_diam_active_spine_KA_fact_50Ra.hoc')

v = [-47.7233744163709, 0.011401720134011, 0.17866165872572, 0.110533546361937, 0.018195328014435, 4.13376819751696E-06, 0.558280689990683, -55.1977064419009, 0.000102007454842, 2.35151579716906, 0.00018773285843, 0.008177991377281, 0.082608363126436, 3.69975735209366, 0.00058114570326, 3.69310750687799, 5.1266274824495E-05, 1.15317561526235E-05, -25.8009959332103, 9.72470557972213, 9.75671751943343]

for sec in h.all:
    sec.gmax_Leak_pyr = v[5]
    sec.e_Leak_pyr = v[7]
    sec.cm = 1
    sec.Ra = 50
for sec in h.all_dendrites:
    sec.gkd_kd_params3=v[10]
    sec.shift_K_A_prox = v[19]
    sec.shift_K_A_dist = v[19]
    sec.X_k0_K_A_prox = v[20]
    sec.X_k0_K_A_dist = v[20]
    sec.Y_v0_Na_BG_dend = v[0]
    sec.gmax_K_DRS4_params_voltage_dep = v[1]
    sec.gbar_km_q10_2 = v[8]
    sec.pbar_car = v[17]
    sec.X_v0_Na_BG_dend = v[18]
    for seg in sec:
            #h('soma distance()')
            h.distance(sec=h.soma)
            #dist=(h.distance(seg.x))
            dist=(h.distance(seg.x, sec=sec)) 

            seg.gmax_Na_BG_dend = (v[2]+v[2]*(-0.002)*dist)
            seg.gk_sKCa = (v[16]+v[16]*(0.06)*dist)
    
            if (dist>100) and (dist<=150):
                    seg.gmax_H_CA1pyr_dist=(v[6]*0.00002+v[6]*4e-07*dist)
                    seg.gmax_H_CA1pyr_prox=0
                    seg.gmax_K_A_prox=0
                    seg.gmax_K_A_dist=(0.0035*v[9]+v[9]*5.5e-05*dist)
                    seg.pbar_CaL_pool2_ghk=(0.002*0.2*v[13]+v[13]*0.2*(-1.3333e-05)*dist)
            elif (dist>150.0) and (dist<=400.0):
                    seg.gmax_H_CA1pyr_dist=(v[6]*0.00002+v[6]*4e-07*dist)
                    seg.gmax_H_CA1pyr_prox=0
                    seg.gmax_K_A_prox=0
                    seg.gmax_K_A_dist=(0.0035*v[9]+v[9]*5.5e-05*dist)
                    seg.pbar_CaL_pool2_ghk=(0.002*0.2*v[13]+v[13]*0.2*(-1.3333e-05)*150)
            elif (dist<=100.0) and (dist>0.0):
                    seg.gmax_H_CA1pyr_dist=0
                    seg.gmax_H_CA1pyr_prox=(v[6]*0.00002+v[6]*4e-07*dist)
                    seg.gmax_K_A_prox=(0.0035*v[9]+v[9]*5.5e-05*dist)
                    seg.gmax_K_A_dist=0
                    seg.pbar_CaL_pool2_ghk=(0.002*0.2*v[13]+v[13]*0.2*(-1.3333e-05)*dist)
            elif (dist>400.0):
                    seg.gmax_H_CA1pyr_dist=0.00018*v[6]
                    seg.gmax_H_CA1pyr_prox=0
                    seg.gmax_K_A_prox=0
                    seg.gmax_K_A_dist=0.0255*v[9]
                    seg.pbar_CaL_pool2_ghk=(0.002*0.2*v[13]+v[13]*0.2*(-1.3333e-05)*150)
                    sec.gmax_Na_BG_dend = 0
            elif (dist<=0.0):
                    seg.gmax_H_CA1pyr_prox=0.00002*v[6]
                    seg.gmax_H_CA1pyr_dist=0
                    seg.gmax_K_A_prox=0.0035*v[9]
                    seg.gmax_K_A_dist=0
                    seg.pbar_CaL_pool2_ghk=0.002*0.2*v[13]
for sec in h.oblique_dendrites:
        sec.pbar_cat3 = 0.000072608* 0.01* v[15]
        sec.gmax_Na_BG_dend = v[2]*(2/3)
for sec in h.trunk:
        sec.pbar_cat3 = 0.000024292* 0.01* v[15]
for sec in h.basal_dendrites:
        sec.pbar_cat3 = 0.000069983* 0.01* v[15]
        sec.gmax_Na_BG_dend = v[2]*(2/3)
for sec in h.tuft:
        sec.pbar_cat3 = 0.000043133* 0.01* v[15]
        sec.gmax_Na_BG_dend = v[2]/3
for sec in h.soma:
    sec.gmax_Na_BG_soma = v[2]
    sec.Y_v0_Na_BG_soma = v[0]
    sec.gmax_K_DRS4_params_voltage_dep = v[3]
    sec.gmax_H_CA1pyr_prox = 0.00002*v[6] 
    sec.gbar_km_q10_2 = v[8]    
    sec.gmax_K_A_prox=0.0035*v[9] 
    sec.gkd_kd_params3=v[10]
    sec.pbar_CaN_BG_pool2_ghk=v[11]
    sec.gbar_bk_ch_pool=v[12]
    sec.pbar_CaL_pool2_ghk=0.00021*0.2*v[13]
    sec.gmax_K_AHP3_lpool = v[14]  
    sec.pbar_cat3 = 0.000002125* 0.01*v[15] 
    sec.gk_sKCa = v[16]
    sec.X_v0_Na_BG_soma = v[18]
    sec.shift_K_A_prox = v[19]
    sec.X_k0_K_A_prox = v[20]
for sec in h.all_axon:
    sec.gmax_Na_BG_axon = v[2]*40
    sec.Y_v0_Na_BG_axon = v[0]-3
    sec.gmax_K_DRS4_params_voltage_dep = v[3]*40
    sec.gmax_H_CA1pyr_prox = 0.00002*v[6]
    sec.gbar_km_q10_2 = v[8]*4
    sec.gkd_kd_params3=v[4]
    sec.X_v0_Na_BG_axon = v[18]-3
h.add_F_factor_to_all_dendrites()



stim = h.IClamp(h.soma(0.5))
stim.amp = 0.25
stim.delay = 500
stim.dur = 1000

sect_loc=h.soma(0.5)
axon = h.axon[0](0.5)

rec_t = h.Vector()
rec_t.record(h._ref_t)

rec_v = h.Vector()
rec_v.record(sect_loc._ref_v)

rec_v_axon = h.Vector()
rec_v_axon.record(axon._ref_v)

h.v_init = -70
h.tstop =1750
#h.finitialize(-70)
h.run()

t = numpy.array(rec_t)
v = numpy.array(rec_v)
v_axon = numpy.array(rec_v_axon)

plt.figure()
plt.plot(t, v_axon, 'r', label = 'axon')
plt.plot(t, v, label = 'soma')
plt.xlabel("time (ms)")
plt.ylabel("Voltage (mV)")
plt.legend(loc = 'upper left')
plt.title('Somatic and axonal response to the highest optimized current intensity (0.25 nA)') 
plt.tight_layout()
plt.savefig('spike_init_from_axon.svg')

plt.show()




