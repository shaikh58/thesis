# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:27:22 2016
REV1: 16 Jan 2017
@author: Mustafa
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#Equations obtained from Estimation of the number of people under
#controlled ventilation using a CO2 concentration sensor (Hiroaki Nishi)
def calculate_pop(co2_int):
    min_co2 = np.min(co2_int)
    num_ppl = np.zeros(len(co2_int))

    met = 2
    delta_time = 0.5 #in hours
    gen_rate = 0.008 #L/s, (state of the art review of CO2 DCV)
    k = 3.6*gen_rate #convert L/s to m^3/hr
    #C0 = 400 #min CO2 = 285, but average is around 400; can do a more rigorous average
    V = 70
    Q = 0.06*800 #0.06cfm/ft^2 as per ASHRAE ventilation standard for supermarkets, but
    #paper on supermarket energy efficiency (http://www.nrel.gov/docs/fy15osti/63796.pdf) uses 0.5
    #t =
    #s =

    #Q = (-V/(t-s))*np.log((Ct - C0)/(Cs - C0))

    '''
    Ct is the CO2 concentration at time t (measured)
    C0 is the CO2 concentration when there is no one in the room (measured)
    Cs is the CO2 concentration at time s (measured)
    i = present time
    s = reference time
    Q = ventilation rate
    k = CO2 generation rate of people
    '''


    C0_array = []
    for i in xrange(len(co2_int)):
        #exp_term = np.exp((-Q/V)*(t-s))
        if co2_int[i]<500:
            if co2_int[i] - co2_int[i-1] < 0.05*co2_int[i]:
                if (i+1)!= len(co2_int) and co2_int[i+1] - co2_int[i] < 0.05*co2_int[i+1]:
                    C0_array.append(co2_int[i])
    C0 = np.average(C0_array)

    for i in xrange(len(co2_int)):
        if i>0:
            Cs = co2_int[i-1]
            Ct = co2_int[i]
            #Q = (-V/delta_time)*np.log((Ct - C0)/(Cs - C0))
            #print i
            #print (Ct-C0)/(Cs-C0)
            exp_term = np.exp((-Q/V)*delta_time)
            scale_factor = Q/(k*(1-exp_term))
            main_section = (Ct - C0 - (Cs - C0)*exp_term)
            num_ppl[i] = scale_factor*main_section
            if num_ppl[i] <0:
                num_ppl[i] = 0
            elif 100*12000< num_ppl[i] < 200*12000:
               num_ppl[i] = num_ppl[i]/1.8
            elif num_ppl[i] > 200*12000:
                num_ppl[i] = num_ppl[i]/3
        elif i == 0:
            pass
    num_ppl = num_ppl/18000

    '''
    x = 0
    for i in range(len(num_ppl)):
        if i!=0:
            if num_ppl[i] - num_ppl[i-1] > 100:
              x+=1
    print x
    '''
    '''Apply a Savitzky - Golay filter (low pass smoothing filter) to smooth out the
    high frequency components of the model.'''
    window_size = 101
    #num_ppl_spl = num_ppl[1:3000:1]
    num_ppl_filt = savgol_filter(num_ppl, window_size, 8)
    for i in range(len(num_ppl_filt)-1):
        if num_ppl_filt[i]<0:
            num_ppl_filt[i] = 0


    #x_axis = np.linspace(0,len(num_ppl),len(num_ppl))
    return num_ppl_filt
    #plt.plot(x_axis,num_ppl_spl)
    #plt.show()

    #plt.plot(x_axis,num_ppl_filt)
    #plt.show()
