# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 01:04:08 2016

@author: Original - Simone, Conversion to Python- Mustafa- NOTE: original code written by Simone in MATLAB; I have just converted it to Python code''' 
"""
'''
%%%     F_PMV is a function for the determination of the thermal comfort
%%%     parameters: PMV and PPD


%%%---------------------------- Variables ------------------------------%%%
% Variables definition:
%   - Tint: Internal temperature
%   - Text_my: External mean Temperature of the year [°C]
%   - Text_mm: External mean Temperature of the month [°C]
%   - Text_ymin: minimum month external temperature of the year [°C]
%   - Text_ymax: maximum month external temperature of the year [°C]
%   - Icl: clothing insulation of the present month [m2 °C m2/W]
%   - Tsk: Mean skin Temperature
%   - Tcl: Clothing surface Temperature 
%   - v_air= air velocity: it can be assumed as 0.1 [m/s] (typical value for internal building)
%   - Icl_min: clothing Insulation of the hottest month. It can be assumed as
%     0.5clo  [m2 °C m2/W]
%   - Icl_max: clothing insulation of the coldest month. It can be assumed as 0.5clo  [m2 °C m2/W] Conversion-> 1 clo=0,15 m2*°C m2/W
%   - Icl_my: Mean clothing insulation of the year [m2 °C m2/W]
%   - M: Metabolic rate [W/m]
%   - Hu_int: Internal Humidity
%   - k1=eps*sigma*(Ar/Adu), wher Ar is the effective eadiant area of a
%   body, Adu is the DuBois body surface area, eps=emission coefficient of the body, sigma is the Stefan-Boltzman costant.
%   k1 can be correctly assumed as 39.6*10^-9
%   - H: Heat Loss from the body surface through convection, radiation and
%   conduction [W/m2]
%   - fcl: Clothing area factor 
%   - hc: Convective heat transfer coefficient [W/m2/°C] 
%   - p_sat: saturation pressure of water vapor in humid air [Pa]
%   - p_vap: Partial water vapour pressure in the air [Pa]
%   - Cres: Respiratory convective heat exchange [W/m2]
%   - Ec: Respiratory evaporative heat exchange at the skin when the
%   person experiencees a sensation of therma neutrality [W/m2]
%   - Eres: Respiratory evaporative heat exchange [W/m2]
%   - PMV: Predicted Mean Vote (Thermal sensation)
%   - PPD: Predicted Percentage Dissatisfied

%%%-------------------------- Assumptions ------------------------------%%%
% The following variables arecostants that have to be fixed at the
% beginning
%   - v_air=0.1 m/s well verified for indoor environment
%   - Icl_max=1.0 clo 
%   - Icl_min=0.5 clo
%   - Tr=Tint-> mean radiant temperature (temperature of the walls) is considered equal to the internal temperature 
%   - M has to be fixed. Maybe an average between 1.2met (sedentary work, activity that characterized workers at supermarket)and 1.9met (walking, activity that characterized costumers moving in the supermarket)
%%%---------------------------------------------------------------------%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
import numpy as np 
import matplotlib.pyplot as plt

def F_PMV(Tint, Text_mm, v_air, Icl_min,Icl_max, Text_ymin, Text_ymax, M, Hu_int):

    #determination of the actual clothing insulation for the current period (it is a linear function of the external temperature)
    Icl_my=(Icl_min+Icl_max)/2
    #Icl= Icl_my+(Icl_max-Icl_min)*(Text_mm-Text_my)/(Text_ymax-Text_ymin); %ATTENTION icl Has to be written in [°C*m2/W]
    Icl= Icl_max+(-Icl_max+Icl_min)*(Text_mm-Text_ymin)/(Text_ymax-Text_ymin) #ATTENTION icl Has to be written in [°C*m2/W]
    Icl=Icl/0.15
    Icl=np.round(Icl*10)/10 # Icl can change only taking discrete staps of 0.1 [clo]
    Icl=Icl*0.15
    
    
    # determination of Tcl essential to calculate the heat loss from the body surface H
    Tsk=35.7-0.028*M
    k1=39.6*(10**-9)
    if Icl < 0.078:
        fcl=1+1.29*Icl 
    else:
        fcl=1.05+0.645*Icl

    Tcl_new=Tint*1.2 #initialization of Tcl
    err=10
    while np.abs(err)>0.0001:
        Tcl=Tcl_new
        if (2.38*(Tcl-Tint)**0.25) > 12.1*np.sqrt(v_air):
            hc=2.38*(Tcl-Tint)**0.25
        else:
            hc=12.2*np.sqrt(v_air)
        
        Tcl_new=Tsk-Icl*k1*fcl*((Tcl+273)**4-(Tint+273)**4)-Icl*fcl*hc*(Tcl-Tint)
        err=Tcl_new-Tcl

    Tcl=Tcl_new
    #determination of the Heat loss from the body surface H
    H=k1*fcl*((Tcl+273)**4-(Tint+273)**4)+fcl*hc*(Tcl-Tint)
    
    #determination of the PMV coefficient
    p_sat=np.exp((17.438*Tint/(239.78+Tint))+6.4147)# %pressure in [Pascal] Tint in (°C)->this equation has to be checked
    p_vap=Hu_int*p_sat
    Cres=0.0014*M*(34-Tint)
    Ec=3.05*10**(-3)*(5733-6.99*M-p_vap)+0.42*(M-58.15)
    Eres=1.72*10**(-5)*M*(5867-p_vap)
    PMV=(0.303*np.exp(-0.036*M)+0.028)*(M-H-Ec-Cres-Eres)
    PPD=100-95*np.exp(-0.03353*PMV**4-0.2179*PMV**2)
    
    return PMV, PPD

def test_PMV(Tint,Text_mm):
    
    #Tint=22
    #Text_mm=26
    v_air=0.1
    Icl_min=0.5*0.15
    Icl_max=1.0*0.15
    Text_ymin=0
    Text_ymax=27
    M=1.4*58
    Hu_int=0.52
    #Text_my=(Text_ymin+Text_ymax)/2;
    
    [PMV, PPD]=F_PMV(Tint, Text_mm, v_air, Icl_min,Icl_max, Text_ymin, Text_ymax, M, Hu_int)
    return [PMV, PPD]

if __name__ == "__main__":
  
  v_air=0.1
  Icl_min=0.5*0.15
  Icl_max=1.0*0.15
  Text_ymin=0
  Text_ymax=27
  M=1.4*58
  Hu_int=0.5415
  
  
  c_matrix = np.zeros((5,8))
  int_temps = [22,23,24,25,26]
  ext_temps = [20,22,24,26,28,30,32,34]
  for k in xrange(len(int_temps)):
    for l in xrange(len(ext_temps)):
      c_matrix[k,l] = F_PMV(int_temps[k], ext_temps[l], v_air, Icl_min,Icl_max, Text_ymin, Text_ymax, M, Hu_int)[0]
  
  
  comfort_vis = plt.pcolor(c_matrix,cmap=plt.cm.coolwarm)
  
  plt.yticks(range(5),('22','23','24','25','26'))
  plt.xticks(range(8),('20','22','24','26','28','30','32','34'))
  
  
  plt.xlabel('External Temperature')
  plt.ylabel('Internal Temperature')
  plt.colorbar(comfort_vis)
  plt.show()  
  
  flat_matrix = c_matrix.reshape(-1)
  var = np.var(flat_matrix)