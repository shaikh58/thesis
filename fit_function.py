# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:45:02 2017

@author: Mustafa
"""
import numpy as np
from decimal import getcontext, Decimal
from scipy.optimize import curve_fit
from q_learning_algorithm import * 

int_temp_defn = [22, 23, 24, 25, 26]
num_ppl_defn = [0.1, 0.2, 0.3, 0.4, 0.5]
ext_temp_defn = [0.01, 0.02, 0.03]
getcontext().prec = 4
states_list = np.zeros(75)
a=0
for i in range(len(int_temp_defn)):
    for j in range(len(num_ppl_defn)):
        for k in range(len(ext_temp_defn)):
            states_list[a] = int_temp_defn[i] + num_ppl_defn[j] + ext_temp_defn[k]
            states_list[a] = Decimal(states_list[a])/1
            #states_list[a] = np.round(states_list[a], decimals = 2)
            a+=1
states_list_dict = {} 
for i in range(len(states_list)):
    states_list[i] = Decimal(states_list[i])/1
    states_list_dict.update({states_list[i]: i})
x1 = np.array([22,23,24,25,26])    
x = np.array([22,22,22,22,22,23,23,23,23,23,24,24,24,24,24,25,25,25,25,25,26,26,26,26,26])
y1 = (np.array(range(len(x1)),np.float64)+1)/10
y = np.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])

all_states = []
all_states_ext1 = []
all_states_ext3 = []
for i in xrange(len(x1)):
  for j in xrange(len(y1)):
    value =x1[i]+y1[j]+0.02
    value1 = x1[i]+y1[j]+0.01
    value3 = x1[i]+y1[j]+0.03
    value = np.round(value,2)
    value1 = np.round(value1,2)
    value3 = np.round(value3,2)
    all_states.append(value)
    all_states_ext1.append(value1)
    all_states_ext3.append(value3)
    
z = []
z1 = []
z3 = []
r_matrix2 = np.zeros((5,5))
r_matrix1 = np.zeros((5,5))
r_matrix3 = np.zeros((5,5))

#print 'ext temp interval 2:'
counter=0
for i in all_states:
  #print Q[states_list_dict[i]], np.argmin(Q[states_list_dict[i]])
  opt_action = np.argmin(Q[states_list_dict[i]]) + 22
  z.append(opt_action)
  counter+=1
#print 'ext temp interval 1:'
counter=0
for i in all_states_ext1:
  #print Q[states_list_dict[i]], np.argmin(Q[states_list_dict[i]])
  opt_action1 = np.argmin(Q[states_list_dict[i]]) + 22
  z1.append(opt_action1)
#print 'ext temp interval 3:'
counter=0
for i in all_states_ext3:
  #print Q[states_list_dict[i]], np.argmin(Q[states_list_dict[i]])
  opt_action3 = np.argmin(Q[states_list_dict[i]]) + 22
  z3.append(opt_action3)
  
int_temps = np.array([22,23,24,25,26])
num_ppls = np.array([1,2,3,4,5])
r_states1 = [22.11,23.11,24.11,25.11,26.11]
r_states2 = [22.12,23.12,24.12,25.12,26.12]
r_states3 = [22.13,23.13,24.13,25.13,26.13]
for k in xrange(len(int_temps)):
  r_matrix1[k] = R[states_list_dict[r_states1[k]]]
  r_matrix2[k] = R[states_list_dict[r_states2[k]]]
  r_matrix3[k] = R[states_list_dict[r_states3[k]]]
 
"""
#this block of code implements a curve fitting function, which maps combinations of internal temperature and occupancy level, to optimal setpoint temperatures (using the Q-matrix). A function was required in order to generate a 3-D surface which showed the optimal policy dynamics. However, the 2-D heat maps showed the policy in a more intuitive manner, and so the 3-D maps were never used. The code has been left here for general interest

coeffs = np.zeros(len(z)-1)
def func(X,a,a1,b,b1,c,c1,d,d1,e,e1,e2,e3,e4,e5,g1,g2,g3,g4):
  x,y = X
  '''
  x_coeffs = coeffs[:13]
  y_coeffs = coeffs[12:,]
  x_power = [1,2,3,4,5,6,7,8,9,10,11,12]
  x_array = [x,x,x,x,x,x,x,x,x,x,x]
  x_elem = np.power(x_array,x_power)
  weighted_x = np.sum(x_coeffs*x_elem)
  y_power = [1,2,3,4,5,6,7,8,9,10,11,12,13]
  y_array = [y,y,y,y,y,y,y,y,y,y,y,y,y]
  y_elem = np.power(y_array,y_power)
  weighted_y = np.sum(y_coeffs*y_elem)
  return np.sum(weighted_x,weighted_y)
  '''
  return a*x + a1*x**2 + b*y + b1*y**2 + c*x**3 + c1*y**3 + d*x**4 + d1*y**4 + e*x**5 + e1*y**5+e2*x**6+e3*y**6 #+ e4*x**7 + e5*y**7 + g1*x**8 + g2*y**8 + g3*x**9 + g4*y**9 +g5*x**10 + g6*y**10 +g7*x**11 + g8*y**11 +g9*x**12 + g10*y**12 + g11*x**13

p = curve_fit(func,(x,y),z)
coeffs = np.zeros(18)
for i in xrange(len(coeffs)):
   coeffs[i] = p[0][i]
a,a1,b,b1,c,c1,d,d1,e,e1,e2,e3,e4,e5,g1,g2,g3,g4=coeffs[0],coeffs[1],coeffs[2],coeffs[3],coeffs[4],coeffs[5],coeffs[6],coeffs[7],coeffs[8],coeffs[9],coeffs[10],coeffs[11],coeffs[12],coeffs[13],coeffs[14],coeffs[15],coeffs[16],coeffs[17]
func_eval = func((x,y),a,a1,b,b1,c,c1,d,d1,e,e1,e2,e3,e4,e5,g1,g2,g3,g4)
func_eval = np.round(func_eval)
"""
