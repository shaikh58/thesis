#this file generates various visualizations of the nature of the optimal policy using heat maps in matplotlib. See the thesis document results section to see what these visualizations look like
import matplotlib.pyplot as plt
import numpy as np
from fit_function import *


''' the following block of code generates a 2 dimensional heat map displaying the optimal actions
as a function of the internal temperature and occupancy level'''
z_matrix2 = np.zeros((5,5))
z_matrix1 = np.zeros((5,5))
z_matrix3 = np.zeros((5,5))
int_temps = np.array([22,23,24,25,26])
num_ppls = np.array([1,2,3,4,5])

q_3d_tint_ext1 = np.zeros((5,5,5))
q_3d_tint_ext2 = np.zeros((5,5,5))
q_3d_tint_ext3 = np.zeros((5,5,5))
q_3d_numppl_ext1 = np.zeros((5,5,5))
q_3d_numppl_ext2 = np.zeros((5,5,5))
q_3d_numppl_ext3 = np.zeros((5,5,5))
counter = 0
counter2=0 
for k in xrange(len(int_temps)):
  numppl_counter = 15*k
  tint_counter=3*k
  for l in xrange(len(num_ppls)):
    z_matrix2[k,l] = z[counter]
    z_matrix1[k,l] = z1[counter]
    z_matrix3[k,l] = z3[counter]
    q_3d_tint_ext1[counter2,l] = Q[tint_counter]
    q_3d_tint_ext2[counter2,l] = Q[tint_counter+1]
    q_3d_tint_ext3[counter2,l] = Q[tint_counter+2]
    q_3d_numppl_ext1[counter2,l] = Q[numppl_counter]
    q_3d_numppl_ext2[counter2,l] = Q[numppl_counter+1]
    q_3d_numppl_ext3[counter2,l] = Q[numppl_counter+2]
    tint_counter+=15
    numppl_counter+=3
    counter +=1 
  counter2+=1
  
iterations = terminal_i

"""
for i in xrange(q_3d_tint_ext1.shape[0]):

  q1_vis = plt.pcolor(q_3d_tint_ext1[i],cmap='Reds')
  plt.xticks(range(5),('22','23','24','25','26'))
  plt.yticks(range(5),('22','23','24','25','26'))
  plt.xlabel('Actions')
  plt.ylabel('Internal temperature')
  plt.colorbar(q1_vis)
  plt.savefig("C:/Users/User/Thesis/newq_original_data_old_energy_hvac1/q_matrix/Tint_ext1_numppl="+str(i) +"iterations="+str(iterations)+".png")
  plt.gcf().clear()
  
  q2_vis = plt.pcolor(q_3d_tint_ext2[i],cmap='Reds')
  plt.xticks(range(5),('22','23','24','25','26'))
  plt.yticks(range(5),('22','23','24','25','26'))
  plt.xlabel('Actions')
  plt.ylabel('Internal temperature')
  plt.colorbar(q2_vis)
  plt.savefig("C:/Users/User/Thesis/newq_original_data_old_energy_hvac1/q_matrix/Tint_ext2_numppl="+str(i)+"iterations="+str(iterations) +".png")
  plt.gcf().clear()
  
  q3_vis = plt.pcolor(q_3d_tint_ext3[i],cmap='Reds')
  plt.xticks(range(5),('22','23','24','25','26'))
  #plt.yticks(range(25),('22,very low','22,low','22,medium','22,high','22,very high','23,very low','23,low','23,medium','23,high','23,very high','24,very low','24,low','24,medium','24,high','24,very high','25,very low','25,low','25,medium','25,high','25,very high','26,very low','26,low','26,medium','26,high','26,very high'))
  plt.yticks(range(5),('22','23','24','25','26'))
  plt.xlabel('Actions')
  plt.ylabel('Internal temperature')
  plt.colorbar(q3_vis)
  plt.savefig("C:/Users/User/Thesis/newq_original_data_old_energy_hvac1/q_matrix/Tint_ext3_numppl="+str(i)+"iterations=" + str(iterations) +".png")
  plt.gcf().clear()
  
  q3_vis = plt.pcolor(q_3d_numppl_ext1[i],cmap='Reds')
  plt.xticks(range(5),('22','23','24','25','26'))
  plt.yticks(range(5),('<7','7-12','13-18','19-24','>25'))
  plt.xlabel('Actions')
  plt.ylabel('Number of people')
  plt.colorbar(q3_vis)
  plt.savefig("C:/Users/User/Thesis/newq_original_data_old_energy_hvac1/q_matrix/num_ppl/num_ppl_ext1_Tint="+str(i+22) +"iterations="+str(iterations)+".png")
  plt.gcf().clear()
  
  q3_vis = plt.pcolor(q_3d_numppl_ext2[i],cmap='Reds')
  plt.xticks(range(5),('22','23','24','25','26'))
  plt.yticks(range(5),('<7','7-12','13-18','19-24','>25'))
  plt.xlabel('Actions')
  plt.ylabel('Number of people')
  plt.colorbar(q3_vis)
  plt.savefig("C:/Users/User/Thesis/newq_original_data_old_energy_hvac1/q_matrix/num_ppl/num_ppl_ext2_Tint="+str(i+22) +"iterations="+str(iterations)+".png")
  plt.gcf().clear()
  
  q3_vis = plt.pcolor(q_3d_numppl_ext3[i],cmap='Reds')
  plt.xticks(range(5),('22','23','24','25','26'))
  plt.yticks(range(5),('<7','7-12','13-18','19-24','>25'))
  plt.xlabel('Actions')
  plt.ylabel('Number of people')
  plt.colorbar(q3_vis)
  plt.savefig("C:/Users/User/Thesis/newq_original_data_old_energy_hvac1/q_matrix/num_ppl/num_ppl_ext3_Tint="+str(i+22) +"iterations="+str(iterations)+".png")
  plt.gcf().clear()
"""

print 'External temperature: Low'

policy_vis1 = plt.pcolor(z_matrix1.T,vmin=22,vmax=26,cmap='Reds')
plt.yticks(range(len(num_ppls)),('<7','7-13','13-18','19-24','>25'))
plt.xticks(range(len(int_temps)),('22','23','24','25','26'))
plt.xlabel('Internal Temperature')
plt.ylabel('Number of people')
plt.colorbar(policy_vis1)
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/policyvis_low_" + "iterations="+ str(iterations)+"_"+"weight"+str(comfort_weight_raw)+"_"+"k_increment=" + str(k_increment)+"_alpha0="+str(alpha_init)+ ".png")
plt.gcf().clear()


print 'External temperature: Medium'

policy_vis2 = plt.pcolor(z_matrix2.T,vmin=22,vmax=26,cmap='Reds')
plt.yticks(range(len(num_ppls)),('<7','7-13','13-18','19-24','>25'))
plt.xticks(range(len(int_temps)),('22','23','24','25','26'))
plt.xlabel('Internal Temperature')
plt.ylabel('Number of people')
plt.colorbar(policy_vis2)
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/policyvis_medium_" +"iterations="+ str(iterations)+"_"+"weight"+str(comfort_weight_raw)+"_"+"k_increment=" + str(k_increment)+"_alpha0="+str(alpha_init)+ ".png")
#plt.savefig("C:/Users/User/Thesis/corrected_model_hvac1/3dsmoothing_epsilonstopping/policyvis2_" + str(iterations)+"_"+"weight"+str(comfort_weight)+ ".png")
plt.gcf().clear()

print 'External temperature: High'

policy_vis3 = plt.pcolor(z_matrix3.T,vmin=22,vmax=26,cmap='Reds')
plt.yticks(range(len(num_ppls)),('<7','7-13','13-18','19-24','>25'))
plt.xticks(range(len(int_temps)),('22','23','24','25','26'))
plt.xlabel('Internal Temperature')
plt.ylabel('Number of people')
plt.colorbar(policy_vis3)
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/policyvis_high_" + "iterations=" +  str(iterations)+"_"+"weight"+str(comfort_weight_raw)+"_"+"k_increment=" + str(k_increment)+"_alpha0="+str(alpha_init)+ ".png")
#plt.savefig("C:/Users/User/Thesis/corrected_model_hvac1/3dsmoothing_epsilonstopping/policyvis3_" + str(iterations)+"_"+"weight"+str(comfort_weight)+ ".png")
plt.gcf().clear()
