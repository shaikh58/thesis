'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from fit_function import *

#fig = plt.figure(figsize=(10,8))
#ax = fig.gca(projection='3d')

# Make data.

#X = np.arange(-5, 5, 0.25)
#Y = np.arange(-5, 5, 0.25)
#X = np.array([22,23,24,25,26])
#Y = np.array(range(len(X)))
#X = np.array([22,22,22,22,22,23,23,23,23,23,24,24,24,24,24,25,25,25,25,25,26,26,26,26,26])

#Y = np.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])
#X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
#Z = np.sin(R)
#Z = fit_function.func_eval

# Plot the surface.
'''
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zticks([22,23,24,25,26])
ax.set_yticks([1,2,3,4,5])
ax.set_xlabel('Internal temperature')
ax.set_ylabel('Number of people')
ax.set_zlabel('Optimal action')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#for i in xrange(len(X)-1):
    #print fit_function.z[i],Z[i]
'''
#mean_squared_error = np.sum(np.power(z-Z, 2))/(2*len(Z))    
#print 'mse: ', mean_squared_error
#create a matrix Z which maps an x,y combination to a z value - i.e. Z[x,y]=z

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

#reward_display1 = plt.pcolor(r_matrix1,cmap='Reds')
#plt.yticks(range(len(int_temps)),('22','23','24','25','26'))
#plt.xlabel('Action (setpoint)')
#plt.ylabel('Internal temperature')
policy_vis1 = plt.pcolor(z_matrix1.T,vmin=22,vmax=26,cmap='Reds')
plt.yticks(range(len(num_ppls)),('<7','7-13','13-18','19-24','>25'))
plt.xticks(range(len(int_temps)),('22','23','24','25','26'))
plt.xlabel('Internal Temperature')
plt.ylabel('Number of people')
plt.colorbar(policy_vis1)
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/policyvis_low_" + "iterations="+ str(iterations)+"_"+"weight"+str(comfort_weight_raw)+"_"+"k_increment=" + str(k_increment)+"_alpha0="+str(alpha_init)+ ".png")
#plt.colorbar(reward_display1)
#plt.savefig("C:/Users/User/Thesis/corrected_model_hvac1/3dsmoothing_epsilonstopping/policyvis1_" + str(iterations)+"_"+"weight"+str(comfort_weight)+ ".png")

#plt.show()
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
'''
reward_display2 = plt.pcolor(r_matrix2,cmap='Reds')
plt.yticks(range(len(int_temps)),('22','23','24','25','26'))
plt.xlabel('Action (setpoint)')
plt.ylabel('Internal temperature')
plt.colorbar(reward_display2)
plt.show()
plt.gcf().clear()
'''
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
'''
reward_display3 = plt.pcolor(r_matrix3,cmap='Reds')
plt.xticks(range(len(int_temps)),('22','23','24','25','26'))
plt.yticks(range(len(int_temps)),('22','23','24','25','26'))
plt.xlabel('Action (setpoint)')
plt.ylabel('Internal temperature')
plt.colorbar(reward_display3)
plt.show()
'''
plt.gcf().clear()
