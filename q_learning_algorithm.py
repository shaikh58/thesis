# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 23:22:21 2016
THIS REVISION:  April 2017
@author: Mustafa

To run on Bash for Windows, first execute command 'export KMP_AFFINITY=disabled' before running
"""

import numpy as np
import trans_mat_single_agent
import comfort_index
import timeit
import matplotlib.pyplot as plt
#from decimal import getcontext, Decimal

from itertools import product

start = timeit.default_timer()

def import_trans_prob_mat():
    ''' this function imports the transition probability matrices calculated
in 'trans_mat_single_agent.py'. It imports both the excel data, which will be used in the reward function
as well as the transition matrices'''
    imported_data = trans_mat_single_agent.ret_trans_matrix(22,23,24,25,26)
    imported_excel_data = imported_data[1]
    states_info = imported_data[2]
    raw_states_matrix_total = imported_data[4]
    return imported_excel_data, trans_mat_single_agent.ret_trans_matrix(22,23,24,25,26)[0], states_info,raw_states_matrix_total

def import_comfort_index(Tint, Text_mm, v_air, Icl_min,Icl_max, Text_ymin, Text_ymax, M, Hu_int):
    '''this function imports the comfort index code'''
    PMV = comfort_index.F_PMV(Tint, Text_mm, v_air, Icl_min,Icl_max, Text_ymin, Text_ymax, M, Hu_int)[0]
    return PMV

v_air=0.1
Icl_min=0.5*0.15
Icl_max=1.0*0.15
Text_ymin=0
Text_ymax=27
M=1.4*58
Hu_int=0.5415

#define local variables for all 5 transition probability matrices
all_imported_data = import_trans_prob_mat()

all_trans_mat = all_imported_data[1]
trans_mat22 = all_trans_mat[0]
trans_mat23 = all_trans_mat[1]
trans_mat24 = all_trans_mat[2]
trans_mat25 = all_trans_mat[3]
trans_mat26 = all_trans_mat[4]
trans_mat_array = [trans_mat22, trans_mat23, trans_mat24, trans_mat25, trans_mat26]
#get a list of all the possible states, and a dictionary as well
states_info = all_imported_data[2]
states_list = states_info[0]
states_list_dict = states_info[1]
raw_states_matrix_complete = all_imported_data[3][0]
raw_action_set = all_imported_data[3][1]


imported_excel_data = all_imported_data[0]
ext_temp_raw = imported_excel_data[3]
hu_ext =  imported_excel_data[4]
int_temp_raw = imported_excel_data[1]
co2_int = imported_excel_data[5]
avg_num_ppl = imported_excel_data[6]

actions = [22,23,24,25,26]
#initialize the Q, R matrices
Q = np.zeros((75,5))
R = np.zeros_like(Q)
gamma = 0.99
#######################################################################
#This block of code initializes temporary arrays for standard deviation and variance determinations later on
Re = np.zeros_like(R)
Re_newmodel = np.zeros_like(R)
Rc = np.zeros_like(R)
VAR_Re = np.zeros_like(R)
VAR_Rc = np.zeros_like(R)
test_pts = 50
comfort_weights = np.linspace(10,1000,test_pts -1)
var_rcvsb = np.zeros_like(comfort_weights)
sd_rcvsb = np.zeros_like(comfort_weights)
sd_revsb = np.zeros_like(comfort_weights)
var_revsb = np.zeros_like(comfort_weights)
########################################################################
initial_state = np.random.choice(74)

#average relative humidity is 54.15 (0.5415) which is used for now and average co2_int = 598 which is used for now

def compute_rewards(states_list,actions,avg_num_ppl,comfort_weight_raw):
  comfort_weight = comfort_weight_raw/avg_num_ppl 
  ##########################-----------COST FUNCTION----------########################### 
  ext_temp_rep = [18, 24, 28]#Take a representative external temperature for the ext_temp_raw part of the energy cost function calc
  for i in xrange(len(R[:,1])):
      for j in xrange(len(R[1,:])):
          #delta_t = actions[j] - np.floor(states_list[i])
          #R[i,j] = a*delta_t
          #choose a representative external temperature in each category
          ext_temp_obs = np.round((((states_list[i] % 1)*10) % 1)*10)
          if ext_temp_obs == 1:
              ext_temp_rel = ext_temp_rep[0]
          elif ext_temp_obs == 2:
              ext_temp_rel = ext_temp_rep[1]
          elif ext_temp_obs ==3:
              ext_temp_rel = ext_temp_rep[2]
          #Choose a representative population number
          num_ppl_obs = np.round((states_list[i]%1)*10)
          if num_ppl_obs == 1:
              num_ppl = 4
          if num_ppl_obs ==2:
              num_ppl = 9
          if num_ppl_obs ==3:
              num_ppl = 15
          if num_ppl_obs ==4:
              num_ppl = 22
          if num_ppl_obs ==5:
              num_ppl=30
                
          '''-----------HERE IS THE ENERGY COST AND THE COMFORT VALUE CALCULATION------------------'''
          energy_cost = np.round(((ext_temp_rel*9.20269047720525)+(54.15*0.362094384564653)+(np.floor(states_list[i])*8.4172944349841)+(598.26*0.092321902035293)+actions[j]*(-17.8734109048992)+6.5059040794673))#NEWMODEL  

          comfort_value = np.abs(import_comfort_index(np.floor(states_list[i]), ext_temp_rel, v_air, Icl_min,Icl_max, Text_ymin, Text_ymax, M, Hu_int))

          '''----------HERE IS THE REWARD (COST) FUNCTION -----------------'''
          R[i,j] = float(energy_cost + num_ppl*comfort_weight*np.abs(comfort_value))
          Re[i,j] = float(energy_cost)
          Rc[i,j] = float(num_ppl*comfort_weight*np.abs(comfort_value))
  return R

'''
#this block of code visualizes the nature of the reward function in a heat map
x = plt.pcolor(R,vmin=-500,vmax=500,cmap=plt.cm.coolwarm)        
plt.colorbar(x)
plt.xticks(range(6),('','22','23','24','25','26'))
plt.xlabel('Action (setpoint)')
plt.ylabel('State')
'''
        
''' 
#this is the block of code which computes the variance and standard deviation for determination of an appropriate comfort weight
var_rcvsb[z] = np.var(Rc)
sd_rcvsb[z] = np.std(Rc)
sd_revsb[z] = np.std(Re)
var_revsb[z] = np.var(Re)
var_Re = np.var(Re)
sd_Re = np.std(Re)
var_Rc = np.var(Rc)
VAR_Re = np.var(Re,axis=1)
VAR_Rc = np.var(Rc,axis=1)
#plt.plot(comfort_weights,var_rcvsb)
plt.plot(comfort_weights,var_revsb)
plt.plot(comfort_weights,var_rcvsb)
plt.legend(['Variance of energy cost', 'Variance of discomfort cost'], loc='best')
plt.xlabel('Comfort weight')
plt.ylabel('Variance of cost term')
plt.show()
plt.gcf().clear()
plt.plot(comfort_weights,sd_revsb)
plt.plot(comfort_weights,sd_rcvsb)
plt.legend(['Standard deviation of energy cost', 'Standard deviation of discomfort cost'], loc='best')
plt.xlabel('Comfort weight')
plt.ylabel('Variance of cost term')
plt.show()
closest_points_sd = sorted(product(sd_rcvsb,sd_revsb), key=lambda t: abs(t[0]-t[1]))[0]#sorts a combination of every pair of each array (n^2 entries) according to the smallest difference between the pair's elements
closest_points_var = sorted(product(var_rcvsb,var_revsb), key=lambda t: abs(t[0]-t[1]))[0]
equalizing_weight_sd = comfort_weights[np.where(sd_rcvsb==closest_points_sd[0])[0][0]]#finds the weight corresponding to the intersection point between the two arrays
equalizing_weight_var = comfort_weights[np.where(var_rcvsb==closest_points_var[0])[0][0]]
#intersection = np.intersect1d(sd_rcvsb,sd_revsb)
'''

def find_next_action(available_actions_range):
    next_action = int(np.random.choice(actions, 1))
    return next_action

action = find_next_action(actions)
normalized_action = int(action - 22)


def update_Q(R, alpha,current_state, actions, normalized_action, action, gamma, trans_mat_array):
    '''this is the Q-value update function. The args are passed in and the Q-value is updated as per the Q-learning algorithm presented by Watkins, 1989'''
    possible_next_states = []
    next_state_prob = [] #the probabilities associated with each possible next state
    counter = 0
    normalized_action_alt = normalized_action
    '''this loop makes sure that the system progresses from current state to any other state for a given temperature setpoint - if it didn't, then according to the system dynamics, there is 0 probability of going to another state for that setpoint. Therefore another setpoint must be chosen'''
    while trans_mat_array[normalized_action_alt][current_state].any() == np.zeros_like(trans_mat_array[normalized_action_alt][current_state]).any():
      #current_state = counter
      action_alt = actions[counter]#if the chosen action doesn't cause a state transition, choose a new action 
      normalized_action_alt = int(action_alt - 22)
      counter+=1
      if counter == len(R[current_state]):#if none of the actions have a transition from current_state, choose another current_state!
      #if counter == len(trans_mat_array[normalized_action][counter])+1:#here another setpoint is chosen in the event that the system doesn't ever go to another state from a given state
              current_state = np.random.randint(0, len(Q[:,1]))#pick a new current_state!
              normalized_action_alt = normalized_action
              counter = 0
    '''this block populates the next possible states by going through the transition matrix for the given setpoint. All occuring next states are appended into an array, along with their corresponding probabilities'''
    for i in xrange(len(trans_mat_array[normalized_action_alt][current_state])):
        if trans_mat_array[normalized_action_alt][current_state][i] != 0:
            possible_next_states.append(states_list[i])
            next_state_prob.append(trans_mat_array[normalized_action_alt][current_state][i])
        else:
          pass

    '''next state chosen by generating a uniformly distributed random variable, with probabilities proportional to the probabilities given by the transition matrix'''

    next_state = np.random.choice(possible_next_states, p=next_state_prob)
    next_state = np.round_(next_state,2)
    next_state_norm = states_list_dict[next_state]

    #here is the actual Q-value update
    Q[current_state, normalized_action] = (1-alpha)*Q[current_state, normalized_action] + alpha*(R[current_state,normalized_action]+gamma*np.min(Q[next_state_norm,:]))

    return

def train_Q(R,alpha_0, k_increment,num_iterations):
  ''' this function trains the Q matrix over num_iterations number of iterations'''
  sum_qrow1 = [0]
  sum_qrow2 = [0]
  sum_qrow3 = [0]
  sum_qrow4 = [0]
  sum_qrow5 = [0]
  sum_qrow6 = [0]
  sum_qrow7 = [0]
  sum_qrow8 = [0]
  sum_qrow9 = [0]
  sum_qrow10 = [0]
  sum_qrow11 = [0]
  sum_qrow12 = [0]
  
  l = 1
  k = 5
  init_zero_rows = []
  counter=0
  learningrate_div = 1.0
  alpha = alpha_0/learningrate_div
  #initial update
  
  action = find_next_action(actions)
  normalized_action = action - 22
  update_Q(R,alpha,initial_state, actions, normalized_action, action, gamma, trans_mat_array)
  epsilon=0.01

  for i in xrange(num_iterations):
      #for each iteration, choose a random state and a random action
      #row_var[i] = np.var(Q)
      ''' Epsilon greedy stopping approach
      q_avg[i] = np.average(Q[44])
      if i>0:
        q_avg_diff[i] = q_avg[i] - q_avg[i-1]
        #rowvar_diff[i] = row_var[i]-row_var[i-1]
      if i>110000:
      #if i>10000:
        if q_avg_diff[i-1] < epsilon:
        #if (np.abs(rowvar_diff[i])+np.abs(rowvar_diff[i-1])+np.abs(rowvar_diff[i-2])+np.abs(rowvar_diff[i-3]))/4.0 < epsilon:
          print q_avg_diff[i]
          #print (np.abs(rowvar_diff[i])+np.abs(rowvar_diff[i-1]))/3.0
          break
      '''
      current_state = np.random.randint(0, len(Q[:,1]))
      action = find_next_action(actions)
      normalized_action = action - 22
      update_Q(R,alpha,current_state, actions, normalized_action, action, gamma, trans_mat_array)
      if current_state == initial_state:
        learningrate_div += k_increment
        alpha = float(1.0/learningrate_div)
      if i%20000 == 0:
        print "i= ", i
      if i>(5000*l) and i%(200*k) == 0:
        '''the following block of code implements a smoothing technique to smoothen the Q-values. This is a customization that we made to deal with the issues arising with a sparse dataset'''
        for j in xrange(Q.shape[0]):
          if Q[j].all() == np.zeros_like(Q[j]).all():
            if counter == 0:
              init_zero_rows.append(j)
              if j==0:
                Q[j] = 0.5*Q[j+1] + 0.5*Q[j+2]
              elif 1<=j<=74:
                try:
                  if np.round((((states_list[j] % 1)*10) % 1)*10)==1:
                   Q[j] =  1/6*(Q[j-15] +Q[j+15])+ 1/6*(Q[j-3]+Q[j+3]) + 1/6*(Q[j+1]+Q[j+2])
                  elif np.round((((states_list[j] % 1)*10) % 1)*10)==2:
                   Q[j] =  1/6*(Q[j-15] +Q[j+15])+ 1/6*(Q[j-3]+Q[j+3]) + 1/6*(Q[j-1]+Q[j+1])
                  elif np.round((((states_list[j] % 1)*10) % 1)*10)==3:
                   Q[j] = 1/6*(Q[j-15] +Q[j+15])+ 1/6*(Q[j-3]+Q[j+3]) + 1/6*(Q[j-2]+Q[j-1])
                  #Q[j] = 0.5*Q[j-3] + 0.5*Q[j+3]
                except:
                  try: 
                    if np.round((((states_list[j] % 1)*10) % 1)*10)==1:
                      Q[j] =  1/4*(Q[j-3]+Q[j+3]) + 1/4*(Q[j+1]+Q[j+2])
                    elif np.round((((states_list[j] % 1)*10) % 1)*10)==2:
                      Q[j] =  1/4*(Q[j-3]+Q[j+3]) + 1/4*(Q[j-1]+Q[j+1])
                    elif np.round((((states_list[j] % 1)*10) % 1)*10)==3:
                      Q[j] =  1/4*(Q[j-3]+Q[j+3]) + 1/4*(Q[j-2]+Q[j-1])
                  except:
                    if np.round((((states_list[j] % 1)*10) % 1)*10)==1:
                      Q[j] =  1/2*(Q[j+1]+Q[j+2])
                    elif np.round((((states_list[j] % 1)*10) % 1)*10)==2:
                      Q[j] =  1/2*(Q[j-1]+Q[j+1])
                    elif np.round((((states_list[j] % 1)*10) % 1)*10)==3:
                      Q[j] =  1/2*(Q[j-2]+Q[j-1])     
        counter+=1
        if counter>0:
         for j in init_zero_rows:#for all rows which need to be smoothened
           if j == 0:# if we are in the first state
             Q[j] = 0.8*Q[j] + 0.1*Q[j+1] + 0.1*Q[j+2]
           elif 1<=j<=74:#if we are not in the first state
           
             try:#try a complete 3 directional smoothing
               if np.round((((states_list[j] % 1)*10) % 1)*10)==1:#for each possible external temperature interval, we must choose the other two states with the same external temperature. The reason for this is that otherwise, we would be varying more than 1 parameter at a time
                 Q[j] = 0.85*Q[j] + 0.05/2*(Q[j-15] +Q[j+15])+ 0.05/2*(Q[j-3]+Q[j+3]) + 0.05/2*(Q[j+1]+Q[j+2])
               elif np.round((((states_list[j] % 1)*10) % 1)*10)==2:
                 Q[j] = 0.85*Q[j] + 0.05/2*(Q[j-15] +Q[j+15])+ 0.05/2*(Q[j-3]+Q[j+3]) + 0.05/2*(Q[j-1]+Q[j+1])
               elif np.round((((states_list[j] % 1)*10) % 1)*10)==3:
                 Q[j] = 0.85*Q[j] + 0.05/2*(Q[j-15] +Q[j+15])+ 0.05/2*(Q[j-3]+Q[j+3]) + 0.05/2*(Q[j-2]+Q[j-1])
             except: #if we can't do a 3 directional smoothing, do a 2 directional smoothing in the direction of external temperature and number of people
               try:
                 if np.round((((states_list[j] % 1)*10) % 1)*10)==1:
                   Q[j] = 0.85*Q[j] +  0.075/2*(Q[j-3]+Q[j+3]) + 0.075/2*(Q[j+1]+Q[j+2])
                 elif np.round((((states_list[j] % 1)*10) % 1)*10)==2:
                   Q[j] = 0.85*Q[j] +  0.075/2*(Q[j-3]+Q[j+3]) + 0.075/2*(Q[j-1]+Q[j+1])
                 elif np.round((((states_list[j] % 1)*10) % 1)*10)==3:
                   Q[j] = 0.85*Q[j] +  0.075/2*(Q[j-3]+Q[j+3]) + 0.075/2*(Q[j-2]+Q[j-1])
                   
               except:# if we can't even do a 2 directional smoothing, do a 1 directional smoothing, in the direction of external temperature
                 if np.round((((states_list[j] % 1)*10) % 1)*10)==1:
                   Q[j] = 0.85*Q[j] +  0.15/2*(Q[j+1]+Q[j+2])
                 elif np.round((((states_list[j] % 1)*10) % 1)*10)==2:
                   Q[j] = 0.85*Q[j] +  0.15/2*(Q[j-1]+Q[j+1])
                 elif np.round((((states_list[j] % 1)*10) % 1)*10)==3:
                   Q[j] = 0.85*Q[j] +  0.15/2*(Q[j-2]+Q[j-1])

      sum_qrow1.append(np.average(Q[58,:]))
      sum_qrow2.append(np.average(Q[59,:]))
      sum_qrow3.append(np.average(Q[21,:]))
      sum_qrow4.append(np.average(Q[22,:]))
      sum_qrow5.append(np.average(Q[44,:]))
      sum_qrow6.append(np.average(Q[45,:]))
      sum_qrow7.append(np.average(Q[32,:]))
      sum_qrow8.append(np.average(Q[33,:]))
      sum_qrow9.append(np.average(Q[6,:]))
      sum_qrow10.append(np.average(Q[7,:]))
      sum_qrow11.append(np.average(Q[65,:]))
      sum_qrow12.append(np.average(Q[66,:]))
      
  return Q,i,alpha,alpha_0,sum_qrow1,sum_qrow2,sum_qrow3,sum_qrow4,sum_qrow5,sum_qrow6,sum_qrow7,sum_qrow8,sum_qrow9,sum_qrow10,sum_qrow11,sum_qrow12,num_iterations

def test_opt_policy(Q,R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions):
  raw_reward = 0
  optimal_reward = 0
  energy_raw_cost = 0
  energy_optimal_cost = 0
  comfort_raw_cost = 0
  comfort_optimal_cost = 0
  
  initial_state = raw_states_matrix_complete[0]
  temp1 = display_optimal_policy(Q,trans_mat_array,states_list_dict[round(initial_state,2)],actions)
  optimal_reward+= R[initial_state,temp1[0]-22]
  energy_raw_cost += Re[initial_state,raw_action_set[0]-22]
  energy_optimal_cost += Re[initial_state,temp1[0]-22]
  comfort_raw_cost += Rc[initial_state,raw_action_set[0]-22]
  comfort_optimal_cost +=  Rc[initial_state,temp1[0]-22]
  raw_reward+=R[states_list_dict[round(initial_state,2)],raw_action_set[0]-22]
  optimal_state= temp1[1]
  for i in xrange(1,len(raw_states_matrix_complete)):
    state = states_list_dict[round(raw_states_matrix_complete[i],2)]
    action = raw_action_set[i]
    raw_reward += R[state,action-22]
    temp = display_optimal_policy(Q,trans_mat_array,optimal_state,actions)
    act_optimal_action = temp[0]
    energy_raw_cost += Re[state,action-22]
    energy_optimal_cost += Re[optimal_state,act_optimal_action-22]
    comfort_raw_cost += Rc[state,action-22]
    comfort_optimal_cost +=  Rc[optimal_state,act_optimal_action-22]
    optimal_reward+=R[optimal_state,act_optimal_action-22]
    optimal_state = temp[1]
    
  return raw_reward,optimal_reward,energy_raw_cost,energy_optimal_cost,comfort_raw_cost,comfort_optimal_cost
  
def generate_cost_comparison(R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions):
  raw_reward_list = []
  optimal_reward_list = []
  energy_raw_cost_list = []
  energy_optimal_cost_list = []
  comfort_raw_cost_list = []
  comfort_optimal_cost_list = []
  for k in xrange(5):
    print "k= ", k
    Q_tuple = train_Q(R,alpha_0=1.0,k_increment=0.1,num_iterations=200000)
    Q = Q_tuple[0]
    Q = Q.astype(int)
    costs = test_opt_policy(Q,R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions)
    raw_reward_list.append(costs[0])
    optimal_reward_list.append(costs[1])
    energy_raw_cost_list.append(costs[2])
    energy_optimal_cost_list.append(costs[3])
    comfort_raw_cost_list.append(costs[4])
    comfort_optimal_cost_list.append(costs[5])
  return raw_reward_list,optimal_reward_list,energy_raw_cost_list,energy_optimal_cost_list,comfort_raw_cost_list,comfort_optimal_cost_list

def compare_energy_costs(R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions):
  energy_raw_cost_oldmodel = 0
  energy_raw_cost_newmodel = 0
  initial_state = raw_states_matrix_complete[0]
  energy_raw_cost_oldmodel += Re[initial_state,raw_action_set[0]-22]
  energy_raw_cost_newmodel += Re_newmodel[initial_state,raw_action_set[0]-22]

  for i in xrange(1,len(raw_states_matrix_complete)):
    state = states_list_dict[round(raw_states_matrix_complete[i],2)]
    action = raw_action_set[i]
    energy_raw_cost_oldmodel += Re[state,action-22]
    energy_raw_cost_newmodel += Re_newmodel[state,action-22]

  return energy_raw_cost_oldmodel, energy_raw_cost_newmodel 

def display_optimal_policy(Q, trans_mat_array, test_state, actions):
  optimal_action = np.argmin(Q[test_state])
  possible_next_states = []
  next_state_prob = [] #the probabilities associated with each possible next state
  counter = 0
  
  while trans_mat_array[optimal_action][test_state].any() == np.zeros_like(trans_mat_array[optimal_action][test_state]).any():
      #current_state = counter
      action_alt = actions[counter]#if the chosen action doesn't cause a state transition, choose a new action 
      optimal_action = int(action_alt - 22)
      counter+=1
      if counter == len(R[test_state]):#if none of the actions have a transition from current_state, choose another current_state!
              test_state = np.random.randint(0, len(Q[:,1]))#pick a new current_state!
              counter = 0
              print 'the current state must be changed to:',  test_state
  '''this block populates the next possible states by going through the transition matrix for the given setpoint. All occuring next states are appended into an array, along with their corresponding probabilities'''
  for i in xrange(len(trans_mat_array[optimal_action][test_state])):
        if trans_mat_array[optimal_action][test_state][i] != 0:
            possible_next_states.append(states_list[i])
            next_state_prob.append(trans_mat_array[optimal_action][test_state][i])
        else:
          pass
   
  '''next state chosen by generating a uniformly distributed random variable, with probabilities proportional to the probabilities given by the transition matrix'''
  next_state = np.random.choice(possible_next_states, p=next_state_prob)
  next_state = np.round_(next_state,2)
  next_state_norm = states_list_dict[next_state]
  act_optimal_action = optimal_action + 22
  return act_optimal_action, next_state_norm
  
'''this block displays the optimal policy over a sequence of states and actions'''

#costs = compare_energy_costs(R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions)

#if __name__ == "__main__":

R = compute_rewards(states_list,actions,avg_num_ppl,comfort_weight_raw=298)
'''
cost_set = generate_cost_comparison(R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions)
training_periods = range(5)
plt.plot(training_periods,cost_set[2])
plt.plot(training_periods,cost_set[3])
plt.plot(training_periods,cost_set[4])
plt.plot(training_periods,cost_set[5])
plt.xticks(range(len(training_periods)))
plt.xlabel('Experiment #')
plt.ylabel('Cost')
plt.legend(['Current energy cost','Optimal energy cost','Current discomfort cost','Optimal discomfort cost'], loc='best')
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/cost_comparisons/all_component_costs"+".png")
plt.gcf().clear()

total_reward_raw = np.array(cost_set[0])
total_reward_optimal = np.array(cost_set[1])
total_reward_percentage = (total_reward_optimal/total_reward_raw)*100

fig, ax1 = plt.subplots(figsize=(10,8))#can change the actual size of the plot (inches)
t = range(len(total_reward_raw))
ax1.plot(t, total_reward_raw, 'b')
ax1.plot(t, total_reward_optimal)
ax1.set_xlabel('Experiment number')
plt.legend(['Total current cost','Total optimal cost'], loc='best')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Total cost', color='b')
ax1.tick_params('y', colors='b')
#plt.yticks(states_list) #specifies the step for the y axis 
ax2 = ax1.twinx()
ax2.plot(t, total_reward_percentage, 'r')
ax2.set_ylabel('Optimal policy as % of current policy', color='r')
ax2.tick_params('y', colors='r')
plt.yticks(range(0,101,11))
plt.xticks(range(len(total_reward_percentage)))#specify that the x axis should be scaled by integer values
#plt.yticks([22,23,24,25,26])
fig.tight_layout()
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/cost_comparisons/total_costs_withpercent"+".png")
plt.gcf().clear()
'''

Q_tuple = train_Q(R,alpha_0=1.0,k_increment=0.1,num_iterations=200000)
Q = Q_tuple[0]
Q = Q.astype(int)
alpha_final = Q_tuple[2]
alpha_init = Q_tuple[3]
terminal_i = Q_tuple[1]
num_iterations = Q_tuple[16]
print 'terminating i:', terminal_i
'''
#costs = test_opt_policy(Q,R,raw_states_matrix_complete,raw_action_set,states_list_dict,states_list,trans_mat_array,actions)
#Q = (Q/np.max(Q)*100)
#sum_qrow1 = np.absolute(sum_qrow1)
x_axis = np.linspace(0,terminal_i,terminal_i+2)
#putting 2 consecutive lines of plt.plot, plots both lines on the same graph. plt.show separates two different graphs
plt.plot(x_axis,Q_tuple[4])
plt.plot(x_axis, Q_tuple[5])
plt.plot(x_axis,Q_tuple[12])
plt.plot(x_axis, Q_tuple[13])
#plt.savefig("C:/Users/User/.spyder/thesis_plots/fig1.png")
plt.legend(['State 58', 'State 59','State 6','State 7'], loc='best')
plt.xlabel('Iterations')
plt.ylabel('Average Q-value')
plt.savefig("C:/Users/User/Thesis/newq_extended_data_new_energy_hvac2/convergence_"+"iterations="+str(num_iterations)+"_alpha0="+str(alpha_init)+"k_increment=0.1"+".png")
plt.show()
plt.gcf().clear()
'''

test_state = 18
state_reached = []
action_taken = []
state_reached.append(test_state)

for j in xrange(1000):
  act_optimal_action = display_optimal_policy(Q,trans_mat_array,test_state,actions)[0]
  next_state_norm = display_optimal_policy(Q,trans_mat_array,test_state,actions)[1]
  state_reached.append(next_state_norm)
  action_taken.append(act_optimal_action)
  test_state = next_state_norm
  
state_reached = np.array(state_reached)
action_taken = np.array(action_taken)  

fullstate_reached = np.zeros_like(state_reached,dtype=np.float64)
for k in xrange(len(state_reached)):
    fullstate_reached[k] =states_list[state_reached[k]]
x_axis2 = np.linspace(0,j+1,j+1)
#-----------------------#############---------------------------#
'''this block creates a plot with 2 y-axes on a single plot, showing the states reached
and the actions taken'''

fig, ax1 = plt.subplots(figsize=(10,8))#can change the actual size of the plot (inches)
t = x_axis2
ax1.plot(t, fullstate_reached[:j+1], 'b.')
ax1.set_xlabel('time step')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('current state', color='b')
ax1.tick_params('y', colors='b')
plt.yticks('') #specifies the step for the y axis 
ax2 = ax1.twinx()
ax2.plot(t, action_taken, 'r.')
ax2.set_ylabel('action taken', color='r')
ax2.tick_params('y', colors='r')
#plt.xticks(range(len(t)))#specify that the x axis should be scaled by integer values
plt.yticks([22,23,24,25,26])
fig.tight_layout()
plt.show()

#-------------------------#############--------------------------#

'''
plt.plot(x_axis,sum_qrow3)
plt.plot(x_axis, sum_qrow4)
plt.plot(x_axis,sum_qrow11)
plt.plot(x_axis, sum_qrow12)
plt.legend(['avg. state 21', 'avg state 22','avg state 65','avg state 66'], loc='best')
plt.show()

plt.plot(x_axis,sum_qrow5)
plt.plot(x_axis, sum_qrow6)
plt.legend(['avg. state 44', 'avg state 45'], loc='best')
plt.show()
plt.plot(x_axis,sum_qrow7)
plt.plot(x_axis, sum_qrow8)
plt.legend(['avg. state 2', 'avg state 3'], loc='best')
plt.show()
'''

stop = timeit.default_timer()

print 'runtime:', stop - start

