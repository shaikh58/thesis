# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:39:16 2016
Rev- 25 Jan 2017
@author: Mustafa

To run on Bash for Windows, first type 'export KMP_AFFINITY=disabled' before running
"""

import numpy as np
import pandas as pd
from collections import Counter
import sys
import co2_estimation
from decimal import getcontext, Decimal
import timeit
import matplotlib.pyplot as plt

start = timeit.default_timer()

def read_co2_model(co2_int):
    return co2_estimation.calculate_pop(co2_int)
     

def import_database():
    #create a pandas dataframe object from which to extract data
    #return pd.read_excel('Database2.xlsx', sheetname = 'Database_C', skiprows = [0,1,2,3,4])
    return pd.read_excel('Database2_extended.xlsx', sheetname = 'Database_C', skiprows = [0,1,2,3,4])
    
df = import_database()   

def read_excel_data():
    #create a pandas dataframe object from which to extract data
    #df = pd.read_excel('Database2.xlsx', sheetname = 'Database_C', skiprows = [0,1,2,3,4])
    #df = co2_estimation.calculate_pop()[1]
    #setpt_temp = np.round(df['HVAC1_Tsetpoint'].values)
    #int_temp1raw = df['HVAC1_Temp1 (Cashiers area)'].values
    #int_temp2raw = df['HVAC1 Temp2 (Central area)'].values
    setpt_temp = np.round(df['HVAC2_Tsetpoint'].values)
    int_temp1raw = df['HVAC2_Temp1  (Cashiers area)'].values
    int_temp2raw = df['HVAC2 Temp2 (Central area)'].values
    int_temp_raw = np.round_((int_temp1raw + int_temp2raw)/2)
    ext_temp_raw = df['T_ext'].values
    hu_ext = df['HU_ext'].values
    co2_int = df['HVAC 2 CO2'].values
    #co2_int = df['HVAC 1 CO2'].values
    num_ppl_raw = read_co2_model(co2_int)
    avg_num_ppl = np.average(num_ppl_raw)
    return setpt_temp, int_temp_raw, num_ppl_raw, ext_temp_raw, hu_ext, co2_int, avg_num_ppl

def compute_raw_states_matrix_alt(setpt_temp, int_temp_raw,setpt):
    setpt_temp = np.array(setpt_temp) 
    count = (setpt_temp==setpt).sum()
    c=0
    #print int_temp_raw[369]
    int_temp_set= np.zeros(count)
    modified_setpt22 = 0
    for i in xrange(len(setpt_temp)):
        if round(setpt_temp[i]) <22:
            setpt_temp[i] = 22
            modified_setpt22 += 1
            int_temp_set = np.append(int_temp_set,0)
        elif setpt_temp[i] > 26:
            setpt_temp[i] = 26
            int_temp_set = np.append(int_temp_set,0)
        if setpt_temp[i] == setpt: 
            int_temp_set[c] = int_temp_raw[i]
            c+=1
    for i in xrange(len(int_temp_set)):
        if int_temp_set[i]<22:
          int_temp_set[i] = 22
        elif int_temp_set[i]>26:
          int_temp_set[i] = 26        
             
    raw_states_matrix_alt = int_temp_set
    return raw_states_matrix_alt
    
def compute_raw_states_matrix(setpt_temp, int_temp_raw, num_ppl_raw, ext_temp_raw, setpt):
  
    setpt_temp = np.array(setpt_temp) 
    count = (setpt_temp==setpt).sum()
    
    int_temp_set= np.zeros(count)
    ext_temp_set = np.zeros(count)
    num_ppl_set = np.zeros(count)
    c=0
    '''currently, setpoint is constrained from 22-26. If we want to take 27
    as a valid set point, we will need to increase the int_temp states to include
    27 as well'''
    modified_setpt22 = 0
    for i in xrange(len(setpt_temp)):
        if round(setpt_temp[i]) <22:
            setpt_temp[i] = 22
            modified_setpt22 += 1
            int_temp_set = np.append(int_temp_set,0)
            ext_temp_set = np.append(ext_temp_set,0)
            num_ppl_set = np.append(num_ppl_set,0)
        elif setpt_temp[i] > 26:
            setpt_temp[i] = 26
            int_temp_set = np.append(int_temp_set,0)
            ext_temp_set = np.append(ext_temp_set,0)
            num_ppl_set = np.append(num_ppl_set,0)
        if setpt_temp[i] == setpt: 
            int_temp_set[c] = int_temp_raw[i]
            ext_temp_set[c] = ext_temp_raw[i]
            num_ppl_set[c] = num_ppl_raw[i]
            c+=1
        
    num_ppl_discset = np.zeros(len(num_ppl_set))
    ext_temp_discset = np.zeros(len(ext_temp_set))
    for i in range(len(num_ppl_set)):
        if isinstance(num_ppl_set[i],float or int) and isinstance(ext_temp_set[i], float or int):
            if num_ppl_set[i]<7 :
                num_ppl_discset[i] = 0.1
            elif 7<=num_ppl_set[i]<13:
                num_ppl_discset[i] = 0.2
            elif 13<=num_ppl_set[i]<19:
                num_ppl_discset[i] = 0.3
            elif 19<=num_ppl_set[i]<25:
                num_ppl_discset[i] = 0.4
            elif num_ppl_set[i]>=25:
                num_ppl_discset[i] = 0.5
            if ext_temp_set[i] <22:
                ext_temp_discset[i] = 0.01
            elif 22<= ext_temp_set[i]<=26:
                ext_temp_discset[i] = 0.02
            elif ext_temp_set[i] >26:
                ext_temp_discset[i] = 0.03
            if int_temp_set[i]<22:
                int_temp_set[i] = 22
            elif int_temp_set[i]>26:
                int_temp_set[i] = 26
        else:
            sys.exit("Program terminated due to invalid data type")
    raw_states_matrix = num_ppl_discset + int_temp_set + ext_temp_discset
    return raw_states_matrix
    
def compute_raw_states_matrix_total(setpt_temp, int_temp_raw, num_ppl_raw, ext_temp_raw):
  
    count = len(setpt_temp)
    
    int_temp_set= np.zeros(count)
    ext_temp_set = np.zeros(count)
    num_ppl_set = np.zeros(count)

    '''currently, setpoint is constrained from 22-26. If we want to take 27
    as a valid set point, we will need to increase the int_temp states to include
    27 as well'''
    for i in xrange(len(setpt_temp)):
        if round(setpt_temp[i]) <22:
            setpt_temp[i] = 22
        elif setpt_temp[i] > 26:
            setpt_temp[i] = 26

        int_temp_set[i] = int_temp_raw[i]
        ext_temp_set[i] = ext_temp_raw[i]
        num_ppl_set[i] = num_ppl_raw[i]

        
    num_ppl_discset = np.zeros(len(num_ppl_set))
    ext_temp_discset = np.zeros(len(ext_temp_set))
    for i in range(len(num_ppl_set)):
        if isinstance(num_ppl_set[i],float or int) and isinstance(ext_temp_set[i], float or int):
            if num_ppl_set[i]<7 :
                num_ppl_discset[i] = 0.1
            elif 7<=num_ppl_set[i]<13:
                num_ppl_discset[i] = 0.2
            elif 13<=num_ppl_set[i]<19:
                num_ppl_discset[i] = 0.3
            elif 19<=num_ppl_set[i]<25:
                num_ppl_discset[i] = 0.4
            elif num_ppl_set[i]>=25:
                num_ppl_discset[i] = 0.5
            if ext_temp_set[i] <22:
                ext_temp_discset[i] = 0.01
            elif 22<= ext_temp_set[i]<=26:
                ext_temp_discset[i] = 0.02
            elif ext_temp_set[i] >26:
                ext_temp_discset[i] = 0.03
            if int_temp_set[i]<22:
                int_temp_set[i] = 22
            elif int_temp_set[i]>26:
                int_temp_set[i] = 26
        else:
            sys.exit("Program terminated due to invalid data type")
    raw_states_matrix = num_ppl_discset + int_temp_set + ext_temp_discset
    return raw_states_matrix, setpt_temp
    
def normalize_data_alt(raw_states_matrix):
    int_temp_defn = [22,23,24,25,26]
    states_list_alt = range(len(int_temp_defn))
    states_list_dict_alt = {22:0,23:1,24:2,25:3,26:4}
    observed_states_disc_alt = np.zeros(len(raw_states_matrix))
    for i in xrange(len(raw_states_matrix)):
        raw_states_matrix[i] = Decimal(raw_states_matrix[i])/1
        observed_states_disc_alt[i] = states_list_dict_alt[raw_states_matrix[i]]
    return observed_states_disc_alt,states_list_alt,states_list_dict_alt
    
def normalize_data(raw_states_matrix):
  states_list = np.zeros(75)
  getcontext().prec = 4

  int_temp_defn = [22, 23, 24, 25, 26]
  num_ppl_defn = [0.1, 0.2, 0.3, 0.4, 0.5]
  ext_temp_defn = [0.01, 0.02, 0.03]
  #ext_temp_dict = {0.01:18, 0.02:24, 0.03:28}
  states_list_dict = {}
  a=0
  for i in range(len(int_temp_defn)):
      for j in range(len(num_ppl_defn)):
          for k in range(len(ext_temp_defn)):
              states_list[a] = int_temp_defn[i] + num_ppl_defn[j] + ext_temp_defn[k]
              states_list[a] = Decimal(states_list[a])/1
              #states_list[a] = np.round(states_list[a], decimals = 2)
              states_list[a] = Decimal(states_list[a])/1
              states_list_dict.update({states_list[a]: a})#this is a dictionary containing key value pairs of discretized state: original state e.g. {22.11:1, etc.}
              a+=1

  '''can also use a hash table'''
  observed_states_disc = np.zeros(len(raw_states_matrix))
  for i in xrange(len(raw_states_matrix)):
      raw_states_matrix[i] = Decimal(raw_states_matrix[i])/1
      observed_states_disc[i] = states_list_dict[raw_states_matrix[i]]#makes use of O(1) lookup time of dict rather than iterating over states list and raw states vector
  return observed_states_disc,states_list,states_list_dict
  
def get_trans_matrix_alt(raw_states_matrix_alt):
    normalized_data_alt = normalize_data_alt(raw_states_matrix_alt)
    states_list_alt = normalized_data_alt[1]
    observed_states_disc_alt = normalized_data_alt[0]
    states_list_dict_alt = normalized_data_alt[2]
    transition_matrix = np.zeros((len(states_list_alt),len(states_list_alt)))
    for (x,y), c in Counter(zip(observed_states_disc_alt, observed_states_disc_alt[1:])).iteritems():
        transition_matrix[x,y] = c
    #this block normalizes the rows to sum to 1
    trans_prob_matrix = transition_matrix
    for i in xrange(len(transition_matrix)):
        row_sum = np.sum(trans_prob_matrix[i])
        for j in xrange(len(transition_matrix)):
            if trans_prob_matrix[i,j] == 0:
                trans_prob_matrix[i,j] = 0
            else:
                trans_prob_matrix[i,j] = trans_prob_matrix[i,j]/row_sum
    counter=0
    #print 'zero rows:'
    for i in range(len(trans_prob_matrix)):
            if trans_prob_matrix[:,i].any() == np.zeros_like(trans_prob_matrix[:,i]).any():
                counter+=1
    return trans_prob_matrix, states_list_alt, states_list_dict_alt
    
def get_trans_matrix(raw_states_matrix):
    #this block constructs the transition matrix, before normalization
    # of the rows
    normalized_data = normalize_data(raw_states_matrix)
    states_list = normalized_data[1]
    observed_states_disc = normalized_data[0]
    states_list_dict = normalized_data[2]
    transition_matrix = np.zeros((len(states_list),len(states_list)))
    for (x,y), c in Counter(zip(observed_states_disc, observed_states_disc[1:])).iteritems():
        transition_matrix[x,y] = c

    #this block normalizes the rows to sum to 1
    trans_prob_matrix = transition_matrix
    for i in xrange(len(transition_matrix)):
        row_sum = np.sum(trans_prob_matrix[i])
        for j in xrange(len(transition_matrix)):
            if trans_prob_matrix[i,j] == 0:
                trans_prob_matrix[i,j] = 0
            else:
                trans_prob_matrix[i,j] = trans_prob_matrix[i,j]/row_sum

    #this block prints each state that the system has been in at some point by
    # printing all non zero rows of the transition probability matrix
    counter=0
    '''
    print 'zero rows:'
    for i in range(len(trans_prob_matrix)):
            if trans_prob_matrix[:,i].any() == np.zeros_like(trans_prob_matrix[:,i]).any():
                #print states_list[i]
                counter+=1
    print counter
    '''
    return trans_prob_matrix, states_list, states_list_dict


def TDlearning_modelfree(raw_states_matrix):
  normalized_data = normalize_data(raw_states_matrix)
  states_list = normalized_data[1]
  observed_states_disc = normalized_data[0]
  states_list_dict = normalized_data[2]
  
  
    
    
def checksum(trans_prob_matrix):
    #this function checks to see if all the rows sum to 1
    for i in range(len(trans_prob_matrix)):
        checksum = np.sum(trans_prob_matrix[i])
        if checksum != 1 and checksum != 0:
            print checksum

def ret_trans_matrix_alt(setpt22,setpt23,setpt24, setpt25, setpt26):
    excel_data = read_excel_data()
    trans_mat22 = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt22))[0]
    trans_mat23 = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt23))[0]
    trans_mat24 = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt24))[0]
    trans_mat25 = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt25))[0]
    trans_mat26 = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt26))[0]
    states_list_alt = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt26))[1]
    states_list_dict_alt = get_trans_matrix_alt(compute_raw_states_matrix_alt(excel_data[0], excel_data[1],setpt26))[2]
    return [trans_mat22, trans_mat23,trans_mat24, trans_mat25, trans_mat26], excel_data, [states_list_alt, states_list_dict_alt]

def ret_trans_matrix(setpt22, setpt23, setpt24, setpt25, setpt26):
    #this function returns the transition probability matrix with the given inputs,
    #for ease of access in the q learning algorithm which calls this function
    excel_data = read_excel_data()
    raw_states_matrix_total = compute_raw_states_matrix_total(excel_data[0], excel_data[1], excel_data[2], excel_data[3])
    trans_mat22 = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt22))[0]
    #trans_mat22[0][56]=0
    trans_mat23 = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt23))[0]
    
    trans_mat24 = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt24))[0]
    trans_mat25 = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt25))[0]
    trans_mat26 = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt26))[0]
    #trans_mat26[0][56] = 0
    states_list = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt26))[1]
    states_list_dict = get_trans_matrix(compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],setpt26))[2]
    num_ppl_avg = excel_data[6]
    

    return [trans_mat22, trans_mat23,trans_mat24, trans_mat25, trans_mat26], excel_data, [states_list, states_list_dict], num_ppl_avg,raw_states_matrix_total

    #return trans_mat22
    
if __name__ == "__main__":
    #excel_data = read_excel_data()
    #raw_states_matrix = compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2], excel_data[3],22)
    ret_trans_mat = ret_trans_matrix(22,23,24,25,26)
    #trans_mats = ret_trans_mat[0]
    #excel_data = ret_trans_mat[1]
    #states_list_dict = ret_trans_mat[2][1]
    #raw_states_matrix_24 = compute_raw_states_matrix(excel_data[0], excel_data[1], excel_data[2],                excel_data[3],24)
    #ret_trans_mat_alt = ret_trans_matrix_alt(22,23,24,25,26)
    '''
    #trans_mat22 = ret_trans_mat[0][4]
    trans_mat22 = ret_trans_mat_alt[0][4]
    transmat_vis = plt.pcolor(trans_mat22,cmap='Reds')
    plt.xticks(range(5),('22','23','24','25','26'))
    plt.yticks(range(5),('22','23','24','25','26'))
    plt.xlabel('To internal temperature')
    plt.ylabel('From internal temperature')
    #plt.xlabel('States')
    #plt.ylabel('States')
    plt.colorbar(transmat_vis)
    plt.show() 
    '''

    stop = timeit.default_timer()
    print 'runtime:', stop - start
    
