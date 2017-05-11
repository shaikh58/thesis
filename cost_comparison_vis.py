# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:15:16 2017

@author: Mustafa
"""
import numpy as np
import matplotlib.pyplot as plt

total_cost_raw = [1330220,1330220,1330220,1330220,1330220,1330220]
total_cost_optimal = [	1069492,	607962,	1040177,	1190305,	1086348,	781793]
optimal_cost_percent = [80,46,78,89,82,59]
energy_cost_optimal = [-124654,	-73609,	-186124,	-498725,	-61914,	-385666]
comfort_cost_optimal= [1194138,	681588,	1226273,	1689049,	1148256, 	1167457]
energy_cost_raw = [-445351,-445351,-445351,-445351,-445351,-445351]
comfort_cost_raw = [1775633,1775633,1775633,1775633,1775633,1775633]
total_cost_raw_byweight = [1330220,1007376,684528,361695,38855,-283989]
total_cost_optimal_byweight = [962679,515545.3333	,199264,	266317.6667	,-196631,	-401503]
discomfort_cost_raw_byweight = [1775633,1452791,	1129948	,807106,	484263	,161421]
energy_cost_raw_byweight = [-445351,-445351,	-445351	,-445351,	-445351,	-445351]
discomfort_cost_optimal_byweight = [1184460, 939767.3333,	645913,	635985.6667,	366690.6667,	115480.6667]
energy_cost_optimal_byweight = [-221782, -424229.6667,	-446646,	-369682,	-563337.6667,	-516990.6667]
comfort_weights = [275,225,175,125,75,25]
training_periods = range(6) 
"""
fig, ax1 = plt.subplots(figsize=(10,8))#can change the actual size of the plot (inches)
t = range(len(total_cost_raw))
ax1.plot(t, total_cost_raw, 'b')
ax1.plot(t, total_cost_optimal)
ax1.set_xlabel('Experiment number')
plt.legend(['Total current cost','Total optimal cost'], loc='upper left')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Total cost', color='b')
ax1.tick_params('y', colors='b')
#plt.yticks(states_list) #specifies the step for the y axis 
ax2 = ax1.twinx()
ax2.plot(t, optimal_cost_percent, 'r')
ax2.set_ylabel('Optimal policy as % of current policy', color='r')
ax2.tick_params('y', colors='r')
plt.xticks(range(len(total_cost_raw)))#specify that the x axis should be scaled by integer values
#plt.yticks([22,23,24,25,26])
fig.tight_layout()
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/total_costs_withpercent"+".png")
plt.gcf().clear()

plt.plot(training_periods, total_cost_raw)
plt.plot(training_periods, total_cost_optimal)
plt.xlabel('Training episode #')
plt.ylabel('Cost')
plt.legend(['Total current cost','Total optimal cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/total_costs"+".png")
plt.gcf().clear()

plt.plot(training_periods,energy_cost_raw)
plt.plot(training_periods,energy_cost_optimal)
plt.xlabel('Training episode #')
plt.ylabel('Cost')
plt.legend(['Current energy cost','Optimal energy cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/energy_costs"+".png")
plt.gcf().clear()

plt.plot(training_periods,comfort_cost_raw)
plt.plot(training_periods,comfort_cost_optimal)
plt.xticks(range(len(training_periods)))
plt.xlabel('Training episode #')
plt.ylabel('Cost')
plt.legend(['Current discomfort cost','Optimal discomfort cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/discomfort_costs"+".png")
plt.gcf().clear()
"""
plt.plot(training_periods,energy_cost_raw)
plt.plot(training_periods,energy_cost_optimal)
plt.plot(training_periods,comfort_cost_raw)
plt.plot(training_periods,comfort_cost_optimal)
plt.xticks(range(len(training_periods)))
plt.xlabel('Training episode #')
plt.ylabel('Cost')
plt.legend(['Current energy cost','Optimal energy cost','Current discomfort cost','Optimal discomfort cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/all_costs"+".png")
plt.gcf().clear()

#################################################################################################
"""
plt.plot(comfort_weights,energy_cost_raw_byweight)
plt.plot(comfort_weights,energy_cost_optimal_byweight)
plt.xlabel('Discomfort penalty weight')
plt.ylabel('Cost')
plt.legend(['Current energy cost','Optimal energy cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/energy_costs_byweights"+".png")
plt.gcf().clear()

plt.plot(comfort_weights,discomfort_cost_raw_byweight)
plt.plot(comfort_weights,discomfort_cost_optimal_byweight)
plt.xlabel('Discomfort penalty weight')
plt.ylabel('Cost')
plt.legend(['Current discomfort cost','Optimal discomfort cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/discomfort_costs_byweights"+".png")
plt.gcf().clear

plt.plot(comfort_weights,energy_cost_raw_byweight)
plt.plot(comfort_weights,energy_cost_optimal_byweight)
plt.plot(comfort_weights,discomfort_cost_raw_byweight)
plt.plot(comfort_weights,discomfort_cost_optimal_byweight)
plt.xlabel('Discomfort penalty weight')
plt.ylabel('Cost')
plt.legend(['Current energy cost','Optimal energy cost','Current discomfort cost','Optimal discomfort cost'], loc='upper left')
plt.savefig("C:/Users/User/Thesis/hvac2_smoothing_epsilonstop/extended_data/cost_comparisons/all_costs_byweights"+".png")
plt.gcf().clear()
"""