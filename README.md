# thesis

Summary of project: A reinforcement learning approach to determine the optimal temperature setpoint policy for a retail store, in order to reduce energy costs and maintain customer comfort


This repository contains my final thesis document, the paper that was submitted and presented at the REMOO2017 conference, as well as the Python code that I wrote as part of the project. Note that the paper we published does not have the latest results - further work yielded better results, which can be found in my actual thesis document. These results will be presented at the conference in May, but will not appear in the publication. 


File directory: 

'trans_mat_single_agent': computes the transition probability matrices which are used in the training of the Q-learning algorithm which will in turn contain information about the optimal policy

'q_learning_algorithm': trains the Q-learning algorithm to determine the optimal policy for the retail store's HVAC system temperature setpoints

'vizualizations': generates various visualizations of the optimal policy. See my thesis document results section to see what these look like

'fit_function': contains code to populate matrices with optimal policy information for all possible states. These are then used in 'visualizations.py' to generate heat maps that aid in viewing the behaviour of the optimal policy

'comfort_index': human thermal comfort model (Fanger, 1970). This file determines the thermal comfort level based on various environmental information. This comfort level forms part of the cost function which is to be minimized by the learning algorithm

'co2_estimation': occupancy estimation model based on monitored CO2 levels in the retail store

