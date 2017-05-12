# thesis
Python code from my undergraduate thesis 

The following is the abstract of the paper that I, along with my team, published at REMOO 2017. It gives a high level overview of the project and the results. This respository only contains the code that I developed as part of the project.

'This paper presents a study in which a retail store aims to reduce the energy cost of its heating, ventilating and air conditioning (HVAC) systems, while providing a comfortable shopping environment to its customers. Various models which describe the underlying physical systems and processes are employed in combination with an optimization problem in order to determine the optimal policy for the store’s HVAC system operation.  The optimization problem was formulated as a multi-agent Markov decision process, in which each agent controls its own HVAC such that the system-wide cost can be optimized. The solution is to provide a complete policy by which the store can determine the temperature set-point of the three HVAC systems in response to observing various system states involving the external and internal temperatures, and the estimated occupancy level. The complexity of the problem is high and therefore a variant of a well known reinforcement learning algorithm is used to make effective use of available historical data and solve the optimization task. To reduce the complexity further, we devised a threshold-based state space reduction method by partitioning the temperature, humidity and population range into intervals. We present numerical studies and visualizations to demonstrate the energy savings and increased comfort levels that can be achieved by the optimal control policy found by the proposed algorithm.'



File directory: 

'trans_mat_single_agent': computes the transition probability matrices which are used in the training of the Q-learning algorithm which will in turn contain information about the optimal policy

'q_learning_algorithm': trains the Q-learning algorithm to determine the optimal policy for the retail store's HVAC system temperature setpoints
