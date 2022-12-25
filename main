"""
LIBRARY IMPORTATION
"""

import random
import numpy as np # v 1.22.3
import math
import matplotlib.pyplot as plt # v 3.5.1

"""
GREEDY ALGORITHM IMPLEMENTATION
"""

class GreedyAlgorithm:

    def __init__ (self, K, alpha, beta): # Initialise variables
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.theta = [0.0 for _ in range(K)]

    def estimate_model (self): # For GA model estimate is expectation given dist params
        for k in range(self.K):
            self.theta[k] = self.alpha[k] / (self.alpha[k] + self.beta[k])

    def select_and_apply_action (self): # (Randomly) choose action that maximises expected reward
        max_theta = max(self.theta)
        max_theta_indices = [i for i, x in enumerate(self.theta) if math.isclose(x, max_theta)]
        x_t = random.choice(max_theta_indices)
        return x_t

    def reward (self, p): # Reward is 1 with prob=p and 0 with prob=1-p
        return random.choices([0, 1], weights=[1 - p, p])[0]

    def update_distribution (self, x_t, r_t): # Alpha increases if success, beta increases if failure
        self.alpha[x_t] += r_t
        self.beta[x_t] += 1 - r_t

"""
THOMPSON SAMPLING IMPLEMENTATION
"""

class ThompsonSampling:

    def __init__ (self, K, alpha, beta): # Initialise variables
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.theta = [0.0 for _ in range(K)]

    def estimate_model (self): # For TS model estimate is sampling from dist
        for k in range(self.K):
            self.theta[k] = np.random.beta(self.alpha[k], self.beta[k])

    def select_and_apply_action (self): # (Randomly) choose action that maximises expected reward
        max_theta = max(self.theta)
        max_theta_indices = [i for i, x in enumerate(self.theta) if math.isclose(x, max_theta)]
        x_t = random.choice(max_theta_indices)
        return x_t

    def reward (self, p): # Reward is 1 with prob=p and 0 with prob=1-p
        return random.choices([0, 1], weights=[1 - p, p])[0]

    def update_distribution (self, x_t, r_t): # Alpha increases if success, beta increases if failure
        self.alpha[x_t] += r_t
        self.beta[x_t] += 1 - r_t

"""
SIMULATIONS
"""

def simulate_and_store_actions_GA (K=3, mean_rewards=[0.9, 0.8, 0.7], n_iterations=1000, n_simulations=10000):

    initial_probability = 1 / K # Eliminate bias
    
    total_action_history = [0] * n_simulations # Stores actions in every iteration and simulation
    
    for simulation in range(n_simulations):
    
        alpha = [initial_probability for _ in range(K)] # Eliminate bias
        beta = [(1 - initial_probability) for _ in range(K)] # Eliminate bias
        model = GreedyAlgorithm(K, alpha, beta) # Model set to GA
        
        action_history = [0] * n_iterations # Stores actions in every iteration
        
        for t in range(n_iterations):
        
            model.estimate_model() # Obtain theta values
            x_t = model.select_and_apply_action() # Choose action
            action_history[t] = x_t # Store the chosen action
            p = mean_rewards[x_t] # Probability of reward for chosen action
            r_t = model.reward(p) # Apply action, observe reward
            model.update_distribution(x_t, r_t) # Update dist params
            
        total_action_history[simulation] = action_history
        
    return total_action_history

def simulate_and_store_actions_TS(K=3, mean_rewards=[0.9, 0.8, 0.7], n_iterations=1000, n_simulations=10000):

    initial_probability = 1 / K # Eliminate bias
    
    total_action_history = [0] * n_simulations # Stores actions in every iteration and simulation
    
    for simulation in range(n_simulations):
    
        alpha = [initial_probability for _ in range(K)] # Eliminate bias
        beta = [(1 - initial_probability) for _ in range(K)] # Eliminate bias
        model = ThompsonSampling(K, alpha, beta) # Model set to GA
        
        action_history = [0] * n_iterations # Stores actions in every iteration
        
        for t in range(n_iterations):
        
            model.estimate_model() # Obtain theta values
            x_t = model.select_and_apply_action() # Choose action
            action_history[t] = x_t # Store the chosen action
            p = mean_rewards[x_t] # Probability of reward for chosen action
            r_t = model.reward(p) # Apply action, observe reward
            model.update_distribution(x_t, r_t) # Update dist params
            
        total_action_history[simulation] = action_history
        
    return total_action_history

def obtain_probabilities(total_action_history):

    total_action_history = list(map(list, zip(*total_action_history))) # Transpose matrix
    n_cols = len(total_action_history) # Number of columns
    
    proportions = {} # Stores proportions of each action in each column
    
    for col in range(n_cols):
    
        proportions[col] = {}
        
        for row in total_action_history[col]:
        
            if row in proportions[col]:
            
                proportions[col][row] += 1
                
            else:
            
                proportions[col][row] = 1
                
    for col in range(n_cols): # Divide by number of columns
    
        for key in proportions[col]:
        
            proportions[col][key] /= len(total_action_history[col])

    return proportions

"""
FIGURE GENERATION
"""

def generate_figure():

    n_cols = len(simulate_and_store_actions_GA()[0]) # Number of columns
    data_GA = obtain_probabilities(simulate_and_store_actions_GA()) # Gather GA data
    data_TS = obtain_probabilities(simulate_and_store_actions_TS()) # Gather TS data
    
    fig, (ax1, ax2) = plt.subplots(1, 2) # Figure with two plots

    for number, color in zip([0, 1, 2], ['r', 'b', 'g']):
    
        y_values = [data_GA[col][number] if number in data_GA[col] else 0 for col in range(n_cols)]
        ax1.plot(range(1, n_cols + 1), y_values, color=color, label=f"action {number + 1}")
        
    ax1.set_xlabel("time period (t)")
    ax1.set_ylabel("action probability")
    ax1.set_title("(a) greedy algorithm")
    ax1.set_yticks(np.arange(0, 1.1, 0.25))
    ax1.legend(title="variable")

    for number, color in zip([0, 1, 2], ['r', 'b', 'g']):
    
        y_values = [data_TS[col][number] if number in data_TS[col] else 0 for col in range(n_cols)]
        ax2.plot(range(1, n_cols + 1), y_values, color=color, label=f"action {number + 1}")
        
    ax2.set_xlabel("time period (t)")
    ax2.set_ylabel("action probability")
    ax2.set_title("(b) Thompson sampling")
    ax2.set_yticks(np.arange(0, 1.1, 0.25))
    ax2.legend(title="variable")

    plt.show()
