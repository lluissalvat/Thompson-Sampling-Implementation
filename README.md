# Thompson-Sampling-Implementation

In this repository we provide implementations of simple cases of a Multi-Armed Bandit Greedy Algorithm (Algorithm 3.1) and Thompson Sampling (Algorithm 3.2), as well as generate Figure 3.1, which compares the action probabilities in each of 1000 iterations based on 10000 simulations of the two algorithms. Specifically, there are three actions with mean rewards 0.9, 0.8 and 0.7. Initial distribution parameters (alpha and beta) have been set equal in order to eliminate preference and bias. Minor inexactness is to be expected in the figure generation, since these algorithms take randomness into account and thus there are no two equal simulations, by definition.

The algorithms take the number of arms and initial parameters as inputs and, for a number of iterations, estimate the model, choose which action to select according to the estimations, apply the action, observe the reward and update the parameters according to a predefined rule. Both algorithms are fundamentally the same but differ in the key aspect of model estimation. Where the greedy algorithm simply computes the expectation given the distribution parameters, Thompson sampling samples from the distribution. The latter is the better approach, as can be seen in the figure, since it ends up almost always choosing Action 1 (the one with highest mean reward) as the number of iterations becomes large enough; whereas the greedy algorithm gets stuck and is inconclusive. 

Note that if two (or more) actions are estimated to have the maximum value, both algorithms decide randomly which one to select. Also note that, due to truncation error, the same (or equivalent) division may yield a slightly different value for floating point computations, and we check that the difference between variables is very small in order to stablish equality. To compute the action probabilities at each time period, we determine the proportion of times each action has been chosen over a large number of simulations.





